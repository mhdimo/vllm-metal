# SPDX-License-Identifier: Apache-2.0
"""Adaptive DSpark serving mode: calibrated planning, bypass and counters (M6b).

``AdaptivePlanner`` binds a calibration artifact and a cost model to the
proposer. Before the draft backbone runs it decides whether drafting this
batch pays, from the survival each request's previous block predicted (or
the calibration prior for a request without history) and the measured
costs; after the backbone it calibrates the fresh confidence logits and
allocates a causal prefix per request. History is per request generation and
is reset when the request is released (finish, cancel, preemption) or when
it skips a step without a draft, so a stale block never speaks for a resumed
or idle request. ``DSparkCounters`` accumulates the counters the specification
asks for without per-step logging.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vllm_metal.v1.dspark.calibration import (
    CalibrationArtifact,
    CalibrationManifest,
    calibrated_survival,
)
from vllm_metal.v1.dspark.planner import (
    CostModel,
    DraftDecision,
    plan_prefixes,
    should_draft,
)

MODES = ("fixed", "adaptive", "bypass")


@dataclass
class DSparkCounters:
    """Cumulative DSpark accounting (cheap integers, read by tools and logs)."""

    steps: int = 0
    drafting_steps: int = 0
    bypass_reasons: Counter = field(default_factory=Counter)
    proposed_tokens: int = 0
    scheduled_tokens: int = 0
    accepted_tokens: int = 0
    verified_requests: int = 0
    correction_or_bonus_tokens: int = 0
    position_opportunities: Counter = field(default_factory=Counter)
    position_acceptances: Counter = field(default_factory=Counter)
    planner_lengths: Counter = field(default_factory=Counter)
    planner_predicted_ratio: float = 0.0
    planner_target_only_ratio: float = 0.0

    def observe_outcome(self, scheduled: int, accepted: int) -> None:
        self.scheduled_tokens += scheduled
        self.accepted_tokens += accepted
        self.verified_requests += 1
        self.correction_or_bonus_tokens += 1
        for position in range(scheduled):
            self.position_opportunities[position] += 1
        for position in range(accepted):
            self.position_acceptances[position] += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "drafting_steps": self.drafting_steps,
            "bypass_reasons": dict(self.bypass_reasons),
            "proposed_tokens": self.proposed_tokens,
            "scheduled_tokens": self.scheduled_tokens,
            "accepted_tokens": self.accepted_tokens,
            "verified_requests": self.verified_requests,
            "correction_or_bonus_tokens": self.correction_or_bonus_tokens,
            "position_opportunities": dict(sorted(self.position_opportunities.items())),
            "position_acceptances": dict(sorted(self.position_acceptances.items())),
            "planner_lengths": dict(sorted(self.planner_lengths.items())),
            "planner_predicted_ratio": self.planner_predicted_ratio,
            "planner_target_only_ratio": self.planner_target_only_ratio,
        }


def prior_survival(artifact: CalibrationArtifact, mode: str) -> list[float]:
    """Mean survival label per position on the calibration split."""
    item = artifact.modes[mode]
    rows = item.metrics.get("calibration", {}).get("after", [])
    prior = []
    for position in range(len(item.temperatures)):
        value = None
        if position < len(rows):
            value = rows[position].get("label_mean")
        if value is None:
            # No recorded observation: assume the prefix never survives here.
            value = 0.0
        prior.append(float(value))
    return prior


class AdaptivePlanner:
    def __init__(self, calibration: CalibrationArtifact, cost: CostModel) -> None:
        self.calibration = calibration
        self.cost = cost
        self._temperatures = {
            mode: item.temperatures for mode, item in calibration.modes.items()
        }
        self._prior = {
            mode: prior_survival(calibration, mode) for mode in calibration.modes
        }
        # Survival the previous block predicted per request (its identity too).
        self._expected: dict[str, tuple[Any, list[float]]] = {}

    def temperatures(self, mode: str) -> list[float]:
        try:
            return self._temperatures[mode]
        except KeyError as error:
            raise ValueError(f"calibration artifact has no {mode!r} mode") from error

    def expected_survival(
        self, req_id: str, owner: Any, mode: str, cap: int
    ) -> list[float]:
        history = self._expected.get(req_id)
        if history is not None and history[0] is owner:
            return list(history[1][:cap])
        return list(self._prior[mode][:cap])

    def decide(
        self,
        candidates: Sequence[tuple[str, Any, str, int]],
        *,
        active_requests: int,
        context: int,
    ) -> DraftDecision:
        """Draft-or-not before the backbone: ``(req_id, owner, mode, cap)`` per candidate.

        ``active_requests`` counts every request the next target step verifies
        or decodes, drafted or not; undrafted ones contribute one row each.
        ``context`` is the batch's mean decode context in tokens.
        """
        rows: list[list[float]] = [
            self.expected_survival(req_id, owner, mode, cap)
            for req_id, owner, mode, cap in candidates
        ]
        rows.extend([] for _ in range(max(0, active_requests - len(candidates))))
        caps = [cap for _, _, _, cap in candidates] + [0] * (
            len(rows) - len(candidates)
        )
        return should_draft(rows, self.cost, context=context, caps=caps)

    def allocate(
        self,
        candidates: Sequence[tuple[str, Any, str, int, Sequence[float]]],
        *,
        active_requests: int,
        context: int,
    ) -> list[int]:
        """Prefix length per candidate ``(req_id, owner, mode, cap, confidence)``."""
        rows: list[list[float]] = []
        for req_id, owner, mode, cap, confidence in candidates:
            survival = calibrated_survival([list(confidence)], self.temperatures(mode))[
                0
            ]
            values = [float(v) for v in survival[:cap]]
            self._expected[req_id] = (owner, values)
            rows.append(values)
        rows.extend([] for _ in range(max(0, active_requests - len(candidates))))
        caps = [cap for _, _, _, cap, _ in candidates] + [0] * (
            len(rows) - len(candidates)
        )
        plan = plan_prefixes(rows, self.cost, context=context, caps=caps)
        self.last_plan = plan
        return list(plan.lengths[: len(candidates)])

    def forget(self, req_ids: Sequence[str]) -> None:
        for req_id in req_ids:
            self._expected.pop(req_id, None)

    def forget_idle(self, scheduled: Sequence[str], drafted: Sequence[str]) -> None:
        """A scheduled request that was not drafted this step loses its history."""
        drafted_set = set(drafted)
        for req_id in scheduled:
            if req_id not in drafted_set:
                self._expected.pop(req_id, None)


def load_adaptive(
    manifest: CalibrationManifest, calibration_path: str, cost_path: str
) -> AdaptivePlanner:
    """Load and validate both artifacts for ``manifest``; explicit failure otherwise."""
    if not calibration_path or not cost_path:
        raise ValueError(
            "VLLM_METAL_DSPARK_MODE=adaptive needs VLLM_METAL_DSPARK_CALIBRATION and "
            "VLLM_METAL_DSPARK_COST_MODEL; set both, or select the fixed mode"
        )
    for name, path in (("calibration", calibration_path), ("cost model", cost_path)):
        if not Path(path).is_file():
            raise ValueError(
                f"DSpark adaptive mode: {name} artifact {path!r} does not exist"
            )
    calibration = CalibrationArtifact.load(Path(calibration_path))
    calibration.validate(manifest)
    cost = CostModel.load(Path(cost_path))
    cost.validate(manifest)
    return AdaptivePlanner(calibration, cost)


__all__ = [
    "MODES",
    "AdaptivePlanner",
    "DSparkCounters",
    "load_adaptive",
    "prior_survival",
]
