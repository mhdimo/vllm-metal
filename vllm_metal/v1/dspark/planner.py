# SPDX-License-Identifier: Apache-2.0
"""Measured cost model and causal prefix planner for adaptive DSpark (M6).

The cost artifact holds step costs measured on the real pair and this
machine at a grid of active decode requests ``R`` and drafted widths ``k``:
the target forward that verifies ``R * (k + 1)`` rows (with sampling), the
batched draft backbone for ``R`` rows, and the host bookkeeping. The model
interpolates the target cost piecewise-linearly in the number of verify rows
between the two profiled request counts that bracket ``R`` (after making
each level's curve non-decreasing in rows, since verifying more rows never
costs less), the draft and host costs in ``R``, and reports whether a query
lies inside the profiled bounds; outside them the planner refuses to draft
rather than extrapolate.

The planner maximizes expected useful emitted tokens per step time for the
whole drafting batch. Every active request emits at least one token (its
correction or bonus) and needs one target input, so the batch starts at
``R`` rows and ``R`` tokens. Candidate extensions are the next position of
each request scored by its calibrated survival probability (the probability
that the prefix through that position is accepted); they are visited in
descending score with deterministic ties (earlier position first, then lower
batch index), and each admitted extension is frozen before the next
candidate is examined. The walk stops at the first extension that does not
raise the ratio (strict early stopping), or when a hard cap on rows is met.
Survival decreases along a request, so admission is prefix-closed per
request by construction. Drafting itself is a separate decision taken
before the backbone runs, from the survival the previous block predicted
for each request (or the calibration prior for a request without history):
``K=0`` is a normal outcome of that decision, not a failed request.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vllm_metal.v1.dspark.calibration import CalibrationManifest

COST_SCHEMA = "dspark-cost/1"


@dataclass(frozen=True, slots=True)
class CostSample:
    """Median step costs at one profiled (requests, width) cell."""

    requests: int
    width: int
    rows: int
    target_ms: float
    draft_ms: float
    host_ms: float
    steps: int
    target_p95_ms: float

    def __post_init__(self) -> None:
        if (
            self.requests < 1
            or self.width < 0
            or self.rows != self.requests * (self.width + 1)
        ):
            raise ValueError("a cost sample needs rows = requests * (width + 1)")
        for value in (self.target_ms, self.draft_ms, self.host_ms, self.target_p95_ms):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("cost samples must be finite and non-negative")


def _interpolate(points: Sequence[tuple[float, float]], x: float) -> float:
    """Piecewise-linear interpolation on sorted points, clamped at the ends."""
    if not points:
        raise ValueError("no points to interpolate")
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        if x0 <= x <= x1:
            if x1 == x0:
                return y1
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


@dataclass(slots=True)
class CostModel:
    manifest: CalibrationManifest
    samples: list[CostSample]
    machine: dict[str, Any] = field(default_factory=dict)
    context_tokens: int = 0
    schema: str = COST_SCHEMA

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("cost model needs at least one sample")
        cells = {(sample.requests, sample.width) for sample in self.samples}
        if len(cells) != len(self.samples):
            raise ValueError("cost model has a duplicate (requests, width) cell")
        self._by_requests = {}
        for sample in sorted(self.samples, key=lambda item: (item.requests, item.rows)):
            self._by_requests.setdefault(sample.requests, []).append(sample)
        self._request_levels = sorted(self._by_requests)
        # The bypass decision needs the undrafted cost at every profiled R.
        for level in self._request_levels:
            if all(item.width != 0 for item in self._by_requests[level]):
                raise ValueError(
                    f"cost model lacks the width-0 cell for {level} requests"
                )

    _by_requests: dict[int, list[CostSample]] = field(default_factory=dict, repr=False)
    _request_levels: list[int] = field(default_factory=list, repr=False)

    @property
    def max_requests(self) -> int:
        return self._request_levels[-1]

    def max_rows(self, requests: int) -> int:
        """Largest profiled row count at or below ``requests`` requests."""
        level = max((r for r in self._request_levels if r <= requests), default=None)
        if level is None:
            return 0
        return max(item.rows for item in self._by_requests[level])

    def within_bounds(self, requests: int, rows: int) -> bool:
        if requests < 1 or requests > self.max_requests:
            return False
        return requests <= rows <= self._rows_bound(requests)

    def _rows_bound(self, requests: int) -> int:
        lower, upper = self._bracket(requests)
        # The rows-per-request envelope at both brackets, scaled to ``requests``.
        per_request = min(
            max(item.rows for item in self._by_requests[level]) / level
            for level in {lower, upper}
        )
        return int(math.floor(per_request * requests + 1e-9))

    def _bracket(self, requests: int) -> tuple[int, int]:
        levels = self._request_levels
        lower = max((r for r in levels if r <= requests), default=levels[0])
        upper = min((r for r in levels if r >= requests), default=levels[-1])
        return lower, upper

    def _target_at_level(self, level: int, rows: float) -> float:
        # Verifying more rows never costs less: a measured dip between two
        # cells is noise or a kernel-path artifact, so the curve is made
        # non-decreasing by a running maximum before interpolation.
        points = []
        running = 0.0
        for item in self._by_requests[level]:
            running = max(running, item.target_ms)
            points.append((float(item.rows), running))
        return _interpolate(points, rows)

    def target_ms(self, requests: int, rows: int) -> float:
        """Target forward with verification for ``rows`` rows over ``requests``."""
        lower, upper = self._bracket(requests)
        # Interpolate in rows per request so the shape, not the absolute row
        # count, carries across request levels.
        per_request = rows / requests
        low = self._target_at_level(lower, per_request * lower)
        if upper == lower:
            return low
        high = self._target_at_level(upper, per_request * upper)
        return low + (high - low) * (requests - lower) / (upper - lower)

    def _level_cost(self, requests: int, attribute: str) -> float:
        points = [
            (float(level), getattr(self._by_requests[level][0], attribute))
            for level in self._request_levels
        ]
        for level in self._request_levels:
            values = [getattr(item, attribute) for item in self._by_requests[level]]
            points[self._request_levels.index(level)] = (
                float(level),
                sum(values) / len(values),
            )
        return _interpolate(points, float(requests))

    def draft_ms(self, requests: int) -> float:
        return self._level_cost(requests, "draft_ms")

    def host_ms(self, requests: int) -> float:
        return self._level_cost(requests, "host_ms")

    def step_ms(self, requests: int, rows: int, *, drafted: bool) -> float:
        total = self.target_ms(requests, rows)
        if drafted:
            total += self.draft_ms(requests) + self.host_ms(requests)
        return total

    def to_json(self) -> str:
        payload = {
            "schema": self.schema,
            "manifest": self.manifest.to_dict(),
            "machine": self.machine,
            "context_tokens": self.context_tokens,
            "samples": [asdict(sample) for sample in self.samples],
        }
        return json.dumps(payload, indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str) -> CostModel:
        payload = json.loads(text)
        if payload.get("schema") != COST_SCHEMA:
            raise ValueError(
                f"unsupported cost schema {payload.get('schema')!r}; expected {COST_SCHEMA!r}"
            )
        return cls(
            manifest=CalibrationManifest.from_dict(payload["manifest"]),
            samples=[CostSample(**item) for item in payload["samples"]],
            machine=dict(payload.get("machine", {})),
            context_tokens=int(payload.get("context_tokens", 0)),
        )

    @classmethod
    def load(cls, path: Path) -> CostModel:
        return cls.from_json(path.read_text(encoding="utf-8"))

    def validate(self, manifest: CalibrationManifest) -> None:
        if self.manifest != manifest:
            raise ValueError(
                "cost model belongs to another model pair or recipe: "
                f"{self.manifest.to_dict()} != {manifest.to_dict()}"
            )


@dataclass(frozen=True, slots=True)
class Plan:
    lengths: tuple[int, ...]
    rows: int
    expected_tokens: float
    step_ms: float
    ratio: float
    target_only_ratio: float
    admitted: tuple[
        tuple[int, int], ...
    ]  # (request index, position) in admission order
    stopped_by: str


def plan_prefixes(
    survival: Sequence[Sequence[float]],
    cost: CostModel,
    *,
    max_rows: int | None = None,
    caps: Sequence[int] | None = None,
) -> Plan:
    """Causal greedy prefix allocation for one drafting batch.

    ``survival[r][j]`` is the calibrated probability that request ``r``'s
    prefix through drafted position ``j`` is accepted; ``caps[r]`` bounds the
    positions request ``r`` may use (its output/model budget). ``max_rows``
    is a hard cap on the batch's verify rows.
    """
    requests = len(survival)
    if requests == 0:
        raise ValueError("planning needs at least one request")
    limits = [
        min(len(row), caps[index] if caps is not None else len(row))
        for index, row in enumerate(survival)
    ]
    for row in survival:
        for value in row:
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("survival probabilities must lie in [0, 1]")
    bound = cost.max_rows(requests) if max_rows is None else max_rows
    bound = min(bound, cost.max_rows(requests)) if cost.max_rows(requests) else bound
    lengths = [0] * requests
    rows = requests
    tokens = float(requests)
    target_only_ms = cost.step_ms(requests, requests, drafted=False)
    target_only_ratio = requests / target_only_ms if target_only_ms > 0 else math.inf
    step_ms = cost.step_ms(requests, rows, drafted=True)
    ratio = tokens / step_ms if step_ms > 0 else math.inf
    candidates = sorted(
        (
            (-survival[index][position], position, index)
            for index in range(requests)
            for position in range(limits[index])
        ),
    )
    admitted: list[tuple[int, int]] = []
    stopped_by = "exhausted"
    for negative, position, index in candidates:
        if lengths[index] != position:
            # A skipped earlier position of this request already stopped it.
            continue
        if rows + 1 > bound:
            stopped_by = "row-cap"
            break
        score = -negative
        next_tokens = tokens + score
        next_ms = cost.step_ms(requests, rows + 1, drafted=True)
        next_ratio = next_tokens / next_ms if next_ms > 0 else math.inf
        if not next_ratio > ratio:
            stopped_by = "no-improvement"
            break
        lengths[index] = position + 1
        rows += 1
        tokens = next_tokens
        step_ms = next_ms
        ratio = next_ratio
        admitted.append((index, position))
    return Plan(
        lengths=tuple(lengths),
        rows=rows,
        expected_tokens=tokens,
        step_ms=step_ms,
        ratio=ratio,
        target_only_ratio=target_only_ratio,
        admitted=tuple(admitted),
        stopped_by=stopped_by,
    )


@dataclass(frozen=True, slots=True)
class DraftDecision:
    draft: bool
    reason: str
    plan: Plan | None = None


def should_draft(
    expected_survival: Sequence[Sequence[float]],
    cost: CostModel,
    *,
    caps: Sequence[int] | None = None,
) -> DraftDecision:
    """Decide before the backbone runs whether drafting this batch pays.

    ``expected_survival`` is what is known before sampling this block: the
    survival the previous block predicted for each request, or the
    calibration prior for a request without history. Drafting is admitted
    only when the planned ratio strictly exceeds the target-only ratio and
    the batch lies inside the profiled cost bounds.
    """
    requests = len(expected_survival)
    if requests == 0:
        return DraftDecision(False, "no-requests")
    if not cost.within_bounds(requests, requests):
        return DraftDecision(False, "outside-cost-bounds")
    plan = plan_prefixes(expected_survival, cost, caps=caps)
    if plan.rows == requests:
        return DraftDecision(False, "planner-empty", plan)
    if not plan.ratio > plan.target_only_ratio:
        return DraftDecision(False, "target-only-faster", plan)
    return DraftDecision(True, "ok", plan)


__all__ = [
    "COST_SCHEMA",
    "CostModel",
    "CostSample",
    "DraftDecision",
    "Plan",
    "plan_prefixes",
    "should_draft",
]
