# SPDX-License-Identifier: Apache-2.0
"""Confidence recording and sequential temperature scaling for DSpark.

The drafter's confidence head emits one raw logit ``z[k]`` per block position;
``sigmoid(z[k])`` estimates the probability that draft ``k`` is accepted given
that every earlier draft was, so the cumulative product is the survival
probability of the length-``k+1`` prefix. Serving decisions (M6) must not use
those estimates uncalibrated. This module owns:

- the recorder: one sample per verified proposal with the raw logits of the
  scheduled positions and their survival labels (``1`` while the accepted
  prefix continues, ``0`` from the first rejection on); positions beyond the
  scheduled width are censored rather than labelled, and a request that
  finished during verification records nothing (positions past an EOS or an
  output limit are unobservable);
- sequential temperature scaling (STS): one positive temperature per position,
  fitted in position order on a fixed grid by the binary cross-entropy of the
  cumulative survival prediction with every earlier temperature held fixed;
- reliability metrics (expected calibration error over equal-width bins,
  Brier score, per-bin averages) with a bootstrap interval for the ECE;
- the calibration artifact: temperatures with the grid, bin definition,
  objective, sample counts, dataset revision and split, per-position metrics
  before and after fitting on the calibration and holdout splits, and the
  model-pair manifest it belongs to. Loading rejects a wrong block size,
  non-finite or non-positive temperatures and a manifest mismatch.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

CALIBRATION_SCHEMA = "dspark-confidence-calibration/1"
DEFAULT_GRID = tuple(float(f"{2 ** (i / 4):.6g}") for i in range(-8, 13))  # 0.25 .. 8
DEFAULT_BINS = 15
OBJECTIVE = (
    "binary cross-entropy of the cumulative survival prediction "
    "prod_{i<=k} sigmoid(z_i / T_i) against the survival label of position k, "
    "fitted position by position with earlier temperatures fixed"
)


@dataclass(frozen=True, slots=True)
class ConfidenceSample:
    """One verified proposal: raw logits and survival labels per position."""

    mode: str
    logits: tuple[float, ...]
    survival: tuple[int, ...]
    scheduled: int
    accepted: int
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if len(self.logits) != self.scheduled or len(self.survival) != self.scheduled:
            raise ValueError(
                "a sample carries one logit and one label per scheduled position"
            )
        if not 0 <= self.accepted <= self.scheduled:
            raise ValueError("accepted drafts must lie within the scheduled width")


def survival_labels(scheduled: int, accepted: int) -> tuple[int, ...]:
    """``1`` for every position inside the accepted prefix, ``0`` after."""
    if not 0 <= accepted <= scheduled:
        raise ValueError("accepted drafts must lie within the scheduled width")
    return tuple(1 if position < accepted else 0 for position in range(scheduled))


class ConfidenceRecorder:
    """Collect samples in memory and append them as JSON lines."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.samples: list[ConfidenceSample] = []

    def observe(
        self,
        *,
        mode: str,
        logits: Sequence[float],
        scheduled: int,
        accepted: int,
        temperature: float = 0.0,
    ) -> ConfidenceSample | None:
        """Record a verified proposal; ``scheduled`` may be below the proposed width."""
        if scheduled <= 0:
            return None
        values = tuple(float(value) for value in logits[:scheduled])
        sample = ConfidenceSample(
            mode=mode,
            logits=values,
            survival=survival_labels(scheduled, accepted),
            scheduled=scheduled,
            accepted=accepted,
            temperature=float(temperature),
        )
        self.samples.append(sample)
        return sample

    def flush(self) -> int:
        if self.path is None or not self.samples:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for sample in self.samples:
                handle.write(json.dumps(asdict(sample)) + "\n")
        count = len(self.samples)
        self.samples.clear()
        return count

    @staticmethod
    def load(path: Path) -> list[ConfidenceSample]:
        samples = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    samples.append(
                        ConfidenceSample(
                            mode=item["mode"],
                            logits=tuple(item["logits"]),
                            survival=tuple(item["survival"]),
                            scheduled=int(item["scheduled"]),
                            accepted=int(item["accepted"]),
                            temperature=float(item.get("temperature", 0.0)),
                        )
                    )
        return samples


def sample_matrices(
    samples: Sequence[ConfidenceSample], block_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(logits, survival, valid)`` as ``[N, block_size]`` arrays; censored cells invalid."""
    count = len(samples)
    logits = np.zeros((count, block_size), dtype=np.float64)
    survival = np.zeros((count, block_size), dtype=np.float64)
    valid = np.zeros((count, block_size), dtype=bool)
    for row, sample in enumerate(samples):
        width = min(sample.scheduled, block_size)
        logits[row, :width] = sample.logits[:width]
        survival[row, :width] = sample.survival[:width]
        valid[row, :width] = True
    return logits, survival, valid


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=np.float64)))


def survival_predictions(
    logits: np.ndarray, temperatures: Sequence[float]
) -> np.ndarray:
    """Cumulative survival ``prod_{i<=k} sigmoid(z_i / T_i)`` per row."""
    scale = np.asarray(temperatures, dtype=np.float64)
    if scale.shape != (logits.shape[1],):
        raise ValueError("one temperature per block position is required")
    return np.cumprod(sigmoid(logits / scale[None, :]), axis=1)


def binary_cross_entropy(pred: np.ndarray, label: np.ndarray) -> float:
    eps = 1e-12
    clipped = np.clip(pred, eps, 1.0 - eps)
    return float(
        -np.mean(label * np.log(clipped) + (1.0 - label) * np.log(1.0 - clipped))
    )


def fit_sequential_temperatures(
    logits: np.ndarray,
    survival: np.ndarray,
    valid: np.ndarray,
    *,
    grid: Sequence[float] = DEFAULT_GRID,
) -> list[float]:
    """STS: per-position grid search on the cumulative survival objective."""
    if any(not math.isfinite(value) or value <= 0.0 for value in grid):
        raise ValueError("the temperature grid must be positive and finite")
    block_size = logits.shape[1]
    temperatures: list[float] = []
    for position in range(block_size):
        rows = valid[:, position]
        if not rows.any():
            # No observation at this depth: keep the raw logit (temperature 1).
            temperatures.append(1.0)
            continue
        best_value, best_loss = 1.0, math.inf
        for candidate in grid:
            trial = [*temperatures, candidate]
            pred = survival_predictions(logits[rows][:, : position + 1], trial)[:, -1]
            loss = binary_cross_entropy(pred, survival[rows, position])
            if loss < best_loss - 1e-12:
                best_value, best_loss = float(candidate), loss
        temperatures.append(best_value)
    return temperatures


def reliability(
    pred: np.ndarray, label: np.ndarray, *, bins: int = DEFAULT_BINS
) -> dict[str, Any]:
    """ECE, Brier and per-bin averages over equal-width probability bins."""
    pred = np.asarray(pred, dtype=np.float64)
    label = np.asarray(label, dtype=np.float64)
    count = pred.shape[0]
    if count == 0:
        return {"count": 0, "ece": None, "brier": None, "bins": []}
    index = np.minimum((pred * bins).astype(int), bins - 1)
    table = []
    ece = 0.0
    for bin_index in range(bins):
        members = index == bin_index
        weight = int(members.sum())
        if weight == 0:
            continue
        avg_pred = float(pred[members].mean())
        avg_label = float(label[members].mean())
        ece += abs(avg_pred - avg_label) * weight / count
        table.append(
            {
                "bin": bin_index,
                "range": [bin_index / bins, (bin_index + 1) / bins],
                "count": weight,
                "avg_pred": avg_pred,
                "avg_label": avg_label,
            }
        )
    return {
        "count": count,
        "ece": float(ece),
        "brier": float(np.mean((pred - label) ** 2)),
        "pred_mean": float(pred.mean()),
        "label_mean": float(label.mean()),
        "bins": table,
    }


def bootstrap_ece_interval(
    pred: np.ndarray,
    label: np.ndarray,
    *,
    bins: int = DEFAULT_BINS,
    resamples: int = 500,
    seed: int = 0,
) -> tuple[float, float] | None:
    count = pred.shape[0]
    if count == 0:
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        rows = rng.integers(0, count, count)
        values.append(reliability(pred[rows], label[rows], bins=bins)["ece"])
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def per_position_metrics(
    logits: np.ndarray,
    survival: np.ndarray,
    valid: np.ndarray,
    temperatures: Sequence[float],
    *,
    bins: int = DEFAULT_BINS,
    bootstrap: bool = False,
) -> list[dict[str, Any]]:
    pred = survival_predictions(logits, temperatures)
    rows = []
    for position in range(logits.shape[1]):
        mask = valid[:, position]
        entry = reliability(pred[mask, position], survival[mask, position], bins=bins)
        entry["position"] = position
        if bootstrap and mask.any():
            entry["ece_ci95"] = bootstrap_ece_interval(
                pred[mask, position], survival[mask, position], bins=bins
            )
        rows.append(entry)
    return rows


@dataclass(frozen=True, slots=True)
class CalibrationManifest:
    """Identity the artifact is valid for."""

    target: str
    target_revision: str | None
    draft: str
    draft_revision: str | None
    block_size: int
    target_layer_ids: tuple[int, ...]
    markov_rank: int
    confidence_head_with_markov: bool
    draft_dtype: str

    def to_dict(self) -> dict[str, Any]:
        item = asdict(self)
        item["target_layer_ids"] = list(self.target_layer_ids)
        return item

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> CalibrationManifest:
        return cls(
            target=str(item["target"]),
            target_revision=item.get("target_revision"),
            draft=str(item["draft"]),
            draft_revision=item.get("draft_revision"),
            block_size=int(item["block_size"]),
            target_layer_ids=tuple(int(v) for v in item["target_layer_ids"]),
            markov_rank=int(item["markov_rank"]),
            confidence_head_with_markov=bool(item["confidence_head_with_markov"]),
            draft_dtype=str(item["draft_dtype"]),
        )

    @classmethod
    def from_runner(cls, runner: Any) -> CalibrationManifest:
        """The manifest of the pair a loaded DSpark runner serves."""
        config = runner.vllm_config
        spec = config.speculative_config
        proposer = runner._drafter
        draft_config = proposer._config
        return cls(
            target=str(config.model_config.model),
            target_revision=config.model_config.revision,
            draft=str(spec.draft_model_config.model),
            draft_revision=spec.draft_model_config.revision,
            block_size=int(draft_config.block_size),
            target_layer_ids=tuple(int(v) for v in draft_config.target_layer_ids),
            markov_rank=int(draft_config.markov_rank),
            confidence_head_with_markov=bool(draft_config.confidence_head_with_markov),
            draft_dtype=str(proposer._drafter.hidden_norm.weight.dtype),
        )


@dataclass(slots=True)
class ModeCalibration:
    """Temperatures and evidence for one sampling mode."""

    mode: str
    temperatures: list[float]
    sample_counts: list[int]
    metrics: dict[str, Any] = field(default_factory=dict)
    recorded_sampling: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CalibrationArtifact:
    manifest: CalibrationManifest
    modes: dict[str, ModeCalibration]
    grid: list[float]
    bins: int
    dataset: dict[str, Any]
    objective: str = OBJECTIVE
    schema: str = CALIBRATION_SCHEMA

    def to_json(self) -> str:
        payload = {
            "schema": self.schema,
            "manifest": self.manifest.to_dict(),
            "grid": list(self.grid),
            "bins": self.bins,
            "objective": self.objective,
            "dataset": self.dataset,
            "modes": {
                name: {
                    "mode": item.mode,
                    "temperatures": list(item.temperatures),
                    "sample_counts": list(item.sample_counts),
                    "metrics": item.metrics,
                    "recorded_sampling": item.recorded_sampling,
                }
                for name, item in self.modes.items()
            },
        }
        return json.dumps(payload, indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str) -> CalibrationArtifact:
        payload = json.loads(text)
        if payload.get("schema") != CALIBRATION_SCHEMA:
            raise ValueError(
                f"unsupported calibration schema {payload.get('schema')!r}; "
                f"expected {CALIBRATION_SCHEMA!r}"
            )
        modes = {
            name: ModeCalibration(
                mode=str(item["mode"]),
                temperatures=[float(v) for v in item["temperatures"]],
                sample_counts=[int(v) for v in item["sample_counts"]],
                metrics=dict(item.get("metrics", {})),
                recorded_sampling=dict(item.get("recorded_sampling", {})),
            )
            for name, item in payload["modes"].items()
        }
        return cls(
            manifest=CalibrationManifest.from_dict(payload["manifest"]),
            modes=modes,
            grid=[float(v) for v in payload["grid"]],
            bins=int(payload["bins"]),
            dataset=dict(payload.get("dataset", {})),
            objective=str(payload.get("objective", OBJECTIVE)),
        )

    @classmethod
    def load(cls, path: Path) -> CalibrationArtifact:
        return cls.from_json(path.read_text(encoding="utf-8"))

    def validate(self, manifest: CalibrationManifest) -> None:
        """Reject a wrong block size, bad temperatures or another model pair."""
        if not self.modes:
            raise ValueError("calibration artifact carries no sampling mode")
        for name, item in self.modes.items():
            if len(item.temperatures) != manifest.block_size:
                raise ValueError(
                    f"calibration mode {name!r} has {len(item.temperatures)} temperatures "
                    f"for block size {manifest.block_size}"
                )
            if any(
                not math.isfinite(value) or value <= 0.0 for value in item.temperatures
            ):
                raise ValueError(
                    f"calibration mode {name!r} has a non-finite or non-positive temperature"
                )
        if self.manifest != manifest:
            raise ValueError(
                "calibration artifact belongs to another model pair or recipe: "
                f"{self.manifest.to_dict()} != {manifest.to_dict()}"
            )

    def temperatures_for(self, mode: str) -> list[float]:
        item = self.modes.get(mode)
        if item is None:
            raise KeyError(f"calibration artifact has no {mode!r} mode")
        return list(item.temperatures)


def calibrated_survival(
    logits: Sequence[Sequence[float]] | np.ndarray, temperatures: Sequence[float]
) -> np.ndarray:
    """Calibrated cumulative survival for raw logit rows (planner input)."""
    array = np.asarray(logits, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("logits must be [rows, positions]")
    width = array.shape[1]
    return survival_predictions(array, list(temperatures)[:width])


__all__ = [
    "CALIBRATION_SCHEMA",
    "DEFAULT_BINS",
    "DEFAULT_GRID",
    "OBJECTIVE",
    "CalibrationArtifact",
    "CalibrationManifest",
    "ConfidenceRecorder",
    "ConfidenceSample",
    "ModeCalibration",
    "binary_cross_entropy",
    "bootstrap_ece_interval",
    "calibrated_survival",
    "fit_sequential_temperatures",
    "per_position_metrics",
    "reliability",
    "sample_matrices",
    "sigmoid",
    "survival_labels",
    "survival_predictions",
]
