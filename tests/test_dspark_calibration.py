# SPDX-License-Identifier: Apache-2.0
"""Confidence recording, sequential temperature scaling and the artifact."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vllm_metal.v1.dspark.calibration import (
    CALIBRATION_SCHEMA,
    CalibrationArtifact,
    CalibrationManifest,
    ConfidenceRecorder,
    ConfidenceSample,
    ModeCalibration,
    calibrated_survival,
    fit_sequential_temperatures,
    per_position_metrics,
    reliability,
    sample_matrices,
    sigmoid,
    survival_labels,
    survival_predictions,
)


def _manifest(**changes) -> CalibrationManifest:
    base = CalibrationManifest(
        target="target",
        target_revision="t1",
        draft="draft",
        draft_revision="d1",
        block_size=3,
        target_layer_ids=(1, 9, 17),
        markov_rank=256,
        confidence_head_with_markov=True,
        draft_dtype="bfloat16",
    )
    return replace(base, **changes)


def _synthetic(rng, count: int, temperatures: list[float], width: int = 3):
    """Proposals whose conditional acceptance is sigmoid(z / T*) exactly."""
    logits = rng.normal(0.0, 3.0, size=(count, width))
    accepted = np.zeros(count, dtype=int)
    for row in range(count):
        for position in range(width):
            probability = sigmoid(logits[row, position] / temperatures[position])
            if rng.random() < probability:
                accepted[row] += 1
            else:
                break
    return [
        ConfidenceSample(
            mode="greedy",
            logits=tuple(float(v) for v in logits[row]),
            survival=survival_labels(width, int(accepted[row])),
            scheduled=width,
            accepted=int(accepted[row]),
        )
        for row in range(count)
    ]


def test_survival_labels_and_censoring() -> None:
    assert survival_labels(4, 0) == (0, 0, 0, 0)
    assert survival_labels(4, 2) == (1, 1, 0, 0)
    assert survival_labels(4, 4) == (1, 1, 1, 1)
    with pytest.raises(ValueError):
        survival_labels(3, 4)
    recorder = ConfidenceRecorder()
    # The scheduler clipped a four-position proposal to two: only two positions
    # are observed, the rest is censored, never labelled negative.
    sample = recorder.observe(
        mode="greedy", logits=[0.5, 1.0, 2.0, 3.0], scheduled=2, accepted=1
    )
    assert sample is not None and sample.logits == (0.5, 1.0)
    assert sample.survival == (1, 0)
    assert recorder.observe(mode="greedy", logits=[], scheduled=0, accepted=0) is None
    logits, survival, valid = sample_matrices(recorder.samples, block_size=4)
    assert valid.tolist() == [[True, True, False, False]]
    assert survival.tolist() == [[1.0, 0.0, 0.0, 0.0]]
    assert logits[0, 2] == 0.0


def test_recorder_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    recorder = ConfidenceRecorder(path)
    recorder.observe(
        mode="stochastic", logits=[1.0, -1.0], scheduled=2, accepted=2, temperature=0.8
    )
    recorder.observe(
        mode="stochastic", logits=[0.0, 0.0], scheduled=2, accepted=0, temperature=0.8
    )
    assert recorder.flush() == 2 and recorder.samples == []
    loaded = ConfidenceRecorder.load(path)
    assert [item.accepted for item in loaded] == [2, 0]
    assert loaded[0].temperature == 0.8 and loaded[0].survival == (1, 1)


def test_sts_recovers_known_temperatures_sequentially() -> None:
    rng = np.random.default_rng(3)
    truth = [2.0, 0.5, 1.0]
    samples = _synthetic(rng, 6000, truth)
    logits, survival, valid = sample_matrices(samples, block_size=3)
    fitted = fit_sequential_temperatures(logits, survival, valid)
    for value, expected in zip(fitted, truth, strict=True):
        assert expected / 1.25 <= value <= expected * 1.25
    # The grid is the only admissible set of values.
    grid = [0.5, 1.0, 2.0]
    coarse = fit_sequential_temperatures(logits, survival, valid, grid=grid)
    assert all(value in grid for value in coarse)
    # Earlier temperatures stay fixed: refitting the first position alone gives
    # the same value the sequential fit chose.
    first = fit_sequential_temperatures(logits[:, :1], survival[:, :1], valid[:, :1])
    assert first[0] == fitted[0]
    # Calibration lowers the objective on the positions it touched.
    before = per_position_metrics(logits, survival, valid, [1.0, 1.0, 1.0])
    after = per_position_metrics(logits, survival, valid, fitted)
    assert after[0]["ece"] < before[0]["ece"]
    assert after[1]["ece"] < before[1]["ece"]
    with pytest.raises(ValueError, match="positive and finite"):
        fit_sequential_temperatures(logits, survival, valid, grid=[0.0, 1.0])


def test_unobserved_positions_keep_temperature_one() -> None:
    samples = [
        ConfidenceSample(
            mode="greedy", logits=(1.0,), survival=(1,), scheduled=1, accepted=1
        )
    ]
    logits, survival, valid = sample_matrices(samples, block_size=3)
    assert fit_sequential_temperatures(logits, survival, valid, grid=[0.5, 2.0])[
        1:
    ] == [
        1.0,
        1.0,
    ]


def test_reliability_metrics_on_hand_made_data() -> None:
    pred = np.array([0.1, 0.1, 0.9, 0.9])
    label = np.array([0.0, 0.0, 1.0, 1.0])
    perfect = reliability(pred, label, bins=10)
    assert math.isclose(perfect["ece"], 0.1) and math.isclose(perfect["brier"], 0.01)
    assert [row["count"] for row in perfect["bins"]] == [2, 2]
    wrong = reliability(pred, 1.0 - label, bins=10)
    assert math.isclose(wrong["ece"], 0.9) and math.isclose(wrong["brier"], 0.81)
    assert reliability(np.array([]), np.array([]))["ece"] is None
    rows = per_position_metrics(
        np.array([[2.0, 2.0]]),
        np.array([[1.0, 0.0]]),
        np.array([[True, False]]),
        [1.0, 1.0],
    )
    assert rows[0]["count"] == 1 and rows[1]["count"] == 0


def test_survival_predictions_are_cumulative_and_calibrated() -> None:
    logits = np.array([[0.0, 0.0], [4.0, -4.0]])
    pred = survival_predictions(logits, [1.0, 1.0])
    assert np.allclose(pred[0], [0.5, 0.25])
    assert pred[1, 1] < pred[1, 0]
    scaled = calibrated_survival(logits, [2.0, 2.0, 99.0])  # extra temperatures ignored
    assert np.allclose(scaled[0], [0.5, 0.25])
    assert scaled[1, 0] == pytest.approx(sigmoid(np.array(2.0)))
    with pytest.raises(ValueError):
        survival_predictions(logits, [1.0])


def test_artifact_round_trip_and_validation(tmp_path: Path) -> None:
    manifest = _manifest()
    artifact = CalibrationArtifact(
        manifest=manifest,
        modes={
            "greedy": ModeCalibration(
                mode="greedy", temperatures=[1.5, 1.0, 0.75], sample_counts=[10, 8, 5]
            )
        },
        grid=[0.5, 1.0, 1.5],
        bins=15,
        dataset={"records": []},
    )
    path = tmp_path / "calibration.json"
    path.write_text(artifact.to_json())
    loaded = CalibrationArtifact.load(path)
    assert loaded.schema == CALIBRATION_SCHEMA
    assert loaded.manifest == manifest
    assert loaded.temperatures_for("greedy") == [1.5, 1.0, 0.75]
    loaded.validate(manifest)
    with pytest.raises(ValueError, match="another model pair"):
        loaded.validate(_manifest(draft_revision="d2"))
    with pytest.raises(ValueError, match="temperatures for block size"):
        loaded.validate(_manifest(block_size=4))
    with pytest.raises(KeyError):
        loaded.temperatures_for("stochastic")
    bad = replace(artifact)
    bad.modes = {
        "greedy": ModeCalibration(
            mode="greedy",
            temperatures=[1.0, float("nan"), 1.0],
            sample_counts=[1, 1, 1],
        )
    }
    with pytest.raises(ValueError, match="non-finite"):
        bad.validate(manifest)
    payload = json.loads(path.read_text())
    payload["schema"] = "other/9"
    with pytest.raises(ValueError, match="unsupported calibration schema"):
        CalibrationArtifact.from_json(json.dumps(payload))
