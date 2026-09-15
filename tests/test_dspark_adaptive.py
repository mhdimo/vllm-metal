# SPDX-License-Identifier: Apache-2.0
"""Adaptive DSpark mode: planner binding, bypass, prefix cuts, counters, loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_dspark_contracts import (
    _loadable_runner,
    draft_hf_config,
    dspark_config,  # noqa: F401  (pytest fixture)
)
from tests.test_dspark_proposer import (
    _assert_context,
    _context,
    _proposer,
    _seed,
    _state,
    _stochastic_state,
)
from vllm_metal.v1.dspark.adaptive import (
    AdaptivePlanner,
    DSparkCounters,
    prior_survival,
)
from vllm_metal.v1.dspark.calibration import (
    CalibrationArtifact,
    CalibrationManifest,
    ModeCalibration,
)
from vllm_metal.v1.dspark.config import DSparkConfig
from vllm_metal.v1.dspark.planner import CostModel, CostSample


def _manifest(proposer=None) -> CalibrationManifest:
    config = DSparkConfig.from_dict(draft_hf_config().to_dict())
    return CalibrationManifest(
        target="resolved-target",
        target_revision=None,
        draft="resolved-draft",
        draft_revision=None,
        block_size=config.block_size,
        target_layer_ids=tuple(config.target_layer_ids),
        markov_rank=config.markov_rank,
        confidence_head_with_markov=config.confidence_head_with_markov,
        draft_dtype="mlx.core.float32",
    )


def _artifact(manifest: CalibrationManifest, prior: float = 0.9) -> CalibrationArtifact:
    width = manifest.block_size
    metrics = {
        "calibration": {
            "after": [
                {"position": k, "label_mean": prior ** (k + 1), "count": 10}
                for k in range(width)
            ]
        }
    }
    modes = {
        mode: ModeCalibration(
            mode=mode,
            temperatures=[1.0] * width,
            sample_counts=[10] * width,
            metrics=metrics,
        )
        for mode in ("greedy", "stochastic")
    }
    return CalibrationArtifact(
        manifest=manifest, modes=modes, grid=[1.0], bins=15, dataset={"records": []}
    )


def _cost(manifest: CalibrationManifest, *, draft_ms: float = 1.0) -> CostModel:
    samples = []
    for requests in (1, 2, 4):
        for width in range(manifest.block_size + 1):
            rows = requests * (width + 1)
            samples.append(
                CostSample(
                    requests=requests,
                    width=width,
                    rows=rows,
                    context=512,
                    step_ms=6.0 + 0.4 * rows + ((draft_ms + 0.2) if width else 0.0),
                    target_ms=6.0 + 0.4 * rows,
                    draft_ms=draft_ms,
                    host_ms=0.2,
                    steps=20,
                    step_p95_ms=7.0 + 0.4 * rows + ((draft_ms + 0.2) if width else 0.0),
                )
            )
    return CostModel(manifest=manifest, samples=samples)


def _planner(prior: float = 0.9, draft_ms: float = 1.0) -> AdaptivePlanner:
    manifest = _manifest()
    return AdaptivePlanner(
        _artifact(manifest, prior), _cost(manifest, draft_ms=draft_ms)
    )


def test_prior_survival_reads_calibration_means_and_defaults_to_zero() -> None:
    manifest = _manifest()
    artifact = _artifact(manifest, 0.8)
    prior = prior_survival(artifact, "greedy")
    assert len(prior) == manifest.block_size
    assert prior[0] == pytest.approx(0.8) and prior[1] == pytest.approx(0.64)
    artifact.modes["greedy"].metrics = {}
    assert prior_survival(artifact, "greedy") == [0.0] * manifest.block_size


def test_decision_uses_history_for_the_same_generation_only() -> None:
    planner = _planner(prior=0.9)
    owner = object()
    assert planner.expected_survival("r", owner, "greedy", 3) == pytest.approx(
        [0.9, 0.81, 0.729]
    )
    planner.allocate(
        [("r", owner, "greedy", 3, [4.0, -4.0, 0.0])], active_requests=1, context=64
    )
    history = planner.expected_survival("r", owner, "greedy", 3)
    assert history[0] > 0.98 and history[1] < 0.02  # calibrated survival of the block
    assert planner.expected_survival("r", object(), "greedy", 3)[0] == pytest.approx(
        0.9
    )
    planner.forget(["r"])
    assert planner.expected_survival("r", owner, "greedy", 3)[0] == pytest.approx(0.9)
    planner.allocate(
        [("r", owner, "greedy", 3, [4.0, 4.0, 4.0])], active_requests=1, context=64
    )
    planner.forget_idle(["r", "s"], ["s"])
    assert planner.expected_survival("r", owner, "greedy", 3)[0] == pytest.approx(0.9)


def test_decision_and_allocation_follow_costs() -> None:
    cheap = _planner(prior=0.9, draft_ms=1.0)
    decision = cheap.decide(
        [("r", object(), "greedy", 4)], active_requests=1, context=64
    )
    assert decision.draft and decision.reason == "ok"
    pricey = _planner(prior=0.9, draft_ms=60.0)
    decision = pricey.decide(
        [("r", object(), "greedy", 4)], active_requests=1, context=64
    )
    assert not decision.draft and decision.reason == "target-only-faster"
    hopeless = _planner(prior=0.05)
    assert hopeless.decide(
        [("r", object(), "greedy", 4)], active_requests=1, context=64
    ).reason in (
        "planner-empty",
        "target-only-faster",
    )
    assert cheap.decide([], active_requests=0, context=64).reason == "no-requests"
    assert cheap.decide(
        [("r", object(), "greedy", 2)], active_requests=9, context=64
    ).reason == ("outside-cost-bounds")
    # Undrafted active requests add rows without extensions.
    lengths = cheap.allocate(
        [
            ("a", object(), "greedy", 3, [5.0, 5.0, 5.0]),
            ("b", object(), "stochastic", 3, [-6.0, 0.0, 0.0]),
        ],
        active_requests=4,
        context=64,
    )
    assert lengths[0] == 3 and lengths[1] == 0
    assert cheap.last_plan.rows == 4 + 3


def test_proposer_cuts_prefixes_and_counts_outcomes() -> None:
    proposer = _proposer()
    proposer._runner.model_config.seed = 0
    proposer.adaptive = _planner(prior=0.9, draft_ms=0.5)
    state = _state([1, 2, 3])
    drafts = _seed(proposer, state, k=4)
    assert drafts is not None
    row = drafts.draft_token_ids[0]
    assert 1 <= len(row) <= 4
    record = proposer.proposals["r"]
    assert record.token_ids == list(row) and len(record.confidence) == len(row)
    assert proposer.counters.drafting_steps == 1
    assert proposer.counters.proposed_tokens == len(row)
    assert proposer.counters.planner_lengths[len(row)] == 1
    # Verification accepted one draft: the counters see width and acceptance.
    accepted = 1 if len(row) >= 1 else 0
    outputs = row[:accepted] + [9]
    state.token_ids.extend(outputs)
    proposer.propose(_context(decode=[("r", state, 2, [3, *row], outputs)], k=4))
    counters = proposer.counters.snapshot()
    assert counters["scheduled_tokens"] == len(row)
    assert counters["accepted_tokens"] == accepted
    assert counters["verified_requests"] == 1
    assert counters["position_opportunities"][0] == 1
    assert counters["correction_or_bonus_tokens"] == 1


def test_proposer_bypasses_when_drafting_does_not_pay() -> None:
    proposer = _proposer()
    proposer._runner.model_config.seed = 0
    proposer.adaptive = _planner(prior=0.9, draft_ms=60.0)
    state = _stochastic_state([1, 2, 3])
    assert _seed(proposer, state, k=4) is None
    _assert_context(proposer, "r", [0, 1])  # the context still advanced
    assert proposer.counters.bypass_reasons == {"target-only-faster": 1}
    assert "r" not in proposer.proposals
    proposer.release_requests({"r"})
    assert "r" not in proposer.adaptive._expected


def test_runner_loads_adaptive_mode_only_with_matching_artifacts(
    request, monkeypatch, tmp_path: Path
) -> None:
    config = request.getfixturevalue("dspark_config")
    runner = _loadable_runner(config, monkeypatch)
    monkeypatch.setenv("VLLM_METAL_DSPARK_MODE", "adaptive")
    with pytest.raises(ValueError, match="needs VLLM_METAL_DSPARK_CALIBRATION"):
        runner._load_dspark_drafter()
    manifest = CalibrationManifest.from_runner(runner)
    calibration = tmp_path / "calibration.json"
    cost = tmp_path / "cost.json"
    calibration.write_text(_artifact(manifest).to_json())
    cost.write_text(_cost(manifest).to_json())
    monkeypatch.setenv("VLLM_METAL_DSPARK_CALIBRATION", str(calibration))
    monkeypatch.setenv("VLLM_METAL_DSPARK_COST_MODEL", str(cost))
    runner = _loadable_runner(config, monkeypatch)
    runner._load_dspark_drafter()
    assert runner._drafter.adaptive is not None
    other = tmp_path / "other.json"
    payload = json.loads(calibration.read_text())
    payload["manifest"]["draft"] = "another-draft"
    other.write_text(json.dumps(payload))
    monkeypatch.setenv("VLLM_METAL_DSPARK_CALIBRATION", str(other))
    runner = _loadable_runner(config, monkeypatch)
    with pytest.raises(ValueError, match="another model pair"):
        runner._load_dspark_drafter()
    monkeypatch.setenv("VLLM_METAL_DSPARK_MODE", "auto")
    runner = _loadable_runner(config, monkeypatch)
    with pytest.raises(ValueError, match="not one of"):
        runner._load_dspark_drafter()
    monkeypatch.setenv("VLLM_METAL_DSPARK_MODE", "fixed")
    runner = _loadable_runner(config, monkeypatch)
    runner._load_dspark_drafter()
    assert runner._drafter.adaptive is None
    assert isinstance(runner._drafter.counters, DSparkCounters)
    monkeypatch.setenv("VLLM_METAL_DSPARK_MODE", "bypass")
    runner = _loadable_runner(config, monkeypatch)
    runner._load_dspark_drafter()
    assert runner._drafter.adaptive is None and runner._drafter.bypass_only


def test_bypass_mode_advances_contexts_without_drafting() -> None:
    proposer = _proposer()
    proposer.bypass_only = True
    state = _state([1, 2, 3])
    assert _seed(proposer, state, k=4) is None
    _assert_context(proposer, "r", [0, 1])
    assert proposer.counters.bypass_reasons == {"mode-bypass": 1}
