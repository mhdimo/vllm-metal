# SPDX-License-Identifier: Apache-2.0
"""Cost model interpolation and the causal prefix planner against a pure oracle."""

from __future__ import annotations

import itertools
import json
from dataclasses import replace
from pathlib import Path

import pytest

from vllm_metal.v1.dspark.calibration import CalibrationManifest
from vllm_metal.v1.dspark.planner import (
    COST_SCHEMA,
    CostModel,
    CostSample,
    plan_prefixes,
    should_draft,
)


def _manifest() -> CalibrationManifest:
    return CalibrationManifest(
        target="target",
        target_revision=None,
        draft="draft",
        draft_revision="d1",
        block_size=7,
        target_layer_ids=(1, 9, 17, 25, 33),
        markov_rank=256,
        confidence_head_with_markov=True,
        draft_dtype="bfloat16",
    )


def _sample(
    requests: int, width: int, target: float, draft: float = 2.0, host: float = 0.5
) -> CostSample:
    return CostSample(
        requests=requests,
        width=width,
        rows=requests * (width + 1),
        target_ms=target,
        draft_ms=draft,
        host_ms=host,
        steps=50,
        target_p95_ms=target * 1.1,
    )


def _convex_model(requests_levels=(1, 2, 4), widths=(0, 1, 2, 4, 7)) -> CostModel:
    """Target cost grows convexly in rows: 6 + 0.5 * rows + 0.02 * rows**2."""
    samples = []
    for requests in requests_levels:
        for width in widths:
            rows = requests * (width + 1)
            samples.append(
                _sample(
                    requests,
                    width,
                    6.0 + 0.5 * rows + 0.02 * rows * rows,
                    draft=1.0 + 0.5 * requests,
                )
            )
    return CostModel(
        manifest=_manifest(),
        samples=samples,
        machine={"chip": "test"},
        context_tokens=256,
    )


def _oracle(survival, cost, requests):
    """Best ratio over every prefix-closed allocation (small instances only)."""
    best = None
    for lengths in itertools.product(*(range(len(row) + 1) for row in survival)):
        rows = requests + sum(lengths)
        if rows > cost.max_rows(requests):
            continue
        tokens = requests + sum(
            sum(row[:length]) for row, length in zip(survival, lengths, strict=True)
        )
        ratio = tokens / cost.step_ms(requests, rows, drafted=True)
        if best is None or ratio > best[0] + 1e-12:
            best = (ratio, lengths)
    return best


def test_cost_model_interpolates_rows_and_request_levels() -> None:
    cost = _convex_model()
    # Exact at profiled cells.
    assert cost.target_ms(1, 1) == pytest.approx(6.0 + 0.5 + 0.02)
    assert cost.target_ms(4, 32) == pytest.approx(6.0 + 16.0 + 0.02 * 1024)
    # Between profiled widths at one level: linear in rows.
    low, high = cost.target_ms(1, 3), cost.target_ms(1, 5)
    assert low < cost.target_ms(1, 4) < high
    # Between request levels: interpolated at equal rows per request.
    assert cost.target_ms(2, 4) < cost.target_ms(3, 6) < cost.target_ms(4, 8)
    assert cost.draft_ms(3) == pytest.approx(2.5)
    assert cost.host_ms(2) == 0.5
    assert cost.step_ms(2, 2, drafted=False) == cost.target_ms(2, 2)
    assert cost.step_ms(2, 2, drafted=True) == pytest.approx(
        cost.target_ms(2, 2) + 2.0 + 0.5
    )
    # Bounds: profiled request levels and rows per request.
    assert cost.max_requests == 4 and cost.max_rows(3) == 16 and cost.max_rows(4) == 32
    assert cost.within_bounds(4, 32) and not cost.within_bounds(4, 33)
    assert cost.within_bounds(3, 24) and not cost.within_bounds(3, 25)
    assert not cost.within_bounds(5, 5) and not cost.within_bounds(0, 0)


def test_cost_model_rejects_bad_samples_and_round_trips(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="rows = requests"):
        CostSample(1, 2, 2, 1.0, 1.0, 1.0, 10, 1.0)
    with pytest.raises(ValueError, match="width-0"):
        CostModel(manifest=_manifest(), samples=[_sample(1, 1, 5.0)])
    with pytest.raises(ValueError, match="duplicate"):
        CostModel(
            manifest=_manifest(), samples=[_sample(1, 0, 5.0), _sample(1, 0, 6.0)]
        )
    cost = _convex_model()
    path = tmp_path / "cost.json"
    path.write_text(cost.to_json())
    loaded = CostModel.load(path)
    assert loaded.schema == COST_SCHEMA and loaded.samples == cost.samples
    assert loaded.machine == {"chip": "test"} and loaded.context_tokens == 256
    loaded.validate(_manifest())
    other = replace(_manifest(), draft_revision="d2")
    with pytest.raises(ValueError, match="another model pair"):
        loaded.validate(other)
    payload = json.loads(path.read_text())
    payload["schema"] = "x/0"
    with pytest.raises(ValueError, match="unsupported cost schema"):
        CostModel.from_json(json.dumps(payload))


@pytest.mark.parametrize("seed", range(12))
def test_planner_matches_the_oracle_on_convex_costs(seed: int) -> None:
    import random

    rng = random.Random(seed)
    cost = _convex_model()
    requests = rng.choice([1, 2, 3])
    survival = []
    for _ in range(requests):
        conditional = [rng.uniform(0.2, 0.99) for _ in range(rng.choice([2, 3, 4]))]
        row, running = [], 1.0
        for value in conditional:
            running *= value
            row.append(running)
        survival.append(row)
    plan = plan_prefixes(survival, cost)
    best_ratio, best_lengths = _oracle(survival, cost, requests)
    assert plan.ratio == pytest.approx(best_ratio)
    assert plan.rows == requests + sum(plan.lengths)
    assert plan.rows >= requests  # one target input per active request
    assert sum(plan.lengths) == sum(best_lengths)
    # Admission order is descending survival with deterministic ties.
    scores = [survival[index][position] for index, position in plan.admitted]
    assert scores == sorted(scores, reverse=True)


def test_planner_is_causal_prefix_closed_and_deterministic() -> None:
    cost = _convex_model()
    # Equal scores: the earlier position wins, then the lower batch index.
    survival = [[0.9, 0.9], [0.9, 0.9]]
    plan = plan_prefixes(survival, cost)
    assert plan.admitted[:3] == ((0, 0), (1, 0), (0, 1))
    assert all(
        plan.lengths[index] == position + 1 or plan.lengths[index] > position
        for index, position in plan.admitted
    )
    # A request whose next position was skipped never advances later.
    survival = [[0.99, 0.01, 0.01], [0.5]]
    plan = plan_prefixes(survival, cost)
    assert plan.lengths[0] in (1, 2, 3) and plan.stopped_by in (
        "no-improvement",
        "exhausted",
    )
    if plan.lengths[0] == 1:
        assert plan.stopped_by == "no-improvement"
    # Caps and the hard row cap truncate the prefix without reordering.
    capped = plan_prefixes([[0.99, 0.98, 0.97, 0.96]], cost, caps=[2])
    assert capped.lengths == (2,)
    rows = plan_prefixes([[0.99, 0.98, 0.97, 0.96]], cost, max_rows=3)
    assert rows.lengths == (2,) and rows.stopped_by == "row-cap"
    with pytest.raises(ValueError, match="at least one request"):
        plan_prefixes([], cost)
    with pytest.raises(ValueError, match="lie in"):
        plan_prefixes([[1.5]], cost)


def test_planner_stops_early_on_a_non_unimodal_curve() -> None:
    # A step in the verification cost after two rows: strict early stopping
    # keeps the two-row prefix even though five rows would pay off again.
    samples = [
        _sample(1, 0, 5.0),
        _sample(1, 1, 5.5),
        _sample(1, 2, 12.0),
        _sample(1, 4, 12.5),
        _sample(1, 7, 13.0),
    ]
    cost = CostModel(manifest=_manifest(), samples=samples)
    plan = plan_prefixes([[0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65]], cost)
    assert plan.lengths == (1,) and plan.stopped_by == "no-improvement"
    best_ratio, _ = _oracle([[0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65]], cost, 1)
    assert best_ratio > plan.ratio  # no global-optimality claim on such a curve


def test_should_draft_compares_with_target_only_and_bounds() -> None:
    cost = _convex_model()
    confident = [[0.95, 0.9, 0.85]]
    decision = should_draft(confident, cost)
    assert decision.draft and decision.reason == "ok" and decision.plan is not None
    assert decision.plan.ratio > decision.plan.target_only_ratio
    hopeless = [[0.05, 0.01]]
    decision = should_draft(hopeless, cost)
    assert not decision.draft and decision.reason in (
        "planner-empty",
        "target-only-faster",
    )
    assert should_draft([], cost).reason == "no-requests"
    assert should_draft([[0.9]] * 5, cost).reason == "outside-cost-bounds"
    # Drafting costs the backbone: with an expensive draft even a good prefix loses.
    pricey = CostModel(
        manifest=_manifest(),
        samples=[
            _sample(1, w, 6.0 + 0.5 * (w + 1), draft=40.0) for w in (0, 1, 2, 4, 7)
        ],
    )
    assert should_draft(confident, pricey).reason == "target-only-faster"


def test_cost_model_makes_measured_dips_non_decreasing() -> None:
    samples = [
        _sample(1, 0, 5.0),
        _sample(1, 1, 9.0),
        _sample(1, 2, 8.0),  # measured dip
        _sample(1, 4, 8.5),
        _sample(1, 7, 12.0),
    ]
    cost = CostModel(manifest=_manifest(), samples=samples)
    assert cost.target_ms(1, 3) == 9.0 and cost.target_ms(1, 5) == 9.0
    assert cost.target_ms(1, 6) == pytest.approx(10.0)
