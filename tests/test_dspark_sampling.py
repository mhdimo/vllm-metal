# SPDX-License-Identifier: Apache-2.0
"""Exact stochastic proposal and verification helpers of DSpark."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest
from vllm.sampling_params import SamplingParams

from vllm_metal.v1.dspark.sampling import (
    RESIDUAL_MASS_EPS,
    RequestRandomStreams,
    SamplingTransforms,
    acceptance_probabilities,
    batched_transformed_distribution,
    first_rejection,
    residual_distribution,
    sample_from_distribution,
    transformed_distribution,
    verify_rows,
)


def _rows(values) -> mx.array:
    return mx.array(values, dtype=mx.float32)


def _np(array: mx.array) -> np.ndarray:
    return np.array(array.tolist(), dtype=np.float64)


def _random_distributions(rng, count, vocab, floor=0.05):
    rows = rng.dirichlet(np.ones(vocab), size=count) + floor
    return rows / rows.sum(axis=1, keepdims=True)


def test_transforms_come_from_the_request_and_reject_greedy() -> None:
    params = SamplingParams(temperature=0.7, top_k=5, top_p=0.9)
    assert SamplingTransforms.from_params(params) == SamplingTransforms(0.7, 5, 0.9)
    assert SamplingTransforms.from_params(SamplingParams(temperature=0.0)).greedy
    with pytest.raises(ValueError, match="greedy"):
        transformed_distribution(
            _rows([[1.0, 2.0]]), SamplingTransforms(0.0, 0, 1.0), vocab_size=2
        )


def test_transformed_distribution_uses_sampler_mask_semantics() -> None:
    logits = _rows([[2.0, 1.0, 1.0, 0.0, -1.0, 9.0]])
    # Padded lm_head columns (index 5) never carry mass.
    full = _np(
        transformed_distribution(logits, SamplingTransforms(1.0, 0, 1.0), vocab_size=5)
    )
    assert full.shape == (1, 5) and math.isclose(full.sum(), 1.0, abs_tol=1e-6)
    expected = np.exp(np.array([2.0, 1.0, 1.0, 0.0, -1.0]))
    assert np.allclose(full[0], expected / expected.sum(), atol=1e-6)
    # top-k keeps every token tied at the threshold (vLLM semantics).
    top_k = _np(
        transformed_distribution(logits, SamplingTransforms(1.0, 2, 1.0), vocab_size=5)
    )[0]
    assert top_k[3] == 0.0 and top_k[4] == 0.0
    assert top_k[1] == top_k[2] > 0.0 and math.isclose(top_k.sum(), 1.0, abs_tol=1e-6)
    # top-p masks sorted positions individually and always keeps the leader.
    top_p = _np(
        transformed_distribution(logits, SamplingTransforms(1.0, 0, 0.5), vocab_size=5)
    )[0]
    assert top_p[0] > 0.0 and top_p[3] == 0.0 and top_p[4] == 0.0
    # Temperature scales before masking; a very cold distribution is a near point mass.
    cold = _np(
        transformed_distribution(logits, SamplingTransforms(0.05, 0, 1.0), vocab_size=5)
    )[0]
    assert cold[0] > 0.999


def test_batched_distribution_matches_each_row_alone() -> None:
    rng = np.random.default_rng(3)
    logits = _rows(rng.normal(size=(4, 8)))
    transforms = [
        SamplingTransforms(1.0, 0, 1.0),
        SamplingTransforms(0.6, 3, 1.0),
        SamplingTransforms(1.3, 0, 0.7),
        SamplingTransforms(0.6, 3, 1.0),
    ]
    batched = _np(batched_transformed_distribution(logits, transforms, vocab_size=8))
    for index, item in enumerate(transforms):
        alone = _np(
            transformed_distribution(logits[index : index + 1], item, vocab_size=8)
        )
        assert np.array_equal(batched[index], alone[0])
    same = [SamplingTransforms(0.9, 0, 1.0), SamplingTransforms(1.7, 0, 1.0)]
    batched = _np(batched_transformed_distribution(logits[:2], same, vocab_size=8))
    for index, item in enumerate(same):
        alone = _np(
            transformed_distribution(logits[index : index + 1], item, vocab_size=8)
        )
        assert np.allclose(batched[index], alone[0], atol=1e-7)


def test_inverse_cdf_sampling_edges() -> None:
    probs = _rows([[0.0, 0.25, 0.0, 0.75]])

    def pick(uniform: float) -> int:
        token = sample_from_distribution(probs, mx.array([uniform], dtype=mx.float32))
        return int(token.item())

    assert pick(0.0) == 1  # leading zero-mass tokens are skipped
    assert pick(0.1) == 1
    assert pick(0.2499) == 1
    assert pick(0.25) == 3  # boundary belongs to the next token
    assert pick(0.9999) == 3
    assert pick(1.0) == 3  # a uniform rounded up to 1.0 selects the last positive token
    trailing = _rows([[0.5, 0.5, 0.0]])
    assert (
        int(
            sample_from_distribution(trailing, mx.array([1.0], dtype=mx.float32)).item()
        )
        == 1
    )
    # Unnormalized rows scale the uniform by their total mass.
    scaled = _rows([[0.0, 0.5, 0.0, 1.5]])
    assert (
        int(sample_from_distribution(scaled, mx.array([0.2], dtype=mx.float32)).item())
        == 1
    )
    assert (
        int(sample_from_distribution(scaled, mx.array([0.3], dtype=mx.float32)).item())
        == 3
    )
    # Zero-mass tokens are never selected across a dense grid.
    for uniform in np.linspace(0.0, 1.0, 1001):
        assert pick(float(uniform)) in (1, 3)


def test_residual_distribution_clips_and_falls_back_to_target() -> None:
    target = _rows([[0.5, 0.3, 0.2]])
    draft = _rows([[0.2, 0.5, 0.3]])
    assert np.allclose(_np(residual_distribution(target, draft))[0], [1.0, 0.0, 0.0])
    same = _np(residual_distribution(target, target))[0]
    assert np.allclose(same, [0.5, 0.3, 0.2])
    nearly = target + _rows([[RESIDUAL_MASS_EPS / 10, 0.0, -RESIDUAL_MASS_EPS / 10]])
    assert np.allclose(_np(residual_distribution(nearly, target))[0], _np(nearly)[0])


def test_acceptance_probabilities_report_zero_proposal_mass() -> None:
    target = _rows([[0.6, 0.4, 0.0], [0.1, 0.1, 0.8]])
    draft = _rows([[0.3, 0.7, 0.0], [0.0, 0.5, 0.5]])
    probabilities, q_at = acceptance_probabilities(target, draft, [0, 0])
    assert np.allclose(_np(probabilities), [1.0, 0.0])
    assert np.allclose(_np(q_at), [0.3, 0.0])
    probabilities, _ = acceptance_probabilities(target, draft, [1, 2])
    assert np.allclose(_np(probabilities), [0.4 / 0.7, 1.0])
    with pytest.raises(ValueError, match="one draft token"):
        acceptance_probabilities(target, draft, [1])


def test_first_rejection_uses_strict_uniform_comparison() -> None:
    assert first_rejection([1.0, 1.0], [0.999, 0.0]) == 2
    assert first_rejection([0.5, 1.0], [0.5, 0.0]) == 0
    assert first_rejection([0.5, 0.2], [0.1, 0.2]) == 1
    assert first_rejection([], []) == 0
    with pytest.raises(ValueError):
        first_rejection([0.5], [])


def _interval_uniform(probs: np.ndarray, token: int) -> float:
    """A uniform strictly inside the inverse-CDF interval of ``token``."""
    cumulative = np.concatenate([[0.0], np.cumsum(probs)])
    return float((cumulative[token] + cumulative[token + 1]) / 2 / cumulative[-1])


def test_enumerated_tiny_vocabulary_oracle_recovers_the_target() -> None:
    """Enumerate every branch of two-position verification with exact weights.

    Each uniform is chosen inside the interval that selects a branch, and the
    branch is weighted by the interval's exact length; summing over branches
    gives the exact distribution of what the code emits. Position 0 must be
    distributed as ``p_0``, and position 1, given an accepted position 0,
    as ``p_1``; a fully accepted block draws its bonus from ``p_2``.
    """
    rng = np.random.default_rng(11)
    vocab, width = 4, 2
    target_np = _random_distributions(rng, width + 1, vocab)
    draft_np = _random_distributions(rng, width, vocab)
    target, draft = _rows(target_np), _rows(draft_np)
    first: dict[int, float] = dict.fromkeys(range(vocab), 0.0)
    second_given_accept: dict[int, float] = dict.fromkeys(range(vocab), 0.0)
    accept_mass = 0.0
    residual = [np.maximum(target_np[k] - draft_np[k], 0.0) for k in range(width)]
    residual = [row / row.sum() for row in residual]

    def emit(tokens, uniforms, target_uniform, weight):
        output = verify_rows(target, draft, tokens, uniforms, target_uniform)
        return output, weight

    for x0 in range(vocab):
        w0 = draft_np[0][x0]
        a0 = min(1.0, target_np[0][x0] / draft_np[0][x0])
        # Rejected at position 0: recovered token from the residual.
        if a0 < 1.0:
            for t in range(vocab):
                if residual[0][t] == 0.0:
                    continue
                output, weight = emit(
                    [x0, 0],
                    [(1.0 + a0) / 2, 0.0],
                    _interval_uniform(residual[0], t),
                    w0 * (1 - a0) * residual[0][t],
                )
                assert output == [t]
                first[t] += weight
        if a0 == 0.0:
            continue
        for x1 in range(vocab):
            w1 = draft_np[1][x1]
            a1 = min(1.0, target_np[1][x1] / draft_np[1][x1])
            if a1 < 1.0:
                for t in range(vocab):
                    if residual[1][t] == 0.0:
                        continue
                    output, weight = emit(
                        [x0, x1],
                        [a0 / 2, (1.0 + a1) / 2],
                        _interval_uniform(residual[1], t),
                        w0 * a0 * w1 * (1 - a1) * residual[1][t],
                    )
                    assert output == [x0, t]
                    first[x0] += weight
                    second_given_accept[t] += weight
                    accept_mass += weight
            if a1 == 0.0:
                continue
            for t in range(vocab):
                output, weight = emit(
                    [x0, x1],
                    [a0 / 2, a1 / 2],
                    _interval_uniform(target_np[2], t),
                    w0 * a0 * w1 * a1 * target_np[2][t],
                )
                assert output == [x0, x1, t]
                first[x0] += weight
                second_given_accept[x1] += weight
                accept_mass += weight
    assert math.isclose(sum(first.values()), 1.0, abs_tol=1e-9)
    for token in range(vocab):
        assert math.isclose(first[token], target_np[0][token], abs_tol=1e-9)
        assert math.isclose(
            second_given_accept[token] / accept_mass, target_np[1][token], abs_tol=1e-9
        )


def _chi_square(counts: np.ndarray, expected: np.ndarray) -> float:
    return float(((counts - expected) ** 2 / expected).sum())


def test_distribution_gate_with_request_streams_and_negative_control() -> None:
    """Emitted tokens follow the target; a wrong proposal record is detected.

    Sample size 20,000 over six tokens: the chi-square statistic against the
    target stays under the 0.001 critical value (20.5 at five degrees of
    freedom) for the exact verifier, and exceeds it by a wide margin when the
    verifier is handed a proposal distribution that is not the one the drafts
    were sampled from, which is what the record protects against.
    """
    rng = np.random.default_rng(5)
    vocab, trials = 6, 20_000
    target_np = _random_distributions(rng, 2, vocab)
    draft_np = _random_distributions(rng, 1, vocab)
    wrong_np = draft_np[:, ::-1].copy()
    target, draft, wrong = _rows(target_np), _rows(draft_np), _rows(wrong_np)
    streams = RequestRandomStreams.for_request(None, engine_seed=0, ordinal=1)
    proposal = streams.proposal.random(trials)
    drafted = sample_from_distribution(
        mx.broadcast_to(draft, (trials, vocab)), mx.array(proposal, dtype=mx.float32)
    )
    mx.eval(drafted)
    tokens = drafted.tolist()
    exact = np.zeros(vocab)
    biased = np.zeros(vocab)
    for token in tokens:
        uniform = streams.acceptance.random(1).tolist()
        target_uniform = float(streams.target.random())
        exact[verify_rows(target, draft, [token], uniform, target_uniform)[0]] += 1
        biased[verify_rows(target, wrong, [token], uniform, target_uniform)[0]] += 1
    expected = target_np[0] * trials
    assert _chi_square(exact, expected) < 20.5
    assert _chi_square(biased, expected) > 100.0


def test_zero_support_and_truncated_proposals() -> None:
    # The target excludes the drafted token: rejected with probability one and
    # the residual is the target itself where the proposal has no mass.
    target = _rows([[0.0, 0.5, 0.5], [0.2, 0.3, 0.5]])
    draft = _rows([[1.0, 0.0, 0.0]])
    assert verify_rows(target, draft, [0], [0.0], 0.1) == [1]
    assert verify_rows(target, draft, [0], [0.0], 0.9) == [2]
    # A drafted token the proposal never assigns mass to is an invalid record.
    with pytest.raises(ValueError, match="zero proposal mass"):
        verify_rows(target, draft, [1], [0.0], 0.1)
    # Full acceptance draws the bonus from the last target row.
    aligned = _rows([[1.0, 0.0, 0.0], [0.2, 0.3, 0.5]])
    assert verify_rows(aligned, draft, [0], [0.999], 0.15) == [0, 0]
    assert verify_rows(aligned, draft, [0], [0.999], 0.6) == [0, 2]


def test_request_streams_are_seeded_per_request_and_isolated() -> None:
    same_a = RequestRandomStreams.for_request(7, engine_seed=0, ordinal=1)
    same_b = RequestRandomStreams.for_request(7, engine_seed=99, ordinal=5)
    assert same_a.proposal.random(3).tolist() == same_b.proposal.random(3).tolist()
    first = RequestRandomStreams.for_request(None, engine_seed=0, ordinal=1)
    second = RequestRandomStreams.for_request(None, engine_seed=0, ordinal=2)
    replay = RequestRandomStreams.for_request(None, engine_seed=0, ordinal=1)
    assert first.acceptance.random(4).tolist() != second.acceptance.random(4).tolist()
    # Consuming another request's streams (or this request's other streams)
    # leaves this stream's next draws unchanged.
    second.target.random(100)
    first.proposal.random(100)
    replay.acceptance.random(4)
    assert first.acceptance.random(2).tolist() == replay.acceptance.random(2).tolist()
    entropy = RequestRandomStreams.for_request(None, engine_seed=None, ordinal=1)
    assert entropy.proposal.random(2).tolist() != first.proposal.random(2).tolist()
