# SPDX-License-Identifier: Apache-2.0
"""Executable audit contracts. Expected failures are removed at M1/M2 fixes."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from tests.test_dspark_proposer import _context, _proposer, _segment, _state
from tests.test_hidden_state_tap import _toy_backbone
from vllm_metal.v1.dspark.model import CtxCache
from vllm_metal.v1.model_adapter import DefaultModelAdapter
from vllm_metal.v1.model_runner import MetalModelRunner


def test_capture_preserves_packed_logits_selection():
    adapter = DefaultModelAdapter()
    adapter._target_backbone = lambda model: _toy_backbone(1)
    adapter._compute_target_logits = lambda model, hidden: hidden
    layout = MetalModelRunner._paged_logits_layout(
        SimpleNamespace(_selective_logits_supported=True),
        [0, 3, 7, 12],
        num_decode_segments=1,
    )
    result = adapter.target_forward(
        object(),
        mx.arange(12, dtype=mx.float32)[None],
        cache=[None],
        capture_layer_ids=[0],
        logits_indices=layout.indices,
    )
    assert result.hidden_states.shape[0] == 12
    assert result.logits.shape == (1, 5, 1)
    assert result.logits[0, :, 0].tolist() == [0, 1, 2, 6, 11]


@pytest.mark.parametrize(
    "cached,committed", [(5, 4), (2, 8)], ids=["rollback", "missing-span"]
)
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="M2/F3: physical or logical context coverage is incomplete",
)
def test_context_plan_requires_complete_physical_coverage(cached, committed):
    proposer = _proposer()
    caches = [CtxCache() for _ in range(5)]
    for cache in caches:
        cache.append(mx.zeros((1, 1, cached, 1)), mx.zeros((1, 1, cached, 1)))
    proposer._ctx_caches["r"] = caches
    proposer._n_cached["r"] = cached
    state = _state(list(range(committed)))
    ctx = _context(
        decode_reqs=[("r", state)],
        decode_segments=[_segment("r", num_query_tokens=1)],
        target_hidden_states=mx.zeros((1, 4)),
    )
    plan = proposer._ensure_context(ctx, state, ctx.decode_segments[0], 2)
    # Failing closed is valid. Returning a proposal requires every physical
    # layer to cover precisely the committed prefix preceding the anchor.
    if plan is not None:
        assert plan.n_cached == committed - 1
        assert all(cache.length == committed - 1 for cache in plan.ctx_caches)
