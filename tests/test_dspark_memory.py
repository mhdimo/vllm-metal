# SPDX-License-Identifier: Apache-2.0
"""DSpark resource admission, real buffer storage and recovery contracts."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import pytest

from tests.test_dspark_loader import checkpoint
from tests.test_dspark_proposer import _features, _proposer, _seed, _state
from tests.test_v1_worker import TestPagedAttentionPlanDiagnostics as PlanDiagnostics
from vllm_metal.v1.dspark.loader import load_drafter
from vllm_metal.v1.dspark.memory import DSparkMemoryPlan
from vllm_metal.v1.dspark.model import ContextArena, CtxCache


def test_append_reuses_chunks_and_rollback_hides_old_suffix(monkeypatch):
    cache = CtxCache(capacity=300)
    values = mx.arange(300 * 8, dtype=mx.float32).reshape(1, 2, 300, 4)
    cache.append(values[:, :, :250], -values[:, :, :250])
    mx.eval(cache.k, cache.v)
    allocated = cache.allocated_bytes
    concatenate = mx.concatenate

    def no_growth(*args, **kwargs):
        pytest.fail("an append within capacity must not concatenate the context")

    monkeypatch.setattr(mx, "concatenate", no_growth)
    cache.append(values[:, :, 250:256], -values[:, :, 250:256])
    mx.eval(cache.k, cache.v)
    assert cache.allocated_bytes == allocated
    assert mx.array_equal(cache.k, values[:, :, :256]).item()
    cache.trim_to(10)
    cache.append(values[:, :, 20:23], -values[:, :, 20:23])
    mx.eval(cache.k, cache.v)
    assert cache.k.shape[2] == cache.v.shape[2] == 13
    assert mx.array_equal(cache.k[:, :, :10], values[:, :, :10]).item()
    assert mx.array_equal(cache.k[:, :, 10:], values[:, :, 20:23]).item()
    monkeypatch.setattr(mx, "concatenate", concatenate)
    cache.append(values[:, :, 13:], -values[:, :, 13:])
    mx.eval(cache.k, cache.v)
    assert cache.length == 300
    assert cache.allocated_bytes == 2 * values.nbytes
    with pytest.raises(ValueError, match="reserved capacity"):
        cache.append(values[:, :, :1], values[:, :, :1])
    assert cache.length == 300


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_memory_plan_matches_actual_full_context_storage(dtype):
    proposer = _proposer()
    proposer._drafter.set_dtype(dtype)
    plan = DSparkMemoryPlan.build(
        proposer._config,
        itemsize=mx.array(0, dtype=dtype).itemsize,
        max_num_seqs=2,
        max_model_len=257,
        max_num_batched_tokens=64,
    )
    caches = [
        proposer._drafter.make_ctx_cache(plan.max_context_tokens) for _ in range(2)
    ]
    # Input dtype intentionally differs from the BF16/FP16 draft precision.
    features = _features(list(range(plan.max_context_tokens)))[None]
    for row in caches:
        proposer._drafter.update_context(features, 0, row)
        mx.eval([(cache.k, cache.v) for cache in row])
        assert all(cache.k.dtype == cache.v.dtype == dtype for cache in row)
    # The arena the proposer allocates for this plan is exactly the reserved
    # context bytes: every slot at full capacity plus the block's scratch.
    config = proposer._config
    arenas = [
        ContextArena(
            slots=plan.max_contexts,
            kv_heads=config.n_kv_heads,
            capacity=plan.max_context_tokens,
            block_size=config.block_size,
            head_dim=config.attn_head_dim,
            dtype=dtype,
        )
        for _ in proposer._drafter.layers
    ]
    assert sum(arena.nbytes for arena in arenas) == plan.context_bytes
    assert plan.block_size == config.block_size


def test_context_slot_exhaustion_releases_and_recovers():
    proposer = _proposer(max_num_seqs=1)
    first, second = _state([1, 2, 3]), _state([4, 5, 6])
    assert _seed(proposer, first, "a") is not None
    assert _seed(proposer, second, "b") is None
    assert proposer._contexts["b"].disabled_reason == "context capacity exhausted"
    assert not proposer._contexts["b"].caches
    proposer.release_requests({"a"})
    # Only a recompute from zero can recover the skipped feature prefix.
    assert _seed(proposer, second, "b") is not None
    assert proposer._contexts["b"].covered_end == 2


def test_memory_pressure_rejects_before_allocation_then_recovers(monkeypatch):
    proposer = _proposer()
    current = 1234
    monkeypatch.setattr(mx, "get_active_memory", lambda: current)
    # Context slots are reserved up front in the arena, so the live check
    # that admission applies is the workspace the draft step will need.
    budget = current + proposer.memory_plan.workspace_bytes
    proposer._memory_budget_bytes = budget - 1
    owner = _state([1, 2, 3])
    assert _seed(proposer, owner) is None
    assert proposer._contexts["r"].disabled_reason == "context memory budget exhausted"
    assert not proposer._contexts["r"].caches
    assert proposer._arena[0].free_slots == proposer.memory_plan.max_contexts
    proposer._memory_budget_bytes = budget
    assert _seed(proposer, owner) is not None
    slot_bytes = proposer.memory_plan.context_bytes // proposer.memory_plan.max_contexts
    assert (
        sum(cache.allocated_bytes for cache in proposer._contexts["r"].caches)
        == slot_bytes
    )
    assert proposer._arena[0].free_slots == proposer.memory_plan.max_contexts - 1


@pytest.mark.parametrize(
    "error",
    [
        MemoryError("injected"),
        RuntimeError("[metal::malloc] Resource limit (injected)"),
    ],
)
@pytest.mark.parametrize("phase", ["ingest", "draft"])
def test_allocation_failure_drops_partial_context_and_next_request_recovers(
    monkeypatch, phase, error
):
    proposer = _proposer()
    owner = _state([1, 2, 3])
    assert _seed(proposer, owner, "old") is not None
    name = "update_context" if phase == "ingest" else "backbone"
    original = getattr(proposer._drafter, name)

    def fail(*args, **kwargs):
        if phase == "ingest":
            # Fail after real writes, not before any state exists.
            original(*args, **kwargs)
        raise error

    monkeypatch.setattr(proposer._drafter, name, fail)
    assert _seed(proposer, _state([4, 5, 6]), "failed") is None
    assert not proposer._contexts
    monkeypatch.setattr(proposer._drafter, name, original)
    assert _seed(proposer, _state([7, 8, 9]), "new") is not None
    assert proposer._contexts["new"].covered_end == 2


def test_scheduler_span_cannot_exceed_reserved_workspace():
    proposer = _proposer()
    proposer.memory_plan = replace(proposer.memory_plan, max_step_tokens=2)
    with pytest.raises(ValueError, match="reserved scheduler token bound"):
        _seed(proposer, _state([1, 2, 3, 4]))
    assert not proposer._contexts


def test_target_kv_plan_subtracts_complete_dspark_reservation(monkeypatch):
    proposer = _proposer()
    runner = SimpleNamespace(
        is_hybrid=False,
        draft_scratch_reserve_bytes=lambda: 0,
        _dspark_memory_plan=proposer.memory_plan,
    )
    planner = PlanDiagnostics()._make_planner(
        runner, memory_fraction=0.5, per_block_bytes=4096
    )
    monkeypatch.setattr(planner, "_metal_limit_bytes", lambda: 8_000_000_000)
    monkeypatch.setattr(planner, "get_model_memory_usage", lambda: 1_000_000_000)
    plain = planner._paged_attention_plan(overhead=100_000_000)
    planner._worker.vllm_config.speculative_config = SimpleNamespace(method="dspark")
    spec = planner._paged_attention_plan(overhead=100_000_000)
    assert plain.kv_budget - spec.kv_budget == proposer.memory_plan.reserve_bytes
    assert (
        spec.num_blocks * spec.per_block_bytes
        + spec.model_memory
        + spec.overhead
        + proposer.memory_plan.reserve_bytes
        <= spec.usable_metal
    )
    runner._dspark_memory_plan = None
    with pytest.raises(RuntimeError, match="before target KV allocation"):
        planner._paged_attention_plan(overhead=100_000_000)


def test_loader_budget_and_resolved_identity_fail_before_reading_weights(
    tmp_path, monkeypatch
):
    model, _ = checkpoint(tmp_path)

    def no_read(*args, **kwargs):
        pytest.fail("startup admission must precede weight data reads")

    monkeypatch.setattr(mx, "load", no_read)
    with pytest.raises(ValueError, match="startup memory budget"):
        load_drafter(str(tmp_path), memory_budget_bytes=1)
    with pytest.raises(ValueError, match="resolved draft ModelConfig"):
        load_drafter(str(tmp_path), expected_config=replace(model.config, block_size=5))


def test_memory_plan_honors_configured_context_cap():
    proposer = _proposer()
    common = {"itemsize": 2, "max_model_len": 256, "max_num_batched_tokens": 32}
    build = DSparkMemoryPlan.build
    assert (
        build(proposer._config, max_num_seqs=8, max_contexts=2, **common).max_contexts
        == 2
    )
    assert (
        build(proposer._config, max_num_seqs=8, max_contexts=64, **common).max_contexts
        == 8
    )
    assert build(proposer._config, max_num_seqs=64, **common).max_contexts == 32
    # Each slot holds the full context plus the drafted block's scratch K/V.
    block = proposer._config.block_size
    assert (
        build(proposer._config, max_num_seqs=8, max_contexts=2, **common).context_bytes
        == 2
        * (256 + block)
        * build(proposer._config, max_num_seqs=8, **common).kv_bytes_per_token
    )
    with pytest.raises(ValueError, match="at least one"):
        build(proposer._config, max_num_seqs=8, max_contexts=0, **common)
