# SPDX-License-Identifier: Apache-2.0
"""Tests for the draft-model proposer's committed/scratch block split.

A stub draft model returns logits of the right shape, so ingest, drafting,
and the release path run through ``propose`` without loading any weights.
The committed portion of the block table is scheduler-assigned (as it would
be by a real KVCacheManager for the draft's registered KV-cache group, see
``cache_policy.ModelCachePolicy._draft_layer_specs``); these tests simulate
that assignment directly on ``RequestState.block_ids`` rather than driving a
real scheduler.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
from vllm.sampling_params import SamplingParams

from vllm_metal.attention.context import OffsetCache, get_context
from vllm_metal.v1.draft_model_proposer import DraftModelProposer
from vllm_metal.v1.model_runner import PrefillRequest, RequestState
from vllm_metal.v1.proposer import ProposeContext
from vllm_metal.v1.spec_decode import SpeculativeDecodeController

BLOCK_SIZE = 16
COMMITTED_GROUP_INDEX = 0
COMMITTED_NUM_BLOCKS = 2
SCRATCH_RESERVE_BLOCKS = 2
VOCAB_SIZE = 8
PROMPT_LEN = 20


class _StubDraftModel:
    """mlx_lm-shaped draft model: logits per input token, recorded block tables."""

    def __init__(self) -> None:
        self.block_tables: list[list[list[int]]] = []
        self.input_lens: list[int] = []

    def __call__(self, input_ids: mx.array, *, cache: list[OffsetCache]) -> mx.array:
        ctx = get_context()
        assert ctx is not None
        self.block_tables.append([list(block_ids) for block_ids in ctx.block_tables])
        self.input_lens.append(int(input_ids.shape[1]))
        return mx.zeros((1, int(input_ids.shape[1]), VOCAB_SIZE), dtype=mx.float32)


def _proposer(
    model: _StubDraftModel,
    *,
    committed_num_blocks: int = COMMITTED_NUM_BLOCKS,
    scratch_reserve_blocks: int = SCRATCH_RESERVE_BLOCKS,
) -> DraftModelProposer:
    proposer = DraftModelProposer(
        model=model,
        block_size=BLOCK_SIZE,
        committed_num_blocks=committed_num_blocks,
        scratch_reserve_blocks=scratch_reserve_blocks,
        num_layers=1,
        controller=SpeculativeDecodeController(),
        extract_logits=lambda output: output,
    )
    proposer.adopt_committed_group(COMMITTED_GROUP_INDEX)
    return proposer


def _request_state(
    *,
    committed_block_ids: list[int],
    num_computed_tokens: int = 0,
    sampling_params: SamplingParams | None = None,
) -> RequestState:
    return RequestState(
        token_ids=list(range(PROMPT_LEN)),
        prompt_len=PROMPT_LEN,
        cache=[],
        sampling_params=sampling_params or SamplingParams(temperature=0.0),
        block_ids=[list(committed_block_ids)],
        num_computed_tokens=num_computed_tokens,
    )


def _context(
    req_id: str,
    state: RequestState,
    request_states: dict[str, RequestState],
    *,
    num_speculative_tokens: int = 1,
) -> ProposeContext:
    return ProposeContext(
        target_hidden_states=None,
        decode_reqs=[(req_id, state)],
        decode_segments=[],
        decode_token_ids=[[state.token_ids[-1]]],
        prefill_reqs=[],
        prefill_token_ids=[],
        prefill_result_modes=[],
        request_states=request_states,
        cu_seqlens=[],
        num_decode_segments=1,
        num_speculative_tokens=num_speculative_tokens,
        finished_req_ids=set(),
    )


def _drafting_blocks(model: _StubDraftModel, forward_index: int) -> set[int]:
    (block_ids,) = model.block_tables[forward_index]
    return set(block_ids)


def test_propose_before_adopt_committed_group_raises() -> None:
    model = _StubDraftModel()
    proposer = DraftModelProposer(
        model=model,
        block_size=BLOCK_SIZE,
        committed_num_blocks=COMMITTED_NUM_BLOCKS,
        scratch_reserve_blocks=SCRATCH_RESERVE_BLOCKS,
        num_layers=1,
        controller=SpeculativeDecodeController(),
        extract_logits=lambda output: output,
    )
    state = _request_state(committed_block_ids=[0, 1])
    with pytest.raises(RuntimeError, match="adopt_committed_group"):
        proposer.propose(_context("r1", state, {"r1": state}))


def test_committed_blocks_come_from_scheduler_assignment() -> None:
    """The committed portion of the block table is exactly what the
    scheduler assigned on RequestState.block_ids, not a proposer-owned pool."""
    model = _StubDraftModel()
    proposer = _proposer(model)
    state = _request_state(committed_block_ids=[1, 0])  # order matters
    drafts = proposer.propose(_context("r1", state, {"r1": state}))

    assert drafts is not None
    used = model.block_tables[0][0]
    assert used[:2] == [1, 0]


def test_lookahead_tail_draws_from_scratch_pool_beyond_committed() -> None:
    """A request whose committed blocks exactly cover its prompt still needs
    a scratch block for the speculative lookahead position(s)."""
    model = _StubDraftModel()
    proposer = _proposer(model, committed_num_blocks=1)
    state = _request_state(committed_block_ids=[0])
    drafts = proposer.propose(
        _context("r1", state, {"r1": state}, num_speculative_tokens=4)
    )

    assert drafts is not None
    used = model.block_tables[0][0]
    assert used[0] == 0  # committed block, scheduler-assigned
    assert used[1] >= 1  # scratch block, drawn from the offset range


def test_cache_hit_seeds_draft_seq_len_from_scheduler_boundary() -> None:
    """A resubmitted/shared prefix reported via num_computed_tokens must not
    be re-ingested -- this is #482's core fix."""
    model = _StubDraftModel()
    proposer = _proposer(model)
    state = _request_state(
        committed_block_ids=[0, 1], num_computed_tokens=PROMPT_LEN - 1
    )
    drafts = proposer.propose(_context("r1", state, {"r1": state}))

    assert drafts is not None
    assert model.input_lens[0] == 1


def test_no_cache_hit_ingests_the_whole_committed_range() -> None:
    model = _StubDraftModel()
    proposer = _proposer(model)
    state = _request_state(committed_block_ids=[0, 1], num_computed_tokens=0)
    drafts = proposer.propose(_context("r1", state, {"r1": state}))

    assert drafts is not None
    assert model.input_lens[0] == PROMPT_LEN


def test_non_greedy_request_ingests_but_never_drafts() -> None:
    """Non-greedy requests must still keep the committed group's KV in sync
    (the scheduler advances num_computed_tokens for them regardless), but
    must never receive draft tokens."""
    model = _StubDraftModel()
    proposer = _proposer(model)
    state = _request_state(
        committed_block_ids=[0, 1], sampling_params=SamplingParams(temperature=1.0)
    )
    drafts = proposer.propose(_context("r1", state, {"r1": state}))

    assert drafts is None  # nothing to draft for
    assert len(model.block_tables) == 1  # but ingest still ran


def test_intermediate_prefill_chunk_ingests_without_drafting() -> None:
    """An intermediate (not-yet-final) prefill chunk must ingest its
    scheduled slice so the committed group's KV stays in sync, but must not
    produce draft tokens (mirrors non-greedy: keep pace, never draft)."""
    model = _StubDraftModel()
    proposer = _proposer(model)
    prefill = PrefillRequest(
        req_id="r1",
        token_ids=list(range(16)),
        sampling_params=SamplingParams(temperature=0.0),
        block_ids=[[0]],
        generator=None,
        prompt_len=None,
        start_pos=0,
        full_prompt_token_ids=None,
    )
    state = _request_state(committed_block_ids=[0])
    ctx = ProposeContext(
        target_hidden_states=None,
        decode_reqs=[],
        decode_segments=[],
        decode_token_ids=[],
        prefill_reqs=[prefill],
        prefill_token_ids=[],
        prefill_result_modes=["intermediate"],
        request_states={"r1": state},
        cu_seqlens=[],
        num_decode_segments=0,
        num_speculative_tokens=1,
        finished_req_ids=set(),
    )
    drafts = proposer.propose(ctx)

    assert drafts is None
    assert len(model.block_tables) == 1


def test_release_requests_returns_scratch_blocks_to_the_free_pool() -> None:
    model = _StubDraftModel()
    proposer = _proposer(model, committed_num_blocks=1)
    waiting = _request_state(committed_block_ids=[0])
    resumed = _request_state(committed_block_ids=[0])
    request_states = {"waiting": waiting, "resumed": resumed}

    assert (
        proposer.propose(
            _context("waiting", waiting, request_states, num_speculative_tokens=4)
        )
        is not None
    )
    pool = _drafting_blocks(model, 0)

    proposer.release_requests({"waiting"})

    drafts = proposer.propose(
        _context("resumed", resumed, request_states, num_speculative_tokens=4)
    )

    assert drafts is not None
    assert list(drafts.req_ids) == ["resumed"]
    assert _drafting_blocks(model, 1) == pool


def test_scratch_pool_exhaustion_raises() -> None:
    model = _StubDraftModel()
    proposer = _proposer(model, committed_num_blocks=1, scratch_reserve_blocks=1)
    waiting = _request_state(committed_block_ids=[0])
    resumed = _request_state(committed_block_ids=[0])
    request_states = {"waiting": waiting, "resumed": resumed}

    assert (
        proposer.propose(
            _context("waiting", waiting, request_states, num_speculative_tokens=4)
        )
        is not None
    )

    calls_before = len(model.block_tables)
    with pytest.raises(RuntimeError, match="scratch pool exhausted"):
        proposer.propose(
            _context("resumed", resumed, request_states, num_speculative_tokens=4)
        )
    assert len(model.block_tables) == calls_before


# -- Sliding-window / hybrid draft rejection ---------------------------------


class _FakeModelConfig:
    """Minimal mock of ``vllm.config.ModelConfig`` for config-only checks."""

    def __init__(
        self,
        *,
        sliding_window: int | None = None,
        layer_types: list[str] | None = None,
    ) -> None:
        self._sliding_window = sliding_window
        # hf_text_config is where layer_types and sliding_window live
        self.hf_text_config = type("_HFConfig", (), {})()
        if layer_types is not None:
            self.hf_text_config.layer_types = layer_types

    def get_sliding_window(self) -> int | None:
        return self._sliding_window

    def get_total_num_hidden_layers(self) -> int:
        return 4

    def get_num_kv_heads(self, parallel_config: object) -> int:
        return 2

    def get_head_size(self) -> int:
        return 64


def test_sliding_window_draft_rejected() -> None:
    """A draft model with sliding_window must be rejected at config time."""
    from vllm_metal.v1.draft_model_proposer import _require_full_attention_draft

    cfg = _FakeModelConfig(sliding_window=4096)
    with pytest.raises(NotImplementedError, match="sliding_window=4096"):
        _require_full_attention_draft(cfg)


def test_hybrid_layer_types_draft_rejected() -> None:
    """A draft model with mixed layer types (e.g. Gemma4) must be rejected."""
    from vllm_metal.v1.draft_model_proposer import _require_full_attention_draft

    cfg = _FakeModelConfig(
        layer_types=["sliding_attention", "full_attention"] * 2,
    )
    with pytest.raises(NotImplementedError, match="non-full-attention layer types"):
        _require_full_attention_draft(cfg)


def test_full_attention_draft_accepted() -> None:
    """A plain full-attention draft model must pass the check."""
    from vllm_metal.v1.draft_model_proposer import _require_full_attention_draft

    # No sliding window, no layer_types at all (standard transformer)
    cfg_plain = _FakeModelConfig()
    _require_full_attention_draft(cfg_plain)  # should not raise

    # Explicit full_attention layer_types only
    cfg_explicit = _FakeModelConfig(
        layer_types=["full_attention"] * 4,
    )
    _require_full_attention_draft(cfg_explicit)  # should not raise
