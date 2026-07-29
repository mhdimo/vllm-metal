# SPDX-License-Identifier: Apache-2.0
"""Tests that the scheduler-managed draft KV cache honors the scheduler's
cache policy -- the gaps a proposer-private draft cache cannot see.

One property per gap named when the proposer-private draft cache was
rejected (see #500):

- ``cache_salt``: a resubmitted prompt under a different salt must not
  reuse draft KV.
- ``skip_reading_prefix_cache``: a request that opts out of cache reads
  must not reuse draft KV.
- Allocation limit: the proposer-local scratch reserve is sized to cover
  every concurrently active request drafting ``num_speculative_tokens``
  positions, so the draft pool cannot exhaust near the request limit.

The first two run end-to-end in a spawned child process (``spawn`` start
method -- Metal is not fork-safe) with Qwen3-0.6B draft==target, recording
each request's first draft plan (``draft_seq_len`` / ingest length) via a
monkeypatched proposer. The third is stub-level, no weights.
"""

from __future__ import annotations

import multiprocessing as mp
import os

import pytest
from vllm.utils.math_utils import cdiv

from tests.test_draft_model_proposer import (
    BLOCK_SIZE,
    _proposer,
    _StubDraftModel,
)
from tests.test_paged_deterministic import (
    DEFAULT_PAGED_MEMORY_FRACTION,
    DEFAULT_USE_PAGED_ATTENTION,
    MODEL_NAME,
)

K = 3
GEN = 8

SALT_PROMPT = (
    "Once upon a time in a far away kingdom there was a great king "
    "named Aragorn who ruled the land of Gondor with wisdom and "
    "grace, and his people loved him dearly because "
)
SKIP_PROMPT = (
    "The engineer examined the trace carefully, noting where the latency "
    "spiked and which subsystem held the lock the longest before yielding. "
)


def _setenv_default(key: str, default: str) -> None:
    if os.environ.get(key) is None:
        os.environ[key] = default


def _run_policy_e2e() -> None:
    """Body of the e2e test -- runs in a spawned child process."""
    _setenv_default("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    _setenv_default("VLLM_METAL_USE_PAGED_ATTENTION", DEFAULT_USE_PAGED_ATTENTION)
    _setenv_default("VLLM_METAL_MEMORY_FRACTION", DEFAULT_PAGED_MEMORY_FRACTION)

    if os.environ.get("VLLM_METAL_USE_PAGED_ATTENTION", "0") != "1":
        return  # non-paged path: nothing to test

    from vllm import LLM, SamplingParams

    from vllm_metal.v1 import draft_model_proposer as dmp

    plans: list[tuple[int, int]] = []

    def _note_plan(plan) -> None:
        if plan is not None:
            plans.append((plan.draft_seq_len, len(plan.ingest_tokens)))

    if hasattr(dmp.DraftModelProposer, "_make_plan"):
        _orig_make_plan = dmp.DraftModelProposer._make_plan

        def _logged_make_plan(self, req_id, state, num_speculative_tokens):
            plan = _orig_make_plan(self, req_id, state, num_speculative_tokens)
            _note_plan(plan)
            return plan

        dmp.DraftModelProposer._make_plan = _logged_make_plan
    else:
        _orig_decode = dmp.DraftModelProposer._make_decode_plan
        _orig_prefill = dmp.DraftModelProposer._make_prefill_plan

        def _logged_decode(self, req_id, state, k, drafting_req_ids):
            plan = _orig_decode(self, req_id, state, k, drafting_req_ids)
            _note_plan(plan)
            return plan

        def _logged_prefill(self, prefill, result_mode, k, drafting_req_ids):
            plan = _orig_prefill(self, prefill, result_mode, k, drafting_req_ids)
            _note_plan(plan)
            return plan

        dmp.DraftModelProposer._make_decode_plan = _logged_decode
        dmp.DraftModelProposer._make_prefill_plan = _logged_prefill

    llm = LLM(
        model=MODEL_NAME,
        max_model_len=1024,
        max_num_seqs=1,
        enable_prefix_caching=True,
        async_scheduling=False,
        speculative_config={
            "method": "draft_model",
            "model": MODEL_NAME,
            "num_speculative_tokens": K,
        },
    )
    sp = SamplingParams(temperature=0, max_tokens=GEN)

    def first_plan(prompt, sampling_params) -> tuple[int, int] | None:
        plans.clear()
        llm.generate([prompt], sampling_params)
        return plans[0] if plans else None

    # cache_salt: same prompt, different salt must not reuse draft KV.
    # max_tokens=2 keeps the warm generate short while still running
    # propose(), which writes the draft group's KV.
    warm = SamplingParams(temperature=0, max_tokens=2)
    first_plan({"prompt": SALT_PROMPT, "cache_salt": "a"}, warm)
    hit_a = first_plan({"prompt": SALT_PROMPT, "cache_salt": "a"}, sp)
    miss_b = first_plan({"prompt": SALT_PROMPT, "cache_salt": "b"}, sp)
    if hit_a is None or hit_a[0] == 0:
        raise AssertionError(f"same cache_salt should reuse draft KV, got {hit_a}")
    if miss_b is None or miss_b[0] != 0:
        raise AssertionError(
            f"different cache_salt must not reuse draft KV, got {miss_b}"
        )
    tokens = list(
        llm.generate([{"prompt": SALT_PROMPT, "cache_salt": "a"}], sp)[0]
        .outputs[0]
        .token_ids
    )
    if tokens != list(
        llm.generate([{"prompt": SALT_PROMPT, "cache_salt": "b"}], sp)[0]
        .outputs[0]
        .token_ids
    ):
        raise AssertionError("cache_salt changed the generated tokens")

    # skip_reading_prefix_cache: opting out of cache reads must not reuse
    # draft KV, and must not change the generated tokens.
    first_plan(SKIP_PROMPT, warm)
    hit = first_plan(SKIP_PROMPT, sp)
    skip = first_plan(
        SKIP_PROMPT,
        SamplingParams(temperature=0, max_tokens=GEN, skip_reading_prefix_cache=True),
    )
    if hit is None or hit[0] == 0:
        raise AssertionError(f"default read should reuse draft KV, got {hit}")
    if skip is None or skip[0] != 0:
        raise AssertionError(
            f"skip_reading_prefix_cache must not reuse draft KV, got {skip}"
        )
    tokens = list(llm.generate([SKIP_PROMPT], sp)[0].outputs[0].token_ids)
    skipped = list(
        llm.generate(
            [SKIP_PROMPT],
            SamplingParams(
                temperature=0, max_tokens=GEN, skip_reading_prefix_cache=True
            ),
        )[0]
        .outputs[0]
        .token_ids
    )
    if tokens != skipped:
        raise AssertionError("skip_reading_prefix_cache changed the generated tokens")


@pytest.mark.slow
def test_draft_cache_honors_scheduler_cache_policy_e2e() -> None:
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_run_policy_e2e)
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise AssertionError(
            "Draft cache-policy e2e test failed in spawned child "
            f"(exit code: {proc.exitcode})"
        )


def test_scratch_reserve_covers_max_concurrency() -> None:
    """The reserve formula fits every concurrent drafter at full concurrency.

    Each request's committed allocation covers its committed length (one
    block for 16 tokens here); the lookahead to ``committed_len + K - 1``
    then needs exactly ``cdiv(K, block_size)`` scratch block(s) per request.
    ``max_num_seqs`` requests must fit in ``max_num_seqs * cdiv(K, BLOCK_SIZE)``
    scratch blocks, and one block short must exhaust.
    """
    num_speculative_tokens = 4
    max_num_seqs = 3
    reserve = max_num_seqs * cdiv(num_speculative_tokens, BLOCK_SIZE)

    from vllm.sampling_params import SamplingParams

    from vllm_metal.v1.model_runner import RequestState
    from vllm_metal.v1.proposer import ProposeContext

    def _state() -> RequestState:
        return RequestState(
            token_ids=list(range(16)),
            prompt_len=16,
            cache=[],
            sampling_params=SamplingParams(temperature=0.0),
            block_ids=[[0]],
            num_computed_tokens=0,
        )

    def _context_for(states: dict[str, RequestState]) -> ProposeContext:
        return ProposeContext(
            target_hidden_states=None,
            decode_reqs=list(states.items()),
            decode_segments=[],
            decode_token_ids=[[state.token_ids[-1]] for state in states.values()],
            prefill_reqs=[],
            prefill_token_ids=[],
            prefill_result_modes=[],
            request_states=states,
            cu_seqlens=[],
            num_decode_segments=1,
            num_speculative_tokens=num_speculative_tokens,
            finished_req_ids=set(),
        )

    states = {f"r{i}": _state() for i in range(max_num_seqs)}

    model = _StubDraftModel()
    proposer = _proposer(model, committed_num_blocks=1, scratch_reserve_blocks=reserve)
    drafts = proposer.propose(_context_for(states))
    assert drafts is not None

    model = _StubDraftModel()
    starved = _proposer(
        model, committed_num_blocks=1, scratch_reserve_blocks=reserve - 1
    )
    with pytest.raises(RuntimeError, match="scratch pool exhausted"):
        starved.propose(_context_for(states))
