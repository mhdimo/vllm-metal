# SPDX-License-Identifier: Apache-2.0
"""Runner delivery of absolute feature spans on no-sample prefill steps."""

from types import SimpleNamespace

import pytest

import vllm_metal.v1.model_runner as mr
from tests.stub_runner import make_stub_runner
from tests.test_hidden_state_tap import _toy_backbone
from tests.test_v1_model_runner_generate import (
    TestIntermediateBodyOnlyForward as _PrefillFixture,
)


@pytest.mark.parametrize("prompt_logprobs", [False, True])
def test_intermediate_features_reach_proposer_without_sampling(
    monkeypatch, prompt_logprobs
):
    helper = _PrefillFixture()
    first = helper._intermediate_prefill_request("a", start_pos=3)
    second = helper._intermediate_prefill_request("b", start_pos=0)
    # Two independent spans: equal row counts must not obscure their positions.
    runner = make_stub_runner(
        model=SimpleNamespace(model=_toy_backbone(3), lm_head=lambda h: h),
    )
    runner.num_layers = 3
    runner._paged_block_size = 4
    runner._paged_group_block_sizes = (4,)
    seen = []
    runner._drafter = SimpleNamespace(
        capture_layer_ids=[0, 2],
        needs_target_hidden_states=lambda *a, **k: True,
        propose=lambda ctx: seen.append(ctx),
    )
    monkeypatch.setattr(mr, "prepare_grouped", lambda *a, **k: None)
    monkeypatch.setattr(
        runner._prompt_logprobs_tracker, "wants_any", lambda ids: prompt_logprobs
    )
    # Prompt-logprobs collection has separate coverage; verify it forces the
    # full head here without consuming a random sample on intermediate rows.
    monkeypatch.setattr(runner, "_gather_prefill_prompt_logprobs", lambda *a: None)
    monkeypatch.setattr(
        mr, "sample_prefill_tokens", lambda *a, **k: pytest.fail("sampled")
    )
    batch = mr._ExecutionBatch()
    for request in (first, second):
        idx = batch.add_output(request.req_id, [])
        batch.paged_prefill_entries.append(
            mr._PendingPrefillEntry(idx, request, "intermediate")
        )
    runner._start_paged_forward(
        batch, [first, second], [], helper._make_scheduler_output({"a": 2, "b": 2})
    )
    assert (runner._execute_model_state.logits is not None) is prompt_logprobs
    runner._sample_paged_batch()
    assert len(seen) == 1
    ctx = seen[0]
    assert ctx.target_hidden_states.shape == (4, 2)
    assert ctx.target_hidden_states[:, 0].tolist() == [5, 6, 5, 6]
    assert ctx.cu_seqlens == [0, 2, 4]
    assert [(pr.req_id, pr.start_pos) for pr in ctx.prefill_reqs] == [
        ("a", 3),
        ("b", 0),
    ]
    assert ctx.prefill_token_ids == []
    assert ctx.prefill_result_modes == ["intermediate", "intermediate"]
    assert batch.sampled_tokens == [[], []]
    assert runner._paged_request_seq_lens == {"a": 5, "b": 2}


def test_native_and_paged_capture_match_tiny_qwen3():
    import mlx.core as mx
    from mlx_lm.models.qwen3 import Model, ModelArgs

    from tools.dspark_target_check import check_target

    mx.random.seed(42)
    model = Model(
        ModelArgs(
            model_type="qwen3",
            vocab_size=64,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=128,
            rms_norm_eps=1e-6,
            max_position_embeddings=128,
            rope_theta=1_000_000.0,
            tie_word_embeddings=False,
        )
    )
    model.set_dtype(mx.float16)
    results = check_target(model, [0, 1])
    assert len(results) == 12
