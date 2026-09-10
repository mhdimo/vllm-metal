# SPDX-License-Identifier: Apache-2.0
"""Pure helpers of the HTTP serving qualification harness."""

from __future__ import annotations

from tools.dspark_serving_check import (
    TIE_LOGPROB_GAP,
    first_divergence,
    parse_metrics,
    token_logprobs,
)


def test_parse_metrics_sums_labelled_samples_and_skips_comments() -> None:
    text = "\n".join(
        [
            "# HELP vllm:spec_decode_num_draft_tokens_total drafts",
            "# TYPE vllm:spec_decode_num_draft_tokens_total counter",
            'vllm:spec_decode_num_draft_tokens_total{model_name="target"} 2219.0',
            'vllm:spec_decode_num_accepted_tokens_total{model_name="target"} 1396.0',
            'vllm:num_requests_running{model_name="target"} 0.0',
            'vllm:spec_decode_num_accepted_tokens_per_pos{position="0"} 900.0',
            'vllm:spec_decode_num_accepted_tokens_per_pos{position="1"} 496.0',
            "",
            "garbage line without value",
        ]
    )
    values = parse_metrics(text)
    assert values["vllm:spec_decode_num_draft_tokens_total"] == 2219.0
    assert values["vllm:spec_decode_num_accepted_tokens_total"] == 1396.0
    assert values["vllm:num_requests_running"] == 0.0
    assert values["vllm:spec_decode_num_accepted_tokens_per_pos"] == 1396.0
    assert "garbage" not in " ".join(values)


def test_token_logprobs_reads_only_token_id_keys() -> None:
    entry = {"token_id:9664": -0.01, "token_id:198": -4.5, " the": -9.0}
    assert token_logprobs(entry) == {9664: -0.01, 198: -4.5}


def test_first_divergence_and_tie_gap_semantics() -> None:
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert first_divergence([1, 2], [1, 2, 3]) == 2
    # Two bfloat16 ULPs at logit magnitude 32 to 64 is 0.5; the HTTP gate uses
    # that bound because logprob differences equal logit differences.
    assert TIE_LOGPROB_GAP == 0.5
