# SPDX-License-Identifier: Apache-2.0
"""Row identity and divergence labelling of the DSpark diagnostic tools."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest
from vllm.sampling_params import SamplingParams

from tools.dspark_divergence_classify import (
    MAX_TIE_ULPS,
    assign_requests,
    classify_pair,
    emitted_tokens,
    first_divergence,
    ulp,
)
from tools.dspark_memory_check import top_logit_rows
from vllm_metal.v1.model_runner import PrefillRequest, RequestState
from vllm_metal.v1.spec_decode import PagedDecodeSegment

VOCAB = 16


def _state(tokens: list[int], prompt_len: int) -> RequestState:
    return RequestState(
        list(tokens), prompt_len, [], SamplingParams(temperature=0.0, max_tokens=8)
    )


def _logits(rows: int) -> mx.array:
    # Row r has argmax token r+1 (value 10), runner-up r+2 (value 9.875), the
    # rest descending, so identity errors show as wrong tokens, not near ties.
    base = mx.arange(VOCAB, dtype=mx.float32)[None, :] * 0.01
    out = mx.broadcast_to(base, (rows, VOCAB)) + 0.0
    for r in range(rows):
        out[r, (r + 1) % VOCAB] = 10.0
        out[r, (r + 2) % VOCAB] = 9.875
    return out[None]


def _paged_state(logits, decode, segments, prefills, boundaries, intermediate=False):
    return SimpleNamespace(
        logits=logits,
        decode_reqs=decode,
        decode_segments=segments,
        prefill_reqs=prefills,
        logits_cu_seqlens=boundaries,
        intermediate_only=intermediate,
    )


def test_rows_carry_request_identity_and_absolute_positions() -> None:
    verify_state = _state([1, 2, 3, 4, 5], prompt_len=4)  # anchor at index 4
    decode_state = _state([7, 8, 9], prompt_len=2)
    segments = [
        PagedDecodeSegment("v", (5, 11, 12), 0, 3, (11, 12), 4, ((0,),)),
        PagedDecodeSegment("d", (9,), 3, 1, (), 2, ((1,),)),
    ]
    prefills = [
        PrefillRequest("i", [1, 2], SamplingParams(), [[0]], None, None, 0, None),
        PrefillRequest("f", [3, 4], SamplingParams(), [[0]], None, 4, 2, [1, 2, 3, 4]),
    ]
    # Packed: 3 verify rows, 1 decode row, then one selected row per prefill.
    state = _paged_state(
        _logits(6),
        [("v", verify_state), ("d", decode_state)],
        segments,
        prefills,
        [0, 3, 4, 5, 6],
    )
    rows = top_logit_rows(state, round_index=3, forward_index=9, top_k=4)
    assert [
        (r["req"], r["row"], r["pos"], r["kind"], r["rows"], r["draft"]) for r in rows
    ] == [
        ("v", 0, 5, "verify", 3, 11),
        ("v", 1, 6, "verify", 3, 12),
        ("v", 2, 7, "verify", 3, None),
        ("d", 3, 3, "decode", 1, None),
        ("f", 5, 4, "prefill", 2, None),
    ]
    assert all(r["round"] == 3 and r["forward"] == 9 for r in rows)
    for r in rows:
        assert r["top"][0] == [(r["row"] + 1) % VOCAB, 10.0]
        assert r["top"][1] == [(r["row"] + 2) % VOCAB, 9.875]
        assert len(r["top"]) == 4
        assert r["top"][2][1] >= r["top"][3][1]


def test_intermediate_prefill_rows_are_not_traced() -> None:
    prefill = PrefillRequest(
        "i", [1, 2, 3], SamplingParams(), [[0]], None, None, 0, None
    )
    state = _paged_state(_logits(1), [], [], [prefill], [0, 1])
    assert top_logit_rows(state, round_index=0, forward_index=0) == []


def test_verification_window_commits_through_first_mismatch() -> None:
    def row(forward, index, pos, draft, argmax, kind="verify"):
        return {
            "forward": forward,
            "row": index,
            "pos": pos,
            "kind": kind,
            "rows": 3,
            "draft": draft,
            "top": [[argmax, 10.0], [argmax + 1, 9.0]],
        }

    rows = [
        row(0, 0, 10, 11, 11),  # draft accepted
        row(0, 1, 11, 12, 30),  # correction at position 11
        row(0, 2, 12, None, 40),  # bonus row discarded
        row(1, 0, 12, 41, 41),  # next window, accepted
        row(1, 1, 13, 42, 42),
        row(1, 2, 14, None, 43),  # bonus committed
        row(2, 0, 11, None, 99, kind="decode"),  # recompute rewrites 11
    ]
    committed = emitted_tokens(rows)
    assert {pos: committed[pos]["top"][0][0] for pos in sorted(committed)} == {
        10: 11,
        11: 99,
        12: 41,
        13: 42,
        14: 43,
    }


def test_identical_prompts_are_assigned_by_their_token_streams() -> None:
    prompts = [[1, 2], [1, 2]]
    streams = [[5, 6], [5, 7]]

    def rows(tokens):
        return [
            {
                "forward": i,
                "row": 0,
                "pos": 2 + i,
                "kind": "decode",
                "rows": 1,
                "draft": None,
                "top": [[token, 1.0], [0, 0.0]],
            }
            for i, token in enumerate(tokens)
        ]

    grouped = {"req-b": rows([5, 7]), "req-a": rows([5, 6])}
    assert assign_requests(grouped, prompts, streams) == {0: "req-a", 1: "req-b"}
    with pytest.raises(ValueError):
        assign_requests(
            {"req-a": rows([5, 6]), "req-b": rows([9, 9])}, prompts, streams
        )


def test_ulp_and_tie_classification() -> None:
    assert ulp(20.0) == 0.125  # bfloat16 spacing in [16, 32)
    assert ulp(29.125) == 0.125
    tie = classify_pair(
        {
            "top": [[279, 19.875], [60650, 19.75], [8330, 19.5]],
            "rows": 1,
            "kind": "decode",
        },
        {
            "top": [[60650, 20.0], [279, 19.875], [8330, 19.625]],
            "rows": 8,
            "kind": "verify",
            "draft": 60650,
        },
        279,
        60650,
        7,
    )
    assert tie["label"] == "tie"
    assert abs(tie["baseline"]["gap_ulps"]) <= MAX_TIE_ULPS
    real = classify_pair(
        {"top": [[320, 14.75], [9664, 14.0], [1, 13.0]], "rows": 1, "kind": "decode"},
        {
            "top": [[9664, 41.25], [89630, 25.0], [43436, 24.25]],
            "rows": 3,
            "kind": "verify",
            "draft": 9664,
        },
        320,
        9664,
        7,
    )
    assert real["label"] == "engine-disagreement"
    assert real["top1_logit_difference"] == pytest.approx(41.25 - 14.75)
    assert first_divergence([1, 2, 3], [1, 2, 4]) == 2
    assert first_divergence([1, 2], [1, 2]) is None
