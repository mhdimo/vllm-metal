# SPDX-License-Identifier: Apache-2.0
"""Tests for paged speculative decode metadata helpers."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import mlx.core as mx
import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

from vllm_metal.v1.dspark.sampling import (
    DSparkProposal,
    RequestRandomStreams,
    SamplingTransforms,
)
from vllm_metal.v1.gemma4_mtp import Gemma4MTPDraftSeed
from vllm_metal.v1.spec_decode import (
    PagedDecodeSegment,
    SpeculativeDecodeController,
)


def _state(token_ids: list[int], block_ids: list[int]) -> SimpleNamespace:
    return SimpleNamespace(token_ids=token_ids, block_ids=[block_ids])


def _request_state(
    temperature: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        sampling_params=SamplingParams(temperature=temperature),
    )


def _scheduler_output(
    *,
    scheduled_spec_decode_tokens: dict[str, list[int]],
    num_scheduled_tokens: dict[str, int] | None = None,
    num_invalid_spec_tokens: dict[str, int] | None = None,
) -> SchedulerOutput:
    num_scheduled = num_scheduled_tokens or {
        req_id: len(tokens) + 1
        for req_id, tokens in scheduled_spec_decode_tokens.items()
    }
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled,
        total_num_scheduled_tokens=sum(num_scheduled.values()),
        scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        num_invalid_spec_tokens=num_invalid_spec_tokens,
    )


def _gemma4_mtp_speculative_config() -> SimpleNamespace:
    return SimpleNamespace(
        method="mtp",
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="gemma4_assistant",
                architectures=["Gemma4AssistantForCausalLM"],
            )
        ),
    )


def _logits(token_ids: list[int], vocab_size: int = 16) -> mx.array:
    rows = []
    for token_id in token_ids:
        row = [0.0] * vocab_size
        row[token_id] = 10.0
        rows.append(row)
    return mx.array([rows])


class TestPagedDecodeSegment:
    def test_freezes_single_row_decode_shape(self) -> None:
        segment = PagedDecodeSegment(
            req_id="r0",
            input_token_ids=(9,),
            start_row=0,
            num_query_tokens=1,
            draft_token_ids=(),
            cache_start_pos=7,
            block_ids=((11, 12),),
        )

        assert segment.req_id == "r0"
        assert segment.input_token_ids == (9,)
        assert segment.draft_token_ids == ()
        assert segment.draft_verification_rows == ()
        assert segment.bonus_row == 0

        with pytest.raises(FrozenInstanceError):
            segment.start_row = 1  # type: ignore[misc]


class TestBuildPagedDecodeSegments:
    def test_single_row_decode_matches_current_shape(self) -> None:
        segments = SpeculativeDecodeController().build_decode_segments(
            [("r0", _state([5, 9], [41, 42]))],
            scheduled_spec_decode_tokens={},
            paged_request_seq_lens={"r0": 7},
        )

        assert len(segments) == 1
        segment = segments[0]
        assert segment.req_id == "r0"
        assert segment.input_token_ids == (9,)
        assert segment.start_row == 0
        assert segment.num_query_tokens == 1
        assert segment.draft_token_ids == ()
        assert segment.cache_start_pos == 7
        assert segment.block_ids == ((41, 42),)
        assert segment.draft_verification_rows == ()
        assert segment.bonus_row == 0

    def test_single_row_decode_uses_len_minus_one_fallback(self) -> None:
        segments = SpeculativeDecodeController().build_decode_segments(
            [("r0", _state([5, 9, 17], [41, 42]))],
            scheduled_spec_decode_tokens=None,
            paged_request_seq_lens={},
        )

        assert segments[0].cache_start_pos == 2

    def test_draft_tokens_expand_the_row_span(self) -> None:
        segments = SpeculativeDecodeController().build_decode_segments(
            [("r0", _state([5, 9], [41, 42]))],
            scheduled_spec_decode_tokens={"r0": [23, 24]},
            paged_request_seq_lens={"r0": 7},
        )

        segment = segments[0]
        assert segment.input_token_ids == (9, 23, 24)
        assert segment.num_query_tokens == 3
        assert segment.draft_token_ids == (23, 24)
        assert segment.start_row == 0
        assert segment.draft_verification_rows == (0, 1)
        assert segment.bonus_row == 2

    def test_mixed_batch_uses_cumulative_start_rows(self) -> None:
        segments = SpeculativeDecodeController().build_decode_segments(
            [
                ("r0", _state([1, 2], [10])),
                ("r1", _state([3, 4, 5], [11])),
                ("r2", _state([6], [12])),
            ],
            scheduled_spec_decode_tokens={
                "r1": [21, 22],
                "r2": [31],
            },
            paged_request_seq_lens={"r0": 1, "r1": 2, "r2": 0},
        )

        assert [segment.start_row for segment in segments] == [0, 1, 4]
        assert [segment.num_query_tokens for segment in segments] == [1, 3, 2]
        assert segments[1].draft_verification_rows == (1, 2)
        assert segments[1].bonus_row == 3
        assert segments[2].draft_verification_rows == (4,)
        assert segments[2].bonus_row == 5

    def test_rejects_handoff_for_request_outside_decode_set(self) -> None:
        with pytest.raises(ValueError, match="outside the current decode set"):
            SpeculativeDecodeController().build_decode_segments(
                [("r0", _state([1, 2], [10]))],
                scheduled_spec_decode_tokens={"missing": [3]},
                paged_request_seq_lens={"r0": 1},
            )

    def test_rejects_invalid_draft_token_sentinel(self) -> None:
        with pytest.raises(NotImplementedError, match="invalid draft-token"):
            SpeculativeDecodeController().build_decode_segments(
                [("r0", _state([1, 2], [10]))],
                scheduled_spec_decode_tokens={"r0": [-1]},
                paged_request_seq_lens={"r0": 1},
            )


class TestSpecDecodePolicy:
    def test_empty_scheduled_tokens_are_supported(self) -> None:
        SpeculativeDecodeController().validate_supported(
            _scheduler_output(scheduled_spec_decode_tokens={}),
            (),
            paged_attention_enabled=False,
            is_hybrid=True,
        )

    def test_gemma4_mtp_rejects_async_scheduling(self) -> None:
        with pytest.raises(NotImplementedError, match="no-async-scheduling"):
            SpeculativeDecodeController().validate_supported(
                _scheduler_output(scheduled_spec_decode_tokens={}),
                (),
                paged_attention_enabled=True,
                is_hybrid=False,
                use_async_scheduling=True,
                speculative_config=_gemma4_mtp_speculative_config(),
            )

    def test_dspark_may_run_under_async_scheduling(self) -> None:
        SpeculativeDecodeController().validate_supported(
            _scheduler_output(scheduled_spec_decode_tokens={}),
            (),
            paged_attention_enabled=True,
            is_hybrid=False,
            use_async_scheduling=True,
            speculative_config=SimpleNamespace(method="dspark"),
        )
        with pytest.raises(NotImplementedError, match="no-async-scheduling"):
            SpeculativeDecodeController().validate_supported(
                _scheduler_output(scheduled_spec_decode_tokens={}),
                (),
                paged_attention_enabled=True,
                is_hybrid=False,
                use_async_scheduling=True,
                speculative_config=SimpleNamespace(method="ngram"),
            )

    def test_retained_drafts_fill_placeholder_slots_and_report_unused(self) -> None:
        output = _scheduler_output(
            scheduled_spec_decode_tokens={
                "a": [-1, -1, -1],
                "b": [-1, -1, -1],
                "c": [5, 6],
            }
        )
        resolved, runner_invalid = (
            SpeculativeDecodeController.substitute_retained_drafts(
                output, {"a": [7, 8], "c": [9], "d": [1]}
            )
        )
        # "a": two retained drafts in three slots; "b": none; "c": the
        # synchronous handoff passes through; "d": not scheduled.
        assert resolved == {"a": (7, 8), "c": (5, 6)}
        assert runner_invalid == {"a": 1, "b": 3}
        assert output.num_invalid_spec_tokens == {"a": 1, "b": 3}
        # More retained drafts than slots are cut to the slots.
        output = _scheduler_output(scheduled_spec_decode_tokens={"a": [-1, -1]})
        resolved, runner_invalid = (
            SpeculativeDecodeController.substitute_retained_drafts(
                output, {"a": [7, 8, 9]}
            )
        )
        assert resolved == {"a": (7, 8)} and runner_invalid == {}
        assert output.num_invalid_spec_tokens is None
        with pytest.raises(ValueError, match="real token ids"):
            SpeculativeDecodeController.substitute_retained_drafts(
                _scheduler_output(scheduled_spec_decode_tokens={"a": [-1]}),
                {"a": [-1]},
            )

    def test_validate_supported_accepts_runner_resolved_slots(self) -> None:
        controller = SpeculativeDecodeController()
        output = _scheduler_output(
            scheduled_spec_decode_tokens={"a": [-1, -1, -1], "b": [-1, -1, -1]}
        )
        resolved, runner_invalid = controller.substitute_retained_drafts(
            output, {"a": [7, 8]}
        )
        # Fewer verified rows than scheduled slots is the runner's accounting.
        controller.validate_supported(
            output,
            [("a", _request_state()), ("b", _request_state())],
            paged_attention_enabled=True,
            is_hybrid=False,
            use_async_scheduling=True,
            speculative_config=SimpleNamespace(method="dspark"),
            resolved_spec_tokens=resolved,
            runner_invalid_counts=runner_invalid,
        )
        # A scheduler-reported invalid count on another request still fails closed.
        output.num_invalid_spec_tokens = {**output.num_invalid_spec_tokens, "c": 1}
        with pytest.raises(NotImplementedError, match="scheduler-invalid"):
            controller.validate_supported(
                output,
                [
                    ("a", _request_state()),
                    ("b", _request_state()),
                    ("c", _request_state()),
                ],
                paged_attention_enabled=True,
                is_hybrid=False,
                use_async_scheduling=True,
                speculative_config=SimpleNamespace(method="dspark"),
                resolved_spec_tokens=resolved,
                runner_invalid_counts=runner_invalid,
            )

    def test_non_paged_scheduled_tokens_are_rejected(self) -> None:
        with pytest.raises(NotImplementedError, match="requires paged attention"):
            SpeculativeDecodeController().validate_supported(
                _scheduler_output(scheduled_spec_decode_tokens={"r0": [1]}),
                [("r0", _request_state())],
                paged_attention_enabled=False,
                is_hybrid=False,
            )

    def test_hybrid_scheduled_tokens_are_rejected(self) -> None:
        with pytest.raises(NotImplementedError, match="hybrid models"):
            SpeculativeDecodeController().validate_supported(
                _scheduler_output(scheduled_spec_decode_tokens={"r0": [1]}),
                [("r0", _request_state())],
                paged_attention_enabled=True,
                is_hybrid=True,
            )

    def test_rejects_invalid_draft_token_sentinel(self) -> None:
        with pytest.raises(NotImplementedError, match="invalid draft-token"):
            SpeculativeDecodeController().validate_supported(
                _scheduler_output(scheduled_spec_decode_tokens={"r0": [7, -1]}),
                [("r0", _request_state())],
                paged_attention_enabled=True,
                is_hybrid=False,
            )

    def test_rejects_scheduler_invalid_spec_tokens(self) -> None:
        with pytest.raises(NotImplementedError, match="scheduler-invalid"):
            SpeculativeDecodeController().validate_supported(
                _scheduler_output(
                    scheduled_spec_decode_tokens={"r0": [-1]},
                    num_invalid_spec_tokens={"r0": 1},
                ),
                [("r0", _request_state())],
                paged_attention_enabled=True,
                is_hybrid=False,
            )

    def test_rejects_handoff_for_request_outside_decode_set(self) -> None:
        with pytest.raises(ValueError, match="outside the current decode set"):
            SpeculativeDecodeController().validate_supported(
                _scheduler_output(scheduled_spec_decode_tokens={"missing": [1]}),
                [("r0", _request_state())],
                paged_attention_enabled=True,
                is_hybrid=False,
            )

    def test_rejects_empty_handoff_for_request_outside_decode_set(self) -> None:
        with pytest.raises(ValueError, match="outside the current decode set"):
            SpeculativeDecodeController().validate_supported(
                _scheduler_output(scheduled_spec_decode_tokens={"missing": []}),
                [("r0", _request_state())],
                paged_attention_enabled=True,
                is_hybrid=False,
            )

    def test_rejects_mismatched_scheduler_token_accounting(self) -> None:
        with pytest.raises(ValueError, match="inconsistent token accounting"):
            SpeculativeDecodeController().validate_supported(
                _scheduler_output(
                    scheduled_spec_decode_tokens={"r0": [1, 2]},
                    num_scheduled_tokens={"r0": 2},
                ),
                [("r0", _request_state())],
                paged_attention_enabled=True,
                is_hybrid=False,
            )


class TestGemma4MTPDraftSeeds:
    def test_decode_seeds_use_last_accepted_target_row(self) -> None:
        seeds = SpeculativeDecodeController().build_gemma4_mtp_draft_seeds(
            decode_reqs=[
                ("r0", _request_state()),
                ("r1", _request_state(temperature=0.7)),
            ],
            decode_segments=[
                PagedDecodeSegment(
                    req_id="r0",
                    input_token_ids=(5, 6),
                    start_row=0,
                    num_query_tokens=2,
                    draft_token_ids=(6,),
                    cache_start_pos=4,
                    block_ids=((1,),),
                ),
                PagedDecodeSegment(
                    req_id="r1",
                    input_token_ids=(7,),
                    start_row=2,
                    num_query_tokens=1,
                    draft_token_ids=(),
                    cache_start_pos=9,
                    block_ids=((2,),),
                ),
            ],
            decode_token_ids=[[6, 8], [9]],
            prefill_reqs=[],
            prefill_token_ids=[],
            prefill_result_modes=[],
            request_states={},
            cu_seqlens=[0, 2, 3],
            num_decode_segments=2,
        )

        assert seeds == (
            Gemma4MTPDraftSeed(
                req_id="r0",
                token_id=8,
                target_hidden_row=1,
                target_position=5,
                block_ids=((1,),),
            ),
        )

    def test_prefill_seeds_skip_intermediate_chunks(self) -> None:
        final_prefill = SimpleNamespace(
            req_id="p0",
            token_ids=[1, 2, 3],
            block_ids=[[4]],
            start_pos=10,
        )
        intermediate_prefill = SimpleNamespace(
            req_id="p1",
            token_ids=[4, 5],
            block_ids=[[5]],
            start_pos=20,
        )

        seeds = SpeculativeDecodeController().build_gemma4_mtp_draft_seeds(
            decode_reqs=[],
            decode_segments=[],
            decode_token_ids=[],
            prefill_reqs=[final_prefill, intermediate_prefill],
            prefill_token_ids=[11, 12],
            prefill_result_modes=["new_final", "intermediate"],
            request_states={
                "p0": _request_state(),
                "p1": _request_state(),
            },
            cu_seqlens=[0, 3, 5],
            num_decode_segments=0,
        )

        assert seeds == (
            Gemma4MTPDraftSeed(
                req_id="p0",
                token_id=11,
                target_hidden_row=2,
                target_position=12,
                block_ids=((4,),),
            ),
        )


class TestVerifyGreedySpecDecode:
    def test_specific_token_logprobs_are_not_draft_eligible(self) -> None:
        state = SimpleNamespace(
            sampling_params=SamplingParams(
                temperature=0.0,
                logprob_token_ids=[0, 3],
            )
        )

        assert not SpeculativeDecodeController().can_draft_greedy("r0", state)

    def test_accepts_all_drafts_and_emits_bonus_token(self) -> None:
        segment = PagedDecodeSegment(
            req_id="r0",
            input_token_ids=(6, 7, 8),
            start_row=0,
            num_query_tokens=3,
            draft_token_ids=(7, 8),
            cache_start_pos=1,
            block_ids=((0,),),
        )

        output = SpeculativeDecodeController().verify_greedy(
            _logits([7, 8, 9]),
            [("r0", _request_state())],
            (segment,),
        )

        assert output == [[7, 8, 9]]

    def test_rejects_first_mismatched_draft_and_stops_before_bonus(self) -> None:
        segment = PagedDecodeSegment(
            req_id="r0",
            input_token_ids=(6, 7, 8),
            start_row=0,
            num_query_tokens=3,
            draft_token_ids=(7, 8),
            cache_start_pos=1,
            block_ids=((0,),),
        )

        output = SpeculativeDecodeController().verify_greedy(
            _logits([7, 5, 9]),
            [("r0", _request_state())],
            (segment,),
        )

        assert output == [[7, 5]]

    def test_rejects_non_greedy_sampling(self) -> None:
        segment = PagedDecodeSegment(
            req_id="r0",
            input_token_ids=(6, 7),
            start_row=0,
            num_query_tokens=2,
            draft_token_ids=(7,),
            cache_start_pos=1,
            block_ids=((0,),),
        )

        with pytest.raises(NotImplementedError, match="greedy sampling"):
            SpeculativeDecodeController().verify_greedy(
                _logits([7, 9]),
                [("r0", _request_state(temperature=0.7))],
                (segment,),
            )


def _sharp_logits(token_ids: list[int], vocab_size: int = 16) -> mx.array:
    """Rows whose float32 softmax is exactly a point mass on ``token_ids``."""
    rows = []
    for token_id in token_ids:
        row = [0.0] * vocab_size
        row[token_id] = 40.0
        rows.append(row)
    return mx.array([rows])


def _segment(
    req_id: str, start_row: int, drafts: tuple[int, ...]
) -> PagedDecodeSegment:
    return PagedDecodeSegment(
        req_id=req_id,
        input_token_ids=(6, *drafts),
        start_row=start_row,
        num_query_tokens=len(drafts) + 1,
        draft_token_ids=drafts,
        cache_start_pos=1,
        block_ids=((0,),),
    )


def _record(
    state, token_ids, *, anchor_position=1, anchor_token=6, owner=None, vocab_size=16
):
    rows = []
    for token_id in token_ids:
        row = [0.0] * vocab_size
        row[token_id] = 0.5
        row[(token_id + 5) % vocab_size] = 0.5
        rows.append(row)
    return DSparkProposal(
        owner=state if owner is None else owner,
        anchor_position=anchor_position,
        anchor_token=anchor_token,
        token_ids=list(token_ids),
        distributions=mx.array(rows, dtype=mx.float32),
        transforms=SamplingTransforms.from_params(state.sampling_params),
        streams=RequestRandomStreams.for_request(1, engine_seed=0, ordinal=1),
    )


class TestVerifyMixedModes:
    def test_mixed_batch_dispatches_by_sampling_mode(self) -> None:
        greedy, stochastic = _request_state(), _request_state(temperature=1.0)
        segments = (_segment("g", 0, (7, 8)), _segment("s", 3, (7, 8)))
        # Greedy rejects at its second draft; the stochastic target puts all
        # its float32 mass on the drafts, so every draft is accepted and the
        # bonus row (a point mass) supplies the last token.
        logits = _sharp_logits([7, 5, 9, 7, 8, 9])
        output = SpeculativeDecodeController().verify(
            logits,
            [("g", greedy), ("s", stochastic)],
            segments,
            proposals={"s": _record(stochastic, [7, 8])},
            vocab_size=16,
        )
        assert output == [[7, 5], [7, 8, 9]]

    def test_stochastic_rejection_samples_the_residual(self) -> None:
        stochastic = _request_state(temperature=0.7)
        output = SpeculativeDecodeController().verify(
            _sharp_logits([7, 5, 9]),
            [("s", stochastic)],
            (_segment("s", 0, (7, 8)),),
            proposals={"s": _record(stochastic, [7, 8])},
            vocab_size=16,
        )
        # p_1 is a point mass on 5 while q_1 has no mass there: the residual
        # is that point mass, so the second token is 5 and the bonus is skipped.
        assert output == [[7, 5]]

    def test_scheduler_clipped_drafts_use_the_record_prefix(self) -> None:
        stochastic = _request_state(temperature=1.0)
        output = SpeculativeDecodeController().verify(
            _sharp_logits([7, 3]),
            [("s", stochastic)],
            (_segment("s", 0, (7,)),),
            proposals={"s": _record(stochastic, [7, 8, 4])},
            vocab_size=16,
        )
        assert output == [[7, 3]]

    @pytest.mark.parametrize(
        "fault", ["missing", "owner", "anchor", "tokens", "transforms", "vocab"]
    )
    def test_stochastic_drafts_without_a_matching_record_fail_closed(
        self, fault
    ) -> None:
        stochastic = _request_state(temperature=1.0)
        record = _record(stochastic, [7, 8])
        if fault == "owner":
            record = _record(stochastic, [7, 8], owner=object())
        elif fault == "anchor":
            record = _record(stochastic, [7, 8], anchor_position=2)
        elif fault == "tokens":
            record = _record(stochastic, [7, 9])
        elif fault == "transforms":
            record = _record(_request_state(temperature=0.5), [7, 8])
        elif fault == "vocab":
            record = _record(stochastic, [7, 8], vocab_size=12)
        proposals = {} if fault == "missing" else {"s": record}
        with pytest.raises(RuntimeError, match="no matching proposal record"):
            SpeculativeDecodeController().verify(
                _sharp_logits([7, 8, 9]),
                [("s", stochastic)],
                (_segment("s", 0, (7, 8)),),
                proposals=proposals,
                vocab_size=16,
            )

    def test_stochastic_drafts_need_records_and_vocab(self) -> None:
        stochastic = _request_state(temperature=1.0)
        with pytest.raises(NotImplementedError, match="greedy sampling only"):
            SpeculativeDecodeController().verify(
                _sharp_logits([7, 8, 9]),
                [("s", stochastic)],
                (_segment("s", 0, (7, 8)),),
            )

    def test_undraftable_parameters_with_drafts_raise(self) -> None:
        state = SimpleNamespace(
            sampling_params=SamplingParams(temperature=0.7, presence_penalty=0.5)
        )
        with pytest.raises(NotImplementedError, match="not draftable"):
            SpeculativeDecodeController().verify(
                _sharp_logits([7, 8, 9]),
                [("s", state)],
                (_segment("s", 0, (7, 8)),),
                proposals={},
                vocab_size=16,
            )

    def test_empty_batch_and_greedy_only_batch(self) -> None:
        controller = SpeculativeDecodeController()
        assert controller.verify(_sharp_logits([7]), [], ()) == []
        greedy = _request_state()
        assert controller.verify(
            _sharp_logits([7, 8, 9]), [("g", greedy)], (_segment("g", 0, (7, 8)),)
        ) == [[7, 8, 9]]

    @pytest.mark.parametrize(
        "params,mode",
        [
            ({"temperature": 0.0}, "greedy"),
            # vLLM normalizes a greedy request's top-k/top-p away.
            ({"temperature": 0.0, "top_k": 3}, "greedy"),
            ({"temperature": 0.8}, "stochastic"),
            ({"temperature": 0.8, "top_k": 3, "top_p": 0.9, "seed": 2}, "stochastic"),
            ({"temperature": 0.8, "min_p": 0.1}, None),
            ({"temperature": 0.8, "logit_bias": {1: 2.0}}, None),
            ({"temperature": 0.8, "repetition_penalty": 1.1}, None),
            ({"temperature": 0.8, "logprobs": 1}, None),
            ({"temperature": 0.8, "bad_words_token_ids": [[3]]}, None),
            ({"temperature": 0.8, "allowed_token_ids": [1, 2]}, None),
        ],
    )
    def test_draft_mode_classification(self, params, mode) -> None:
        if "bad_words_token_ids" in params:
            # The tokenizer fills this read-only field on real requests.
            base = SamplingParams(temperature=0.8)
            fields = {
                name: getattr(base, name)
                for name in (
                    "temperature",
                    "top_k",
                    "top_p",
                    "min_p",
                    "frequency_penalty",
                    "presence_penalty",
                    "repetition_penalty",
                    "num_logprobs",
                    "allowed_token_ids",
                    "logit_bias",
                    "structured_outputs",
                )
            }
            sampling_params = SimpleNamespace(**{**fields, **params})
            state = SimpleNamespace(sampling_params=sampling_params)
        else:
            state = SimpleNamespace(sampling_params=SamplingParams(**params))
        controller = SpeculativeDecodeController()
        assert controller.draft_mode(state) == mode
        assert controller.can_draft("r", state, allow_stochastic=True) == (
            mode is not None
        )
        assert controller.can_draft_greedy("r", state) == (mode == "greedy")


class TestSchedulerPaddedDrafts:
    """vLLM 0.25 pads a newly admitted decode request with placeholder drafts."""

    def test_padded_drafts_are_dropped(self) -> None:
        scheduler_output = _scheduler_output(
            scheduled_spec_decode_tokens={"r0": [-1, -1]},
        )

        active = SpeculativeDecodeController.active_spec_decode_tokens(scheduler_output)

        assert active == {}

    @pytest.mark.parametrize(
        ("spec_tokens", "num_scheduled", "invalid_counts"),
        [
            ([7, 8], None, None),
            ([-1, -1], None, {"r0": 2}),
            ([-2, -2], None, None),
            ([-1, -1], {"r0": 5}, None),
        ],
        ids=[
            "real_drafts",
            "grammar_rejected",
            "foreign_sentinel",
            "mismatched_accounting",
        ],
    )
    def test_non_padding_handoffs_are_kept(
        self,
        spec_tokens: list[int],
        num_scheduled: dict[str, int] | None,
        invalid_counts: dict[str, int] | None,
    ) -> None:
        scheduler_output = _scheduler_output(
            scheduled_spec_decode_tokens={"r0": spec_tokens},
            num_scheduled_tokens=num_scheduled,
            num_invalid_spec_tokens=invalid_counts,
        )

        active = SpeculativeDecodeController.active_spec_decode_tokens(scheduler_output)

        assert active == {"r0": tuple(spec_tokens)}

    def test_padded_request_outside_decode_set_is_accepted(self) -> None:
        scheduler_output = _scheduler_output(
            scheduled_spec_decode_tokens={"new": [-1, -1]},
        )

        SpeculativeDecodeController().validate_supported(
            scheduler_output,
            [],
            paged_attention_enabled=True,
            is_hybrid=False,
        )

    def test_padded_drafts_reach_build_decode_segments_as_zero_drafts(self) -> None:
        controller = SpeculativeDecodeController()
        scheduler_output = _scheduler_output(
            scheduled_spec_decode_tokens={"r0": [-1, -1]},
        )
        decode_reqs = [("r0", _state([5, 9], [41, 42]))]

        segments = controller.build_decode_segments(
            decode_reqs,
            controller.active_spec_decode_tokens(scheduler_output),
            paged_request_seq_lens={"r0": 7},
        )

        assert len(segments) == 1
        assert segments[0].num_query_tokens == 1
        assert segments[0].input_token_ids == (9,)
        assert segments[0].draft_token_ids == ()
        assert segments[0].bonus_row == 0
