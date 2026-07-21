# SPDX-License-Identifier: Apache-2.0
"""Model-free unit tests for the DSpark speculative-decoding proposer.

A ``SimpleNamespace`` stub drafter + a stub controller drive ``propose`` through
the real lifecycle (prompt seeding at first draft, segment-row context growth,
the ``n_cached == committed_len - 1`` invariant, eligibility gating, pruning).
No MLX model is loaded — ``run_backbone_with_capture`` is monkeypatched out so
``_seed_prompt`` needs no real backbone.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import pytest
from vllm.sampling_params import SamplingParams

pytest.importorskip("vllm", reason="vllm not installed")

import vllm_metal.v1.dspark_proposer as dspark_mod  # noqa: E402
from vllm_metal.v1.dspark_proposer import DSparkProposer  # noqa: E402
from vllm_metal.v1.proposer import ProposeContext  # noqa: E402
from vllm_metal.v1.spec_decode import SpeculativeDecodeController  # noqa: E402

TARGET_LAYER_IDS = [1, 9, 17, 25, 33]
BLOCK_SIZE = 7
MASK_TOKEN_ID = 151669


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        target_layer_ids=TARGET_LAYER_IDS,
        block_size=BLOCK_SIZE,
        mask_token_id=MASK_TOKEN_ID,
    )


def _stub_drafter() -> MagicMock:
    """A fake drafter that records calls + returns a `cap`-length 1-D block."""
    d = MagicMock()
    # sample_block -> array of length `cap` (all zeros) so [int(t) ...] works.
    d.sample_block.side_effect = lambda base_logits, first_prev_token: mx.zeros(
        (base_logits.shape[0] or 1,), dtype=mx.int32
    )
    d.embed.return_value = mx.zeros((1, BLOCK_SIZE, 4))
    d.backbone.return_value = mx.zeros((1, BLOCK_SIZE, 4))
    d.compute_logits.return_value = mx.zeros((1, BLOCK_SIZE, 3))
    d.make_ctx_cache.return_value = [object() for _ in range(5)]
    # update_context records (ctx_offset, n_rows) per call.
    updates: list[tuple[int, int]] = []
    d._updates = updates

    def _update(fused, ctx_offset, ctx_caches):
        updates.append((int(ctx_offset), int(fused.shape[1])))

    d.update_context.side_effect = _update
    return d


def _proposer(drafter: MagicMock | None = None) -> DSparkProposer:
    return DSparkProposer(
        drafter=drafter or _stub_drafter(),
        config=_cfg(),
        runner=SimpleNamespace(
            _model_adapter=SimpleNamespace(_target_backbone=lambda model: None),
            _forward_model=object(),
        ),
        controller=SpeculativeDecodeController(),
    )


def _segment(
    req_id: str,
    *,
    start_row: int = 0,
    num_query_tokens: int = 3,
) -> SimpleNamespace:
    return SimpleNamespace(
        req_id=req_id,
        start_row=start_row,
        num_query_tokens=num_query_tokens,
    )


def _state(token_ids: list[int]) -> SimpleNamespace:
    # sampling_params(temperature=0) satisfies _validate_greedy_sampling.
    return SimpleNamespace(
        token_ids=list(token_ids),
        sampling_params=SamplingParams(temperature=0.0),
    )


def _context(
    *,
    decode_reqs: list[tuple[str, SimpleNamespace]],
    decode_segments: list[SimpleNamespace],
    target_hidden_states: mx.array | None,
    request_states: dict[str, SimpleNamespace] | None = None,
    num_speculative_tokens: int = 2,
    eligible: list | None = None,
) -> ProposeContext:
    """A ProposeContext whose controller.draft_eligible_requests returns `eligible`
    (default: all decode_reqs)."""
    ctx = ProposeContext(
        target_hidden_states=target_hidden_states,
        decode_reqs=decode_reqs,
        decode_segments=decode_segments,
        decode_token_ids=[[s.token_ids[-1]] for _, s in decode_reqs],
        prefill_reqs=[],
        prefill_token_ids=[],
        prefill_result_modes=[],
        request_states=request_states or dict(decode_reqs),
        cu_seqlens=[],
        num_decode_segments=len(decode_segments),
        num_speculative_tokens=num_speculative_tokens,
        logitsprocs=None,
    )
    elig = eligible if eligible is not None else list(decode_reqs)
    ctx_ctrl = SpeculativeDecodeController()
    ctx_ctrl.draft_eligible_requests = lambda *a, **k: elig  # type: ignore[assignment]
    return ctx


# -- needs_target_hidden_states + capture_layer_ids --------------------------


class TestDSparkProtocol:
    def test_capture_layer_ids_from_config(self) -> None:
        p = _proposer()
        assert p.capture_layer_ids == TARGET_LAYER_IDS

    @pytest.mark.parametrize(
        "segments, expected",
        [
            ([], False),
            ([_segment("r0")], True),
            ([_segment("r0"), _segment("r1")], True),
        ],
    )
    def test_needs_target_hidden_states(self, segments: list, expected: bool) -> None:
        p = _proposer()
        assert (
            p.needs_target_hidden_states(segments, has_final_prefill=False) is expected
        )


# -- propose gating ----------------------------------------------------------


class TestDSparkProposeGating:
    def test_returns_none_when_no_spec_tokens(self) -> None:
        p = _proposer()
        ctx = _context(
            decode_reqs=[("r0", _state([1, 2, 3]))],
            decode_segments=[_segment("r0")],
            target_hidden_states=mx.zeros((1, 4)),
            num_speculative_tokens=0,
        )
        assert p.propose(ctx) is None

    def test_returns_none_when_no_hidden_states(self) -> None:
        p = _proposer()
        ctx = _context(
            decode_reqs=[("r0", _state([1, 2, 3]))],
            decode_segments=[_segment("r0")],
            target_hidden_states=None,
        )
        assert p.propose(ctx) is None

    def test_returns_none_when_no_eligible(self) -> None:
        p = _proposer()
        ctx = _context(
            decode_reqs=[("r0", _state([1, 2, 3]))],
            decode_segments=[_segment("r0")],
            target_hidden_states=mx.zeros((1, 4)),
            eligible=[],
        )
        assert p.propose(ctx) is None

    def test_skips_request_without_segment(self) -> None:
        # A request present in decode_reqs but with no decode segment (prefill-only
        # this step) is skipped, not drafted.
        p = _proposer()
        ctx = _context(
            decode_reqs=[("r0", _state([1, 2, 3]))],
            decode_segments=[],  # no segment for r0
            target_hidden_states=mx.zeros((1, 4)),
        )
        assert p.propose(ctx) is None


# -- lifecycle invariant -----------------------------------------------------


class TestDSparkLifecycle:
    def test_first_call_seeds_prompt_via_extra_forward(self, monkeypatch) -> None:
        """First draft: _seed_prompt runs the tapped forward (monkeypatched) and
        sets n_cached = committed_len - 1; no segment-row growth yet."""
        drafter = _stub_drafter()
        p = _proposer(drafter)
        # committed_len = 5 -> prompt is token_ids[:-1] (4 tokens) -> n_cached = 4.
        state = _state([10, 20, 30, 40, 50])
        captured = {}

        def _fake_capture(backbone, ids, *, cache, layer_ids):
            captured["ids"] = ids.tolist()
            captured["layer_ids"] = list(layer_ids)
            return mx.zeros((1, 1, 1)), mx.zeros((1, len(ids[0]), len(layer_ids) * 4))

        monkeypatch.setattr(dspark_mod, "run_backbone_with_capture", _fake_capture)
        # Give _seed_prompt a real-ish backbone (the stub runner returns None).
        p._runner._model_adapter._target_backbone = lambda model: SimpleNamespace(
            layers=[object()] * 36
        )

        ctx = _context(
            decode_reqs=[("r0", state)],
            decode_segments=[_segment("r0", num_query_tokens=3)],
            target_hidden_states=mx.zeros((3, 8)),
        )
        out = p.propose(ctx)
        assert out is not None
        assert out.req_ids == ["r0"]
        assert captured["layer_ids"] == TARGET_LAYER_IDS
        # prompt seeded with token_ids[:-1] (4 tokens); n_cached = 4.
        assert captured["ids"] == [[10, 20, 30, 40]]
        assert p._n_cached["r0"] == 4  # committed_len(5) - 1
        # seed update_context called once with ctx_offset=0, 4 rows.
        assert drafter._updates == [(0, 4)]

    def test_later_call_grows_context_from_segment_rows(self, monkeypatch) -> None:
        """Second draft: ingests (committed_len - 1 - n_cached) segment rows,
        preserving n_cached == committed_len - 1; no re-seed."""
        drafter = _stub_drafter()
        p = _proposer(drafter)
        # Pretend the prompt was already seeded (n_cached = 4, committed_len = 5).
        p._ctx_caches["r0"] = [object()]
        p._n_cached["r0"] = 4

        # Now committed_len = 7 (2 newly-committed since the seed): grow by 2.
        state = _state([10, 20, 30, 40, 50, 60, 70])
        seg = _segment("r0", num_query_tokens=3)  # rows 0..2 of the fused tensor
        hidden = mx.zeros((3, 8))  # 3 rows, k*H

        ctx = _context(
            decode_reqs=[("r0", state)],
            decode_segments=[seg],
            target_hidden_states=hidden,
        )
        out = p.propose(ctx)
        assert out is not None
        # Grew by (7 - 1) - 4 = 2 rows at offset 4 -> n_cached = 6 (= committed_len - 1).
        assert drafter._updates == [(4, 2)]
        assert p._n_cached["r0"] == 6

    def test_no_growth_when_context_already_current(self) -> None:
        """If n_cached == committed_len - 1 already, no ingest."""
        drafter = _stub_drafter()
        p = _proposer(drafter)
        p._ctx_caches["r0"] = [object()]
        p._n_cached["r0"] = 4
        state = _state([10, 20, 30, 40, 50])  # committed_len - 1 == 4 == n_cached
        ctx = _context(
            decode_reqs=[("r0", state)],
            decode_segments=[_segment("r0", num_query_tokens=3)],
            target_hidden_states=mx.zeros((3, 8)),
        )
        p.propose(ctx)
        assert drafter._updates == []  # no growth, but a block was still drafted


# -- helpers -----------------------------------------------------------------


class TestDSparkHelpers:
    def test_segment_rows_slices_first_count(self) -> None:
        p = _proposer()
        seg = _segment("r0", start_row=2, num_query_tokens=5)
        hidden = mx.arange(7 * 3, dtype=mx.float32).reshape(7, 3)
        out = p._segment_rows(hidden, seg, count=3)
        assert out is not None
        assert out.shape == (1, 3, 3)  # [1, count, k*H]
        assert mx.array_equal(out[0], hidden[2:5])

    def test_segment_rows_none_when_count_exceeds_segment(self) -> None:
        p = _proposer()
        seg = _segment("r0", start_row=0, num_query_tokens=2)
        hidden = mx.zeros((2, 3))
        assert p._segment_rows(hidden, seg, count=5) is None

    def test_prune_finished_drops_state(self) -> None:
        p = _proposer()
        p._ctx_caches["keep"] = [object()]
        p._n_cached["keep"] = 3
        p._ctx_caches["drop"] = [object()]
        p._n_cached["drop"] = 5
        p._prune_finished({"keep": _state([1, 2, 3])})
        assert "keep" in p._ctx_caches
        assert "drop" not in p._ctx_caches
        assert p._n_cached.get("drop") is None
