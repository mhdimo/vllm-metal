# SPDX-License-Identifier: Apache-2.0
"""DSpark (DeepSeek EAGLE3+Markov) speculative-decoding proposer for Metal.

DSpark is an EAGLE-family drafter: a small backbone that cross-attends over the
*target's* fused hidden states (captured from a selection of target layers),
plus a rank-256 Markov head + optional confidence head. Unlike Gemma4 MTP
(which shares the target KV), DSpark keeps its **own** per-layer context KV
(:class:`CtxCache`) that grows with the committed tokens — so the proposer is
stateful per request, closer to :class:`DraftModelProposer` in shape but
consuming target hidden states (EAGLE-style) instead of running autoregressively.

Lifecycle (per request), mirroring the reference implementation:

- **First draft** (``n_cached == 0``): seed the drafter context with the prompt's
  fused hidden via a standalone tapped target forward (a fresh, non-paged cache,
  so it works regardless of how the prompt was prefilled/chunked). One-time
  O(prompt) cost per request.
- **Each later draft**: ingest the newly-committed positions' fused hidden from
  this step's decode segment (the first ``(committed_len - 1) - n_cached`` rows
  — the anchor + accepted drafts), then draft a block.

The context invariant is ``n_cached == committed_len - 1`` (context holds every
committed token except the current pending, which seeds the draft block).

Greedy only (matches the shared :meth:`verify_greedy` verifier). Per-request
drafting in v1 (batched drafter forward is a follow-up). Drafter + config are
vendored under :mod:`vllm_metal.v1.dspark`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx
from vllm.logger import init_logger
from vllm.v1.outputs import DraftTokenIds

from vllm_metal.v1.dspark.loader import DSparkConfig, DSparkDrafter
from vllm_metal.v1.hidden_state_tap import run_backbone_with_capture

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from vllm_metal.v1.model_runner import MetalModelRunner, RequestState
    from vllm_metal.v1.proposer import ProposeContext
    from vllm_metal.v1.spec_decode import (
        PagedDecodeSegment,
        SpeculativeDecodeController,
    )

logger = init_logger(__name__)


class DSparkProposer:
    """:class:`vllm_metal.v1.proposer.MetalProposer` backed by a DSpark drafter."""

    def __init__(
        self,
        *,
        drafter: DSparkDrafter,
        config: DSparkConfig,
        runner: MetalModelRunner,
        controller: SpeculativeDecodeController,
    ) -> None:
        self._drafter = drafter
        self._config = config
        self._runner = runner
        self._controller = controller
        # Layers of the TARGET the drafter was trained against — the runner
        # captures exactly these (fused) for us each decode step.
        self.capture_layer_ids: list[int] = list(config.target_layer_ids)
        self._block_size = int(config.block_size)
        self._mask_token_id = int(config.mask_token_id)
        # Per-request drafter context. CtxCache is append-only (grows with
        # committed tokens); n_cached is how many committed positions are in it.
        self._ctx_caches: dict[str, list] = {}
        self._n_cached: dict[str, int] = {}

    # -- MetalProposer protocol ---------------------------------------------

    def needs_target_hidden_states(
        self,
        decode_segments: Sequence[PagedDecodeSegment],
        *,
        has_final_prefill: bool,
    ) -> bool:
        # We consume the target's fused hidden each decode step to grow the
        # drafter context. The prompt itself is seeded via a separate forward
        # (see _seed_prompt), so we do not need prefill hidden from the runner.
        return bool(decode_segments)

    def propose(self, ctx: ProposeContext) -> DraftTokenIds | None:
        num_speculative_tokens = ctx.num_speculative_tokens
        if num_speculative_tokens <= 0 or ctx.target_hidden_states is None:
            return None

        self._prune_finished(ctx.request_states)
        eligible = self._controller.draft_eligible_requests(
            ctx.decode_reqs,
            ctx.decode_token_ids,
            ctx.prefill_reqs,
            ctx.prefill_result_modes,
            ctx.request_states,
            logitsprocs=ctx.logitsprocs,
        )
        if not eligible:
            return None

        segment_by_id = {seg.req_id: seg for seg in ctx.decode_segments}
        cap = min(num_speculative_tokens, self._block_size)

        req_ids: list[str] = []
        rows: list[list[int]] = []
        for req_id, state in eligible:
            segment = segment_by_id.get(req_id)
            if segment is None:
                continue  # prefill-only this step; no decode hidden to grow from
            draft_row = self._draft_one(ctx, state, segment, cap)
            if draft_row is None:
                continue
            req_ids.append(req_id)
            rows.append(draft_row)

        if not req_ids:
            return None
        return DraftTokenIds(req_ids=req_ids, draft_token_ids=rows)

    # -- internals ----------------------------------------------------------

    def _draft_one(
        self,
        ctx: ProposeContext,
        state: RequestState,
        segment: PagedDecodeSegment,
        cap: int,
    ) -> list[int] | None:
        req_id = segment.req_id
        token_ids = state.token_ids
        committed_len = len(token_ids)
        if committed_len < 1:
            return None
        pending = int(token_ids[-1])

        ctx_caches = self._ctx_caches.get(req_id)
        if ctx_caches is None:
            # First draft for this request: seed the drafter context with the
            # prompt (every committed token except the pending) via a standalone
            # tapped forward. n_cached becomes len(token_ids) - 1.
            if not self._seed_prompt(req_id, token_ids[:-1]):
                return None
            ctx_caches = self._ctx_caches[req_id]
        else:
            # Grow the context with the newly-committed positions: the first
            # (committed_len - 1 - n_cached) rows of this step's decode segment
            # (the previous pending + accepted drafts).
            new_to_ingest = (committed_len - 1) - self._n_cached[req_id]
            if new_to_ingest > 0:
                fused = self._segment_rows(
                    ctx.target_hidden_states, segment, new_to_ingest
                )
                if fused is None:
                    return None
                self._drafter.update_context(
                    fused,
                    ctx_offset=self._n_cached[req_id],
                    ctx_caches=ctx_caches,
                )
                self._n_cached[req_id] += new_to_ingest

        # Draft a block: embed [pending, masks], cross-attend over the context,
        # take logits over the first `cap` positions, greedy + Markov correction.
        block_ids = [pending] + [self._mask_token_id] * (self._block_size - 1)
        noise = self._drafter.embed(mx.array([block_ids]))
        block_hidden = self._drafter.backbone(noise, self._n_cached[req_id], ctx_caches)
        base_logits = self._drafter.compute_logits(block_hidden[:, :cap, :])[0]
        draft = self._drafter.sample_block(base_logits, first_prev_token=pending)
        mx.eval(draft)
        draft_list = draft.tolist()
        assert isinstance(draft_list, list)  # 1-D block -> list of scalars
        return [int(t) for t in draft_list]

    def _segment_rows(
        self,
        target_hidden_states: mx.array,
        segment: PagedDecodeSegment,
        count: int,
    ) -> mx.array | None:
        """First ``count`` fused rows of this request's decode segment, as
        ``[1, count, k*H]`` (the shape ``update_context`` expects)."""
        start = segment.start_row
        if start + count > segment.start_row + segment.num_query_tokens:
            return None
        rows = target_hidden_states[start : start + count]  # [count, k*H]
        if rows.shape[0] != count:
            return None
        return rows[None, :, :]  # [1, count, k*H]

    def _seed_prompt(self, req_id: str, prompt_ids: Sequence[int]) -> bool:
        """Seed the drafter context with the prompt's fused hidden via a
        standalone tapped target forward (fresh, non-paged cache)."""
        if not prompt_ids:
            ctx_caches = self._drafter.make_ctx_cache()
            self._ctx_caches[req_id] = ctx_caches
            self._n_cached[req_id] = 0
            return True
        body = self._runner._model_adapter._target_backbone(self._runner._forward_model)
        if body is None:
            logger.warning(
                "DSpark: cannot seed prompt — target has no `.model` backbone"
            )
            return False
        ids = mx.array([list(prompt_ids)])
        _, fused = run_backbone_with_capture(
            body,
            ids,
            cache=[None] * len(body.layers),
            layer_ids=self.capture_layer_ids,
        )
        ctx_caches = self._drafter.make_ctx_cache()
        self._drafter.update_context(fused, ctx_offset=0, ctx_caches=ctx_caches)
        self._ctx_caches[req_id] = ctx_caches
        self._n_cached[req_id] = int(len(prompt_ids))
        return True

    def _prune_finished(self, request_states: Mapping[str, RequestState]) -> None:
        for req_id in list(self._ctx_caches.keys()):
            if req_id not in request_states:
                self._ctx_caches.pop(req_id, None)
                self._n_cached.pop(req_id, None)
