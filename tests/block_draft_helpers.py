# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for block-draft proposer and prefix-cache tests."""

import mlx.core as mx
import numpy as np

from vllm_metal.v1.dspark_proposer import DSparkProposer
from vllm_metal.v1.model_runner import PrefillRequest
from vllm_metal.v1.proposer import ProposeContext


def _dense_tokens(proposer, anchors, features, width):
    if isinstance(proposer, DSparkProposer):
        return proposer.draft_model.draft(anchors, features, num_draft_tokens=width)[0]
    return mx.argmax(
        proposer.model.draft_logits(
            anchors,
            features,
            num_draft_tokens=width,
            embed=proposer.embed,
            project=proposer.project,
        ),
        axis=-1,
    )


def _features(count):
    return tuple(mx.random.normal((count, 64)).astype(mx.float16) for _ in range(3))


def _assert_committed(proposer, blocks, features):
    """Compare committed draft KV with independent full-context projections."""
    for layer, (keys, values) in enumerate(
        proposer.model._project_context([f[None] for f in features])
    ):
        for stored, expected in (
            (proposer.cache.cache.key_caches[layer], keys),
            (proposer.cache.cache.value_caches[layer], values),
        ):
            block_size = stored.shape[1]
            actual = mx.stack(
                [
                    stored[blocks[p // block_size], p % block_size]
                    for p in range(len(features[0]))
                ]
            )
            np.testing.assert_allclose(
                np.array(actual),
                np.array(expected[0].transpose(1, 0, 2)),
                atol=0.004,
                rtol=0.004,
            )


def _prefill(state, features, start, final):
    count = features[0].shape[0]
    return ProposeContext(
        target_hidden_states=None,
        target_aux_hidden_states=features,
        decode_reqs=[],
        decode_segments=[],
        decode_token_ids=[],
        prefill_reqs=[
            PrefillRequest(
                req_id="r",
                token_ids=[1] * count,
                sampling_params=state.sampling_params,
                block_ids=state.block_ids,
                generator=None,
                prompt_len=None,
                full_prompt_token_ids=None,
                start_pos=start,
            )
        ],
        prefill_token_ids=[2],
        prefill_result_modes=["final" if final else "intermediate"],
        request_states={"r": state},
        cu_seqlens=[0, count],
        num_decode_segments=0,
        num_speculative_tokens=3,
        finished_req_ids=set(),
    )
