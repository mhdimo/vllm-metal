# SPDX-License-Identifier: Apache-2.0
"""Reusable hidden-state tap: capture a target backbone's residual stream after
an arbitrary selection of layers and fuse them.

Both the existing Gemma4 MTP path (last layer) and DSpark (several specific
intermediate layers) become users of this one helper. Mirrors the explicit
transformer-layer traversal used by the reference implementation: embed,
mask, loop over layers calling ``layer(h, mask, c)``, capture the residual
after each requested index, concatenate along the feature axis.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def capture_layer_hidden_states(
    backbone: Any,
    input_ids: mx.array,
    *,
    cache: Any,
    layer_ids: list[int],
) -> mx.array:
    """Run ``backbone``'s layer loop and return the residuals after ``layer_ids``.

    Args:
        backbone: an MLX transformer body exposing ``embed_tokens``, ``layers``,
            and ``norm`` — i.e. what ``_target_backbone`` returns
            (``text_model(model).model``).
        input_ids: ``[batch, tokens]``.
        cache: per-layer cache list, passed straight through to each layer. The
            paged KV routing for patched models happens inside the attention via
            the active step context, not via this arg.
        layer_ids: 0-indexed layer positions to capture; residuals are fused in
            this order along the feature axis.

    Returns:
        ``[batch, tokens, len(layer_ids) * hidden]`` — the fused residual stream.
    """
    if not layer_ids:
        raise ValueError("capture_layer_hidden_states requires at least one layer_id")
    layers = backbone.layers
    if any(i < 0 or i >= len(layers) for i in layer_ids):
        raise IndexError(
            f"layer_ids {layer_ids} out of range for backbone with {len(layers)} layers"
        )

    tapset = set(layer_ids)
    h = backbone.embed_tokens(input_ids)
    # Match the body's own mask creation; patched attention reads paged KV from
    # the active step context regardless of this arg.
    from mlx_lm.models.base import create_attention_mask

    mask = create_attention_mask(h, cache[0] if cache else None)
    captured: list[mx.array] = []
    for i, (layer, c) in enumerate(zip(layers, cache, strict=True)):
        h = layer(h, mask, c)
        if i in tapset:
            captured.append(h)
    return mx.concatenate(captured, axis=-1)
