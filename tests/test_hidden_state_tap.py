# SPDX-License-Identifier: Apache-2.0
"""Tests for the reusable hidden-state layer tap (DSpark prep)."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

pytest.importorskip("vllm", reason="vllm not installed")

from vllm_metal.v1.hidden_state_tap import capture_layer_hidden_states


class _Embed:
    """Stand-in for embed_tokens: [B, T] ids -> [B, T, 1] (hidden dim of 1)."""

    def __call__(self, ids: mx.array) -> mx.array:
        return ids[..., None]


class _AddLayer:
    """Fake transformer layer: adds its index to the residual stream."""

    def __init__(self, i: int) -> None:
        self.i = i

    def __call__(self, h: mx.array, mask, cache) -> mx.array:  # noqa: ANN001
        return h + self.i


def _toy_backbone(num_layers: int) -> SimpleNamespace:
    return SimpleNamespace(
        embed_tokens=_Embed(),
        layers=[_AddLayer(i) for i in range(num_layers)],
        norm=_Embed(),  # unused by the helper
    )


def test_capture_requires_at_least_one_layer() -> None:
    backbone = _toy_backbone(3)
    with pytest.raises(ValueError):
        capture_layer_hidden_states(
            backbone, mx.array([[1.0, 2.0, 3.0]]), cache=[None] * 3, layer_ids=[]
        )


def test_capture_fuses_requested_layers_in_order() -> None:
    # embed: ids -> ids[...,None]; layer i adds i to the running residual, so
    # after layer i the residual is ids + sum(0..i). Capturing [0, 2] fuses the
    # residual after layer 0 (ids+0) and after layer 2 (ids+0+1+2 = ids+3).
    backbone = _toy_backbone(3)
    ids = mx.array([[1.0, 2.0, 3.0]])  # [1, 3]
    out = capture_layer_hidden_states(backbone, ids, cache=[None] * 3, layer_ids=[0, 2])
    mx.eval(out)
    assert out.shape == (1, 3, 2)  # [batch, tokens, len(layer_ids) * hidden(=1)]
    expected = mx.concatenate([ids[..., None], (ids + 3)[..., None]], axis=-1)
    assert mx.array_equal(out, expected)


def test_capture_single_last_layer_matches_full_residual() -> None:
    # Capturing only the last layer reproduces the body's own final residual
    # (the pre-norm stream) — the fidelity invariant the drafter relies on.
    backbone = _toy_backbone(4)
    ids = mx.array([[1.0, 2.0, 3.0]])
    out = capture_layer_hidden_states(backbone, ids, cache=[None] * 4, layer_ids=[3])
    final = (ids + 0 + 1 + 2 + 3)[..., None]  # residual after all 4 layers
    mx.eval(out)
    assert out.shape == (1, 3, 1)
    assert mx.array_equal(out, final)


def test_capture_rejects_out_of_range_layer() -> None:
    backbone = _toy_backbone(3)
    with pytest.raises(IndexError):
        capture_layer_hidden_states(
            backbone, mx.array([[1.0, 2.0]]), cache=[None] * 3, layer_ids=[5]
        )
