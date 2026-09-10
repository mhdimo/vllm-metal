# SPDX-License-Identifier: Apache-2.0
"""Small-M quantized matmul dispatch: numerics, routing, install and calibration."""

from __future__ import annotations

import json

import mlx.core as mx
import mlx.nn as nn
import pytest

from vllm_metal.quant import small_m

K = 1024
N = 40
SHAPE_KEY = f"{K}x{N}b4"


def _require_metal() -> None:
    try:
        available = mx.metal.is_available()
    except AttributeError:
        available = False
    if not available:
        pytest.skip("MLX Metal is not available")


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.delenv("VLLM_METAL_SMALL_M_QMM", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    small_m._verdicts.clear()
    yield
    small_m._verdicts.clear()


class _Host(nn.Module):
    def __init__(self, layers: int = 2, out_features: int = N) -> None:
        super().__init__()
        self.layers = [nn.Linear(K, out_features, bias=False) for _ in range(layers)]
        self.embed = nn.Embedding(out_features, K)


def _quantized_host(layers: int = 2, out_features: int = N) -> _Host:
    mx.random.seed(3)
    host = _Host(layers, out_features)
    nn.quantize(host, group_size=64, bits=4)
    mx.eval(host.parameters())
    return host


def _stock(module: nn.Module, x: mx.array) -> mx.array:
    return mx.quantized_matmul(
        x,
        module["weight"],
        scales=module["scales"],
        biases=module["biases"],
        transpose=True,
        group_size=64,
        bits=4,
    )


@pytest.mark.parametrize("rows", [1, 5, 6, 7, 8, 9, 13, 16, 17, 24, 31, 32])
def test_kernel_matches_stock_at_every_row_count(rows: int) -> None:
    _require_metal()
    host = _quantized_host(1)
    layer = host.layers[0]
    x = (mx.random.normal((rows, K)) * 0.1).astype(mx.bfloat16)
    tiles = small_m.tiles_for_rows(rows)
    got = small_m.small_m_matmul(
        small_m.pad_rows(x, 8 * tiles),
        layer["weight"],
        layer["scales"],
        layer["biases"],
        rows=rows,
        tiles=tiles,
    )
    ref = _stock(layer, x)
    assert got.shape == (rows, N)
    assert got.dtype == mx.bfloat16
    diff = mx.max(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))).item()
    scale = mx.max(mx.abs(ref.astype(mx.float32))).item()
    assert diff <= 0.02 * max(scale, 1.0)
    assert small_m.numerics_ok(layer, rows)


def test_eligibility_rules() -> None:
    host = _quantized_host(1)
    assert small_m.eligible(host.layers[0])
    assert small_m.eligible(host.embed)
    assert small_m.shape_key(host.layers[0]) == SHAPE_KEY
    assert not small_m.eligible(nn.Linear(K, N))
    narrow = nn.Linear(256, N, bias=False)
    nn.quantize(narrow, group_size=64, bits=4)
    assert not small_m.eligible(narrow)  # K % 512 != 0
    eight = nn.Linear(K, N, bias=False)
    nn.quantize(eight, group_size=64, bits=8)
    assert not small_m.eligible(eight)
    g32 = nn.Linear(K, N, bias=False)
    nn.quantize(g32, group_size=32, bits=4)
    assert not small_m.eligible(g32)


def test_route_by_rows_and_verdict() -> None:
    host = _quantized_host(1)
    layer = host.layers[0]
    layer._small_m_key = SHAPE_KEY
    small_m._verdicts[SHAPE_KEY] = frozenset({1, 2})
    x = mx.zeros((1, 8, K), dtype=mx.bfloat16)
    assert small_m.route(layer, x) == 1
    assert small_m.route(layer, mx.zeros((16, K), dtype=mx.bfloat16)) == 2
    assert small_m.route(layer, mx.zeros((5, K), dtype=mx.bfloat16)) is None
    assert (
        small_m.route(layer, mx.zeros((17, K), dtype=mx.bfloat16)) is None
    )  # 3 tiles off
    assert small_m.route(layer, mx.zeros((33, K), dtype=mx.bfloat16)) is None
    assert small_m.route(layer, mx.zeros((8, K), dtype=mx.float16)) is None
    small_m._verdicts.clear()
    assert small_m.route(layer, x) is None


def test_install_on_swaps_classes_and_dispatches(monkeypatch) -> None:
    _require_metal()
    host = _quantized_host(2)
    swapped = small_m.install(host, mode="on")
    assert swapped == 3
    assert all(isinstance(m, small_m.SmallMQuantizedLinear) for m in host.layers)
    assert isinstance(host.embed, small_m.SmallMQuantizedEmbedding)
    assert all(isinstance(m, nn.QuantizedLinear) for m in host.layers)
    calls: list[int] = []
    real = small_m.small_m_matmul

    def spy(*args, **kwargs):
        calls.append(kwargs["rows"])
        return real(*args, **kwargs)

    monkeypatch.setattr(small_m, "small_m_matmul", spy)
    x7 = (mx.random.normal((1, 7, K)) * 0.1).astype(mx.bfloat16)
    y = host.layers[0](x7)
    assert y.shape == (1, 7, N) and calls == [7]
    ref = _stock(host.layers[0], x7)
    assert mx.max(
        mx.abs(y.astype(mx.float32) - ref.astype(mx.float32))
    ).item() <= 0.02 * max(mx.max(mx.abs(ref.astype(mx.float32))).item(), 1.0)
    z = host.embed.as_linear(x7)
    assert z.shape == (1, 7, N) and calls == [7, 7]
    host.layers[0]((mx.random.normal((1, 2, K)) * 0.1).astype(mx.bfloat16))
    assert calls == [7, 7]  # two rows stay on the stock path
    ids = host.embed(mx.array([1, 2]))
    assert ids.shape == (2, K)  # the embedding lookup is untouched
    assert small_m.enabled_rows(host.layers)[SHAPE_KEY] == list(range(6, 33))
    assert small_m.uninstall(host) == 3
    assert type(host.layers[0]) is nn.QuantizedLinear
    assert type(host.embed) is nn.QuantizedEmbedding


def test_install_off_and_bad_mode(monkeypatch) -> None:
    host = _quantized_host(1)
    monkeypatch.setenv("VLLM_METAL_SMALL_M_QMM", "off")
    assert small_m.install(host) == 0
    assert type(host.layers[0]) is nn.QuantizedLinear
    with pytest.raises(ValueError, match="VLLM_METAL_SMALL_M_QMM"):
        small_m.install(host, mode="fast")


def test_install_auto_uses_measured_verdicts_and_cache(monkeypatch, tmp_path) -> None:
    _require_metal()
    host = _quantized_host(2)
    seen: list[str] = []

    def fake_calibrate(groups, *, mode):
        seen.append(mode)
        return {key: [1] for key in groups}

    monkeypatch.setattr(small_m, "calibrate_shapes", fake_calibrate)
    assert small_m.install(host, mode="auto") == 3
    assert seen == ["auto"]
    path = small_m.cache_path()
    assert path.is_file() and json.loads(path.read_text()) == {SHAPE_KEY: [1]}
    assert small_m.route(host.layers[0], mx.zeros((8, K), dtype=mx.bfloat16)) == 1
    assert small_m.route(host.layers[0], mx.zeros((16, K), dtype=mx.bfloat16)) is None
    small_m.uninstall(host)
    small_m._verdicts.clear()
    # a second install reads the cache and does not measure again
    other = _quantized_host(1)
    assert small_m.install(other, mode="auto") == 2
    assert seen == ["auto"]
    # a shape with no winning tile count is left on the stock class
    monkeypatch.setattr(
        small_m, "calibrate_shapes", lambda groups, *, mode: {k: [] for k in groups}
    )
    small_m._verdicts.clear()
    wide = _quantized_host(1, out_features=48)
    path.unlink()
    assert small_m.install(wide, mode="auto") == 0
    assert type(wide.layers[0]) is nn.QuantizedLinear


def test_calibrate_shapes_rejects_bad_numerics(monkeypatch) -> None:
    host = _quantized_host(1)
    monkeypatch.setattr(small_m, "numerics_ok", lambda module, rows: False)
    verdicts = small_m.calibrate_shapes({SHAPE_KEY: [host.layers[0]]}, mode="on")
    assert verdicts == {SHAPE_KEY: []}


def test_calibrate_shapes_enables_winning_tiles(monkeypatch) -> None:
    host = _quantized_host(1)
    monkeypatch.setattr(small_m, "numerics_ok", lambda module, rows: True)
    monkeypatch.setattr(
        small_m, "race", lambda modules, tiles: {1: 1.5, 2: 1.2, 3: 1.05, 4: 0.9}[tiles]
    )
    verdicts = small_m.calibrate_shapes({SHAPE_KEY: [host.layers[0]]}, mode="auto")
    assert verdicts == {SHAPE_KEY: [1, 2]}
    assert small_m.calibrate_shapes({SHAPE_KEY: [host.layers[0]]}, mode="on") == {
        SHAPE_KEY: [1, 2, 3, 4]
    }


def test_race_returns_a_ratio() -> None:
    _require_metal()
    host = _quantized_host(2)
    ratio = small_m.race(host.layers, 1)
    assert ratio > 0
