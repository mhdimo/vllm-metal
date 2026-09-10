# SPDX-License-Identifier: Apache-2.0
"""Small-M quantized matmul dispatch for affine 4-bit linear layers.

``mx.quantized_matmul`` streams the weights once per call on its GEMV path, so a
few rows cost little more than one (five rows of the pinned Qwen3-4B target cost
1.23x a single row on an M5 Max), but from six rows it switches to a GEMM tiling
that is badly utilized below about sixteen rows: eight rows cost 2.2x a single
row, nearly as much as sixteen. Speculative verification lives exactly there
(``num_speculative_tokens + 1`` rows per request), and so does the DSpark
drafter's block backbone (seven rows per request), so at low concurrency the
verification step pays for rows the hardware could have read the weights once
for.

The kernel here dequantizes each weight group once per threadgroup and applies
it to every row through 8x8 ``simdgroup_matrix`` tiles (``ceil(M / 8)`` tiles,
up to four for M <= 32): a threadgroup owns 32 output columns, one per lane,
each lane loads one 64-wide quantization group of its column per chunk (two
16-byte loads, one scale and one bias) into a per-simdgroup stage, and K is
split across the four simdgroups so the serial loop per simdgroup is short.
The structure follows avlp12's ``qmm_mma4`` (MIT, via mlx-dspark's
``small_m_qmm.py``; see NOTICE), which stages eight columns per threadgroup
over eight simdgroups; the 32-column layout (a quarter of the activation
re-reads, 16-byte weight loads), the multi-tile form and the dispatch are this
repository's. On the M5 Max the stock kernel's GEMV path still streams weights
faster than any MMA layout measured (this one reaches about 60-70% of its rate
at one row), so the gain at eight rows is bounded by that ratio times the stock
GEMM penalty; the race below decides per shape. Accumulation is fp32 in
a different order than the stock kernel, so outputs differ from it at the bf16
ULP level, the same class as the stock kernel's own difference between its GEMV
and GEMM paths; the parity contract classifies those as ties.

Dispatch is by row count and by measurement, never by assumption: at install
every distinct eligible weight shape is checked against the stock kernel for
numerics and raced on a dependent chain over rotated same-shape layers at one to
four tiles, and the kernel is enabled only for the tile counts where it wins by
:data:`MIN_GAIN`. Verdicts are cached per (GPU architecture, MLX version, kernel
version). ``VLLM_METAL_SMALL_M_QMM`` selects ``auto`` (measure), ``off`` or ``on``
(skip the race, keep the numerics check).
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from vllm.logger import init_logger

from vllm_metal import envs as metal_envs

logger = init_logger(__name__)

KERNEL_VERSION = 2
BITS = 4
GROUP_SIZE = 64
TILE_ROWS = 8
MAX_TILES = 4
MAX_ROWS = TILE_ROWS * MAX_TILES
# The stock GEMV path wins below six rows on every machine measured; the race
# re-verifies each tile count, this bound only keeps the one-row decode path
# free of dispatch work.
MIN_ROWS = 6
K_ALIGN = 256  # four simdgroups x one 64-wide quantization group per chunk
THREADGROUP = 128
COLUMNS = 32
MIN_GAIN = 1.10
NUMERICS_TOLERANCE = 0.02
MODES = ("auto", "off", "on")
_CHAIN = 12
_EVALS = 4

_SRC = r"""
    const int K = KD, N = ND, M = MD;
    const int SG = 4;
    const int KPS = KD / SG;
    uint tid  = thread_position_in_threadgroup.x;
    uint tgid = threadgroup_position_in_grid.x;
    uint sg   = tid >> 5;
    uint lane = tid & 31;
    int n0 = (int)tgid * 32;
    // per simdgroup: a 64 k x 32 n bfloat16 stage (4 KB); the split-K reduction reuses it
    threadgroup bfloat16_t bs[4 * 64 * 32];
    threadgroup bfloat16_t* bt = bs + sg * 2048;
    simdgroup_matrix<float, 8, 8> C[TILES][4];
    for (int t = 0; t < TILES; ++t)
        for (int c = 0; c < 4; ++c) C[t][c] = simdgroup_matrix<float, 8, 8>(0);
    int n = n0 + (int)lane;                 // one output column per lane
    int kbeg = (int)sg * KPS;
    for (int kk = 0; kk < KPS; kk += 64) {  // one chunk = one quantization group
        int ka = kbeg + kk;
        if (n < N) {
            int g = ka >> 6;
            float s  = (float)sc[(size_t)n * (K / 64) + g];
            float bb = (float)bi[(size_t)n * (K / 64) + g];
            const device uint4* wr = (const device uint4*)(w + (size_t)n * (K / 8) + (ka >> 3));
            uint4 p0 = wr[0], p1 = wr[1];
            uint words[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
            for (int q = 0; q < 8; ++q) {
                uint p = words[q];
                for (int t = 0; t < 8; ++t)
                    bt[(q * 8 + t) * 32 + lane] =
                        (bfloat16_t)((float)((p >> (4 * t)) & 15u) * s + bb);
            }
        } else {
            for (int kq = 0; kq < 64; ++kq) bt[kq * 32 + lane] = (bfloat16_t)0;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_matrix<bfloat16_t, 8, 8> A, B;
        for (int kt = 0; kt < 8; ++kt) {
            for (int t = 0; t < TILES; ++t) {
                simdgroup_load(A, x + (size_t)t * 8 * K + ka + kt * 8, K);
                for (int c = 0; c < 4; ++c) {
                    simdgroup_load(B, bt + kt * 8 * 32 + c * 8, 32);
                    simdgroup_multiply_accumulate(C[t][c], A, B, C[t][c]);
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* red = (threadgroup float*)bs;   // SG x TILES x 4 x 64 floats <= 16 KB
    for (int t = 0; t < TILES; ++t)
        for (int c = 0; c < 4; ++c)
            simdgroup_store(C[t][c], red + ((sg * TILES + t) * 4 + c) * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = (int)tid; i < TILES * 4 * 64; i += 128) {
        int t = i / 256, rem = i % 256;
        int c = rem >> 6, r = rem & 63;
        int m = t * 8 + (r >> 3), jj = c * 8 + (r & 7);
        int nn = n0 + jj;
        if (m < M && nn < N) {
            float v = 0.0f;
            for (int q = 0; q < SG; ++q) v += red[((q * TILES + t) * 4 + c) * 64 + r];
            out[(size_t)m * N + nn] = (bfloat16_t)v;
        }
    }
"""

_kernel: Any = None


def _get_kernel() -> Any:
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="vllm_metal_small_m_qmm4",
            input_names=["x", "w", "sc", "bi"],
            output_names=["out"],
            source=_SRC,
        )
    return _kernel


def tiles_for_rows(rows: int) -> int:
    return (rows + TILE_ROWS - 1) // TILE_ROWS


def pad_rows(x: mx.array, rows: int) -> mx.array:
    """Zero-pad a ``[m, K]`` activation to ``rows`` rows (a whole number of tiles)."""
    if x.shape[0] == rows:
        return x
    pad = mx.zeros((rows - x.shape[0], x.shape[1]), dtype=x.dtype)
    return mx.concatenate([x, pad], axis=0)


def small_m_matmul(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    rows: int,
    tiles: int,
) -> mx.array:
    """``x @ dequantize(weight).T`` for ``rows`` rows of ``x`` padded to ``8 * tiles``.

    ``x`` is ``[8 * tiles, K]`` bfloat16, the weight the affine 4-bit ``[N, K/8]``
    packing with ``[N, K/64]`` scales and biases; returns ``[rows, N]`` bfloat16.
    """
    out_features = weight.shape[0]
    in_features = x.shape[1]
    (out,) = _get_kernel()(
        inputs=[x, weight, scales, biases],
        template=[
            ("KD", in_features),
            ("ND", out_features),
            ("MD", rows),
            ("TILES", tiles),
        ],
        output_shapes=[(rows, out_features)],
        output_dtypes=[mx.bfloat16],
        grid=(((out_features + COLUMNS - 1) // COLUMNS) * THREADGROUP, 1, 1),
        threadgroup=(THREADGROUP, 1, 1),
    )
    return out


def in_features(module: nn.Module) -> int:
    """Input width of an affine-quantized layer (packed 32 / bits values per word)."""
    return int(module["weight"].shape[1]) * 32 // int(module.bits)


def eligible(module: nn.Module) -> bool:
    """Whether a layer's format and shape can run on the kernel at all."""
    if not isinstance(module, nn.QuantizedLinear | nn.QuantizedEmbedding):
        return False
    if getattr(module, "mode", "affine") != "affine" or "biases" not in module:
        return False
    if int(module.bits) != BITS or int(module.group_size) != GROUP_SIZE:
        return False
    return in_features(module) % K_ALIGN == 0


def shape_key(module: nn.Module) -> str:
    return f"{in_features(module)}x{int(module['weight'].shape[0])}b{int(module.bits)}"


def _stock(module: nn.Module, x: mx.array) -> mx.array:
    return mx.quantized_matmul(
        x,
        module["weight"],
        scales=module["scales"],
        biases=module["biases"],
        transpose=True,
        group_size=module.group_size,
        bits=module.bits,
        mode=module.mode,
    )


# shape key -> enabled tile counts; filled by install(), read by every dispatch.
_verdicts: dict[str, frozenset[int]] = {}


def route(module: nn.Module, x: mx.array) -> int | None:
    """Tile count for a call that goes to the kernel, ``None`` for the stock path."""
    if x.dtype != mx.bfloat16:
        return None
    rows = 1
    for dim in x.shape[:-1]:
        rows *= dim
    if rows < MIN_ROWS or rows > MAX_ROWS:
        return None
    tiles = tiles_for_rows(rows)
    enabled = _verdicts.get(getattr(module, "_small_m_key", ""))
    if not enabled or tiles not in enabled:
        return None
    return tiles


def _apply(module: nn.Module, x: mx.array, tiles: int) -> mx.array:
    width = x.shape[-1]
    flat = x.reshape(-1, width)
    rows = flat.shape[0]
    out = small_m_matmul(
        pad_rows(flat, TILE_ROWS * tiles),
        module["weight"],
        module["scales"],
        module["biases"],
        rows=rows,
        tiles=tiles,
    )
    return out.reshape(*x.shape[:-1], out.shape[-1])


class SmallMQuantizedLinear(nn.QuantizedLinear):
    """``nn.QuantizedLinear`` whose 6..32-row calls run on the small-M kernel."""

    def __call__(self, x: mx.array) -> mx.array:
        tiles = route(self, x)
        if tiles is None:
            return super().__call__(x)
        y = _apply(self, x, tiles)
        if "bias" in self:
            y = y + self["bias"]
        return y


class SmallMQuantizedEmbedding(nn.QuantizedEmbedding):
    """``nn.QuantizedEmbedding`` whose tied-projection calls run on the kernel."""

    def as_linear(self, x: mx.array) -> mx.array:
        tiles = route(self, x)
        if tiles is None:
            return super().as_linear(x)
        return _apply(self, x, tiles)


_SWAPS: dict[type, type] = {
    nn.QuantizedLinear: SmallMQuantizedLinear,
    nn.QuantizedEmbedding: SmallMQuantizedEmbedding,
}
_RESTORE = {new: old for old, new in _SWAPS.items()}


def is_installed(module: nn.Module) -> bool:
    return type(module) in _RESTORE


# ---------------------------------------------------------------- calibration


def _time_chain(step: Any, x0: mx.array) -> float:
    times = []
    for _ in range(_EVALS):
        x = x0
        mx.synchronize()
        started = time.perf_counter()
        y = x0
        for t in range(_CHAIN):
            y = step(x, t)
            x = x0 + mx.mean(y).astype(x0.dtype) * 1e-20
        mx.eval(y, x)
        mx.synchronize()
        times.append((time.perf_counter() - started) / _CHAIN)
    return statistics.median(times[1:])


def numerics_ok(module: nn.Module, rows: int) -> bool:
    """Kernel against the stock kernel at ``rows`` rows: bf16-ULP class differences only."""
    width = in_features(module)
    x = (mx.random.normal((rows, width)) * 0.1).astype(mx.bfloat16)
    ref = _stock(module, x).astype(mx.float32)
    tiles = tiles_for_rows(rows)
    got = small_m_matmul(
        pad_rows(x, TILE_ROWS * tiles),
        module["weight"],
        module["scales"],
        module["biases"],
        rows=rows,
        tiles=tiles,
    ).astype(mx.float32)
    diff = mx.max(mx.abs(ref - got))
    scale = mx.max(mx.abs(ref))
    mx.eval(diff, scale)
    largest = max(float(scale.item()), 1.0)
    return float(diff.item()) <= NUMERICS_TOLERANCE * largest


def race(modules: list[nn.Module], tiles: int) -> float:
    """Stock time over kernel time at ``8 * tiles`` rows, weights rotated across ``modules``."""
    width = in_features(modules[0])
    rows = TILE_ROWS * tiles
    x = (mx.random.normal((rows, width)) * 0.1).astype(mx.bfloat16)
    mx.eval(x)

    def stock_step(xx: mx.array, t: int) -> mx.array:
        return _stock(modules[t % len(modules)], xx)

    def kernel_step(xx: mx.array, t: int) -> mx.array:
        module = modules[t % len(modules)]
        return small_m_matmul(
            xx,
            module["weight"],
            module["scales"],
            module["biases"],
            rows=rows,
            tiles=tiles,
        )

    stock = min(_time_chain(stock_step, x), _time_chain(stock_step, x))
    kernel = min(_time_chain(kernel_step, x), _time_chain(kernel_step, x))
    return stock / kernel if kernel > 0 else 0.0


def calibrate_shapes(
    groups: dict[str, list[nn.Module]], *, mode: str
) -> dict[str, list[int]]:
    """Enabled tile counts per shape: numerics gate, then the race unless ``mode == "on"``."""
    verdicts: dict[str, list[int]] = {}
    for key, modules in groups.items():
        module = modules[0]
        checks = [MIN_ROWS, MIN_ROWS + 1, TILE_ROWS] + [
            TILE_ROWS * t for t in range(2, MAX_TILES + 1)
        ]
        if not all(numerics_ok(module, rows) for rows in checks):
            logger.warning("small-M qmm: shape %s rejected by the numerics check", key)
            verdicts[key] = []
            continue
        if mode == "on":
            verdicts[key] = list(range(1, MAX_TILES + 1))
            continue
        enabled = []
        gains = []
        for tiles in range(1, MAX_TILES + 1):
            gain = race(modules, tiles)
            gains.append(f"{TILE_ROWS * tiles}:{gain:.2f}x")
            if gain >= MIN_GAIN:
                enabled.append(tiles)
        logger.info(
            "small-M qmm: shape %s rows->gain %s -> tiles %s",
            key,
            " ".join(gains),
            enabled or "none",
        )
        verdicts[key] = enabled
    return verdicts


def cache_path() -> Path:
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    arch = str(mx.device_info().get("architecture", "unknown"))
    digest = hashlib.sha256(
        f"{arch}|{mx.__version__}|{KERNEL_VERSION}".encode()
    ).hexdigest()[:16]
    return root / "vllm-metal" / "small-m" / f"{arch}-{digest}.json"


def _load_cache(path: Path) -> dict[str, list[int]]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return (
        {str(k): [int(t) for t in v] for k, v in data.items()}
        if isinstance(data, dict)
        else {}
    )


def _save_cache(path: Path, verdicts: dict[str, list[int]]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(verdicts, indent=1, sort_keys=True) + "\n")
    except OSError as exc:  # pragma: no cover - cache is best effort
        logger.warning("small-M qmm: could not write %s: %s", path, exc)


def collect(model: nn.Module) -> dict[str, list[nn.Module]]:
    groups: dict[str, list[nn.Module]] = {}
    for _, module in model.named_modules():
        if eligible(module):
            groups.setdefault(shape_key(module), []).append(module)
    return groups


def install(model: nn.Module, *, mode: str | None = None, tag: str = "model") -> int:
    """Route the model's eligible layers through the kernel where it measured faster.

    Returns the number of layers swapped. ``mode`` defaults to
    ``VLLM_METAL_SMALL_M_QMM``; ``off`` installs nothing.
    """
    mode = mode or metal_envs.VLLM_METAL_SMALL_M_QMM
    if mode not in MODES:
        raise ValueError(f"VLLM_METAL_SMALL_M_QMM={mode!r} is not one of {MODES}")
    if mode == "off" or not isinstance(model, nn.Module):
        # Pooling shims and test doubles are not module trees; nothing to route.
        return 0
    groups = collect(model)
    if not groups:
        return 0
    path = cache_path()
    cached = {} if mode == "on" else _load_cache(path)
    missing = {k: v for k, v in groups.items() if k not in cached}
    verdicts = dict(cached)
    if missing:
        verdicts.update(calibrate_shapes(missing, mode=mode))
        if mode == "auto":
            _save_cache(path, verdicts)
    swapped = 0
    for key, modules in groups.items():
        enabled = frozenset(verdicts.get(key, []))
        if not enabled:
            continue
        _verdicts[key] = enabled
        for module in modules:
            module._small_m_key = key
            if type(module) in _SWAPS:
                module.__class__ = _SWAPS[type(module)]
            swapped += 1
    logger.info(
        "small-M qmm (%s): %d layers on the kernel, shapes %s",
        tag,
        swapped,
        {
            k: sorted(TILE_ROWS * t for t in v)
            for k, v in _verdicts.items()
            if k in groups
        },
    )
    return swapped


def uninstall(model: nn.Module) -> int:
    """Restore the stock classes (tests and teardown)."""
    restored = 0
    for _, module in model.named_modules():
        if is_installed(module):
            module.__class__ = _RESTORE[type(module)]
            restored += 1
    return restored


def enabled_rows(modules: Iterable[nn.Module]) -> dict[str, list[int]]:
    """Row counts routed to the kernel per installed shape (for logs and tests)."""
    out: dict[str, list[int]] = {}
    for module in modules:
        key = getattr(module, "_small_m_key", None)
        if key and key in _verdicts:
            out[key] = [
                rows
                for rows in range(MIN_ROWS, MAX_ROWS + 1)
                if tiles_for_rows(rows) in _verdicts[key]
            ]
    return out
