# SPDX-License-Identifier: Apache-2.0
"""FP32 matmul precision contract for this suite's numerical oracles.

MLX 0.32 dispatches FP32 matmuls with more than one row to the M5-class tensor
units at TF32 precision unless ``MLX_ENABLE_TF32=0``. Measured on an M5 Max
(applegpu_g17s): eight-row FP32 GEMM had 8e-4 relative error against float64,
single-row 4e-7, and the switch restored 9e-7. Every FP32 oracle in this suite
(tiny DSpark drafters, Whisper features, dense/quantized matmul references)
assumes true FP32, so ``tests/conftest.py`` pins the switch off before MLX is
imported. These tests fail if that pin stops holding on the executing device.
"""

from __future__ import annotations

import os
import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest

_PROBE = """
import mlx.core as mx, numpy as np
mx.random.seed(0)
w = mx.random.normal((1024, 1024), dtype=mx.float32) / 32
x = mx.random.normal((8, 1024), dtype=mx.float32)
mx.eval(w, x)
ref = np.asarray(x, dtype=np.float64) @ np.asarray(w, dtype=np.float64)
got = np.asarray(x @ w, dtype=np.float64)
print(float(np.max(np.abs(got - ref)) / np.max(np.abs(ref))))
"""


def _relative_error(rows: int) -> float:
    mx.random.seed(0)
    w = mx.random.normal((1024, 1024), dtype=mx.float32) / 32
    x = mx.random.normal((rows, 1024), dtype=mx.float32)
    mx.eval(w, x)
    reference = np.asarray(x, dtype=np.float64) @ np.asarray(w, dtype=np.float64)
    actual = np.asarray(x @ w, dtype=np.float64)
    return float(np.max(np.abs(actual - reference)) / np.max(np.abs(reference)))


def test_fp32_matmul_is_full_precision_for_every_row_count() -> None:
    # The pin must be in place before MLX chose its GEMM path.
    assert os.environ.get("MLX_ENABLE_TF32") == "0"
    for rows in (1, 8, 64):
        assert _relative_error(rows) < 1e-5, rows


def test_tf32_switch_controls_multi_row_precision_on_nax_hardware() -> None:
    architecture = str(mx.device_info().get("architecture", ""))
    if not architecture.startswith("applegpu_g17"):
        pytest.skip(f"{architecture or 'unknown GPU'} has no TF32 GEMM path")
    errors = {}
    for value in ("1", "0"):
        completed = subprocess.run(
            [sys.executable, "-c", _PROBE],
            env={**os.environ, "MLX_ENABLE_TF32": value},
            capture_output=True,
            text=True,
            check=True,
        )
        errors[value] = float(completed.stdout.strip())
    # The default path on this hardware is TF32-class; the pin restores FP32.
    assert errors["1"] > 1e-4, errors
    assert errors["0"] < 1e-5, errors
