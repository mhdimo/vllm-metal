# SPDX-License-Identifier: Apache-2.0
"""Check DSpark capture against a local, immutable Qwen3 target checkpoint.

This loads no drafter weights and downloads nothing. It tests the native MLX
body and Metal paged attention at identical execution shapes, including mixed
verification/prefill rows. Run as a module; see docs/design/dspark-progress.md.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_model

from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.context import OffsetCache, clear_context, prepare_grouped
from vllm_metal.attention.impls.sdpa_wrapper import patch_sdpa_attention
from vllm_metal.v1.dspark.config import DSparkConfig
from vllm_metal.v1.model_adapter import DefaultModelAdapter


def check_target(model, layer_ids: list[int]) -> list[dict]:
    """Bit-exact capture parity at the same head/attention shapes."""
    adapter = DefaultModelAdapter()
    assert adapter.supports_selective_logits(model)
    original_layers = model.model.layers
    results = []

    def compare(label, plain, captured):
        mx.eval(plain, captured)
        error = float(
            mx.max(
                mx.abs(plain.astype(mx.float32) - captured.astype(mx.float32))
            ).item()
        )
        assert bool(mx.array_equal(plain, captured).item()), (label, error)
        results.append(
            {"case": label, "shape": list(plain.shape), "max_abs_error": error}
        )

    for selective in (False, True):
        caches = [make_prompt_cache(model), make_prompt_cache(model)]
        for index, length in enumerate((1, 7, 5, 3)):
            ids = (mx.arange(length, dtype=mx.int32) + 20 + index)[None]
            selection = mx.array([length - 1]) if selective else None
            outputs = [
                adapter.target_forward(
                    model,
                    ids,
                    cache=cache,
                    logits_indices=selection,
                    capture_layer_ids=layer_ids if capture else None,
                )
                for capture, cache in zip((False, True), caches, strict=True)
            ]
            compare(
                f"native_{'selected' if selective else 'full'}_chunk{index}",
                outputs[0].logits,
                outputs[1].logits,
            )
            assert outputs[1].hidden_states.shape == (
                length,
                len(layer_ids) * model.args.hidden_size,
            )
            for plain, captured in zip(*caches, strict=True):
                assert plain.offset == captured.offset
                assert mx.array_equal(plain.keys, captured.keys)
                assert mx.array_equal(plain.values, captured.values)

    # Fresh per-mode paged storage; the actual attention wrappers are shared
    # with the target body copy, and must write the same physical KV slots.
    for merged in (False, True):
        for selective in (False, True):
            outputs = []
            saved_kv = []
            for capture in (False, True):
                args = model.args
                kv = MetalPagedKVCache(
                    args.num_hidden_layers,
                    args.num_key_value_heads,
                    args.head_dim,
                    3,
                    32,
                    dtype=mx.float16,
                )
                assert patch_sdpa_attention(model, kv, 32) == args.num_hidden_layers
                offset_caches = [OffsetCache(0) for _ in model.layers]
                prepare_grouped([], [([[0]], 3, 0), ([[1]], 2, 0)], (32,))
                try:
                    seed = model(mx.array([[10, 11, 12, 20, 21]]), cache=offset_caches)
                    mx.eval(seed)
                finally:
                    clear_context()
                prepare_grouped(
                    [([[0]], 3, 3)],
                    [([[1]], 4, 2), ([[2]], 5, 0)],
                    (32,),
                    merge_verify_windows=merged,
                )
                try:
                    output = adapter.target_forward(
                        model,
                        mx.array([[13, 14, 15, 22, 23, 24, 25, 30, 31, 32, 33, 34]]),
                        cache=offset_caches,
                        logits_indices=mx.array([0, 1, 2, 6, 11])
                        if selective
                        else None,
                        capture_layer_ids=layer_ids if capture else None,
                    )
                    mx.eval(output.logits, output.hidden_states)
                finally:
                    clear_context()
                outputs.append(output)
                saved_kv.append([*kv.key_caches, *kv.value_caches])
            compare(
                f"paged_mixed_merged{merged}_selected{selective}",
                outputs[0].logits,
                outputs[1].logits,
            )
            assert outputs[1].hidden_states.shape == (
                12,
                len(layer_ids) * model.args.hidden_size,
            )
            assert all(
                bool(mx.array_equal(a, b).item())
                for a, b in zip(*saved_kv, strict=True)
            )
    assert model.model.layers is original_layers
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", type=Path, required=True, help="Local pinned target snapshot"
    )
    parser.add_argument("--draft-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.target.is_dir() or not args.draft_config.is_file():
        parser.error(
            "target directory and draft configuration must already exist locally"
        )
    config = DSparkConfig.from_json(args.draft_config)
    model, _ = load_model(args.target)
    result = {
        "target_path": str(args.target.resolve()),
        "target_config_sha256": hashlib.sha256(
            (args.target / "config.json").read_bytes()
        ).hexdigest(),
        "draft_config_sha256": hashlib.sha256(
            args.draft_config.read_bytes()
        ).hexdigest(),
        "mlx": importlib.metadata.version("mlx"),
        "mlx_lm": importlib.metadata.version("mlx-lm"),
        "cases": check_target(model, config.target_layer_ids),
        "peak_mlx_bytes": mx.get_peak_memory(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"PASS: {len(result['cases'])} target capture comparisons; {args.output}")


if __name__ == "__main__":
    main()
