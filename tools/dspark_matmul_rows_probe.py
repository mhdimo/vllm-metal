# SPDX-License-Identifier: Apache-2.0
"""Where does the multi-row verification cost come from?

Times the pinned target's own quantized linear layers at 1 to 32 query rows
in process through mlx-lm: one decoder layer's MLP, its attention
projections, and the language-model head, reporting milliseconds per call
and the ratio to a single row. A verification step multiplies these row
counts by the active requests, so the ratios show whether the affine-4
matmul path, not attention, carries the multi-row cost the M4d and M6 cost
profiles measured. Output: JSON for the M7 profile record.
"""

from __future__ import annotations

import json
import sys
import time

import mlx.core as mx


def timeit(fn, reps: int = 30, warm: int = 5) -> float:
    for _ in range(warm):
        mx.eval(fn())
    mx.synchronize()
    started = time.perf_counter()
    for _ in range(reps):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - started) / reps * 1000


def probe_rows(model, rows: int) -> dict[str, float]:
    layer = model.model.layers[0]
    hidden = model.args.hidden_size
    x = mx.random.normal((1, rows, hidden)).astype(mx.bfloat16)
    attn = layer.self_attn
    if getattr(model.args, "tie_word_embeddings", False):

        def head():
            return model.model.embed_tokens.as_linear(model.model.norm(x))

    else:

        def head():
            return model.lm_head(model.model.norm(x))

    # o_proj reads the concatenated heads, which can be wider than the hidden size.
    heads = mx.random.normal((1, rows, attn.q_proj(x).shape[-1])).astype(mx.bfloat16)
    return {
        "mlp_ms": timeit(lambda: layer.mlp(x)),
        "qkv_ms": timeit(lambda: (attn.q_proj(x), attn.k_proj(x), attn.v_proj(x))),
        "o_proj_ms": timeit(lambda: attn.o_proj(heads)),
        "lm_head_ms": timeit(head),
    }


def main() -> None:
    target, output = sys.argv[1], sys.argv[2]
    from mlx_lm import load

    model, _tokenizer = load(target)
    rows_list = [1, 2, 4, 8, 12, 16, 24, 32]
    results: dict = {"target": target, "rows": {}}
    # Two passes over the row counts: the first warms every kernel variant
    # (the one-row path compiles its own), the second is the record.
    for rows in rows_list:
        probe_rows(model, rows)
    for rows in rows_list:
        item = probe_rows(model, rows)
        results["rows"][rows] = item
        print(
            f"M={rows}: mlp {item['mlp_ms']:.3f} ms, qkv {item['qkv_ms']:.3f}, "
            f"o {item['o_proj_ms']:.3f}, lm_head {item['lm_head_ms']:.3f}",
            flush=True,
        )
    base = results["rows"][1]
    for rows in rows_list:
        item = results["rows"][rows]
        item["mlp_ratio"] = item["mlp_ms"] / base["mlp_ms"]
        item["lm_head_ratio"] = item["lm_head_ms"] / base["lm_head_ms"]
    with open(output, "w") as handle:
        json.dump(results, handle, indent=2)
    print("wrote", output)


if __name__ == "__main__":
    main()
