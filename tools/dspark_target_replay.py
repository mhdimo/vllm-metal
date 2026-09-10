# SPDX-License-Identifier: Apache-2.0
"""Replay first greedy divergences through the native target without a drafter.

The input is a failure artifact from dspark_memory_check, or this tool's result.
Identical teacher-forced prefixes are processed with different chunk sizes.
This diagnoses target numerical sensitivity; it does not waive a failed greedy
comparison or prove that every speculative divergence has the same cause.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.target.is_dir():
        parser.error("--target must be an already downloaded snapshot")

    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model

    from tools.dspark_precision_check import snapshot_revision
    from vllm_metal.v1.model_adapter import DefaultModelAdapter

    source = json.loads(args.failure.read_text())
    cases = []
    if "expected_tokens" in source:
        for index, (prompt, expected, actual) in enumerate(
            zip(
                source["prompt_token_ids"],
                source["expected_tokens"],
                source["actual_tokens"],
                strict=True,
            )
        ):
            for position, (baseline, speculative) in enumerate(
                zip(expected, actual, strict=True)
            ):
                if baseline != speculative:
                    cases.append(
                        {
                            "case": index,
                            "prompt_length": len(prompt),
                            "prefix_token_ids": prompt + expected[:position],
                            "output_position": position,
                            "baseline_token": baseline,
                            "speculative_token": speculative,
                        }
                    )
                    break
    else:
        cases = source["cases"]
    assert cases, "no divergent prefixes to diagnose"
    model, _ = load_model(args.target, lazy=False)
    adapter = DefaultModelAdapter()
    for case in cases:
        ids = case["prefix_token_ids"]
        rows = []
        for chunk in (1, 4, 7, 8, 64, len(ids)):
            cache = make_prompt_cache(model)
            start = 0
            while start < len(ids):
                count = case["prompt_length"] if start == 0 else chunk
                tokens = ids[start : start + count]
                result = adapter.target_forward(
                    model,
                    mx.array(tokens)[None],
                    cache=cache,
                    logits_indices=mx.array([len(tokens) - 1]),
                )
                mx.eval(result.logits)
                start += len(tokens)
            logits = result.logits.reshape(-1, result.logits.shape[-1])[-1].astype(
                mx.float32
            )
            top = mx.argsort(logits)[-5:][::-1].tolist()
            rows.append(
                {
                    "chunk_size_after_prompt": chunk,
                    "greedy_token": int(mx.argmax(logits).item()),
                    "top_logits": [[i, logits[i].item()] for i in top],
                    "baseline_token_logit": logits[case["baseline_token"]].item(),
                    "speculative_token_logit": logits[case["speculative_token"]].item(),
                }
            )
        case["native_target_only"] = rows
        case["prefix_sha256"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        print(f"Replayed case {case['case']}", flush=True)
    args.output.write_text(
        json.dumps(
            {
                "target_revision": snapshot_revision(args.target),
                "cases": cases,
                "peak_mlx_bytes": mx.get_peak_memory(),
                "runtime": {
                    package: importlib.metadata.version(package)
                    for package in ("mlx", "mlx-lm")
                },
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
