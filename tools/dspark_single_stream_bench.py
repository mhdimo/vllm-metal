# SPDX-License-Identifier: Apache-2.0
"""Single-stream decode speed on a fixed prompt list: target-only against DSpark.

The comparison protocol of standalone MLX runners (one request at a time, a
few short prompts, greedy decoding, a fixed output budget, warm) differs from
this repository's paired streamed HTTP protocol (``tools.dspark_perf_bench``)
in every axis, so to put the two side by side this tool runs that protocol
here: an offline engine in process, one prompt at a time through the target's
chat template, greedy, ``--max-tokens`` output, the first pass of every prompt
as warm-up, then ``--trials`` timed passes with the median reported. Widths
``0`` (target-only) and the requested speculative widths run in the same
process order; the adaptive mode is selected through the usual environment
variables. Reported per prompt and width: decode tokens per second (output
tokens over the wall time of the generate call, prefill included, as the
standalone runners report it), the accepted length per drafting round and
the speedup over width 0.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

# The three prompts of mlx-dspark's ``benchmark`` command (chat / code / math).
DEFAULT_PROMPTS = {
    "chat": "Explain how rainbows form.",
    "code": "Write a Python function to check if a string is a palindrome.",
    "math": "A train travels 120 km in 1.5 hours. What is its average speed in m/s? "
    "Show your work.",
}


def worker(config: dict, output: Path) -> None:
    import mlx.core as mx
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    width = config["width"]
    tokenizer = AutoTokenizer.from_pretrained(config["target"])
    llm = LLM(
        model=config["target"],
        max_model_len=config["max_model_len"],
        max_num_seqs=1,
        max_num_batched_tokens=config["max_model_len"],
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        disable_log_stats=True,
        speculative_config=(
            {
                "method": "dspark",
                "model": config["draft"],
                "num_speculative_tokens": width,
            }
            if width
            else None
        ),
    )
    proposer = None
    if width:
        runner = llm.llm_engine.model_executor.driver_worker.model_runner
        proposer = runner._drafter
    params = SamplingParams(temperature=0.0, max_tokens=config["max_tokens"])
    results = {}
    for name, prompt in config["prompts"].items():
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        trials = []
        for trial in range(config["trials"] + 1):
            before = proposer.counters.snapshot() if proposer is not None else None
            mx.synchronize()
            started = time.perf_counter()
            out = llm.generate([text], params, use_tqdm=False)[0]
            mx.synchronize()
            wall = time.perf_counter() - started
            tokens = len(out.outputs[0].token_ids)
            accepted_length = None
            if proposer is not None and before is not None:
                after = proposer.counters.snapshot()
                rounds = after["verified_requests"] - before["verified_requests"]
                accepted = after["accepted_tokens"] - before["accepted_tokens"]
                accepted_length = (1 + accepted / rounds) if rounds else None
            if trial:  # the first pass warms the prompt and is not counted
                trials.append(
                    {
                        "tokens": tokens,
                        "wall_s": wall,
                        "tokens_per_s": tokens / wall,
                        "accepted_length": accepted_length,
                    }
                )
        results[name] = {
            "trials": trials,
            "tokens_per_s_median": statistics.median(t["tokens_per_s"] for t in trials),
            "accepted_length_median": (
                statistics.median(t["accepted_length"] for t in trials)
                if all(t["accepted_length"] is not None for t in trials)
                else None
            ),
            "output_tokens": trials[-1]["tokens"],
        }
        print(
            f"width {width} {name}: {results[name]['tokens_per_s_median']:.1f} tok/s "
            f"(accepted length {results[name]['accepted_length_median']})",
            flush=True,
        )
    output.write_text(json.dumps(results, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--widths", default="0,7")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--memory-fraction", default="0.3")
    parser.add_argument(
        "--prompts",
        type=Path,
        help="JSON object name -> prompt (default: chat/code/math)",
    )
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_config:
        worker(
            json.loads(args.worker_config.read_text()),
            args.worker_config.with_suffix(".result.json"),
        )
        return
    if any(path is None for path in (args.target, args.draft, args.output_dir)):
        parser.error("--target, --draft and --output-dir are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(args.prompts.read_text()) if args.prompts else DEFAULT_PROMPTS
    env = dict(
        os.environ,
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
        VLLM_METAL_USE_PAGED_ATTENTION="1",
        VLLM_METAL_MEMORY_FRACTION=str(args.memory_fraction),
        VLLM_METAL_BUILD_FROM_SOURCE="1",
        HF_HUB_OFFLINE="1",
    )
    summary: dict = {"prompts": prompts, "widths": {}}
    for width in (int(w) for w in args.widths.split(",") if w):
        config = {
            "target": str(args.target.resolve()),
            "draft": str(args.draft.resolve()),
            "width": width,
            "trials": args.trials,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "prompts": prompts,
        }
        path = args.output_dir / f"k{width}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        with path.with_suffix(".log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.dspark_single_stream_bench",
                    "--worker-config",
                    str(path),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=3600,
            )
        summary["widths"][str(width)] = json.loads(
            path.with_suffix(".result.json").read_text()
        )
    base = summary["widths"].get("0")
    rows = [
        "| Prompt | Target-only tok/s | "
        + " | ".join(
            f"K={w} tok/s (accept, speedup)" for w in summary["widths"] if w != "0"
        )
        + " |",
        "| --- | --- | "
        + " | ".join("---" for w in summary["widths"] if w != "0")
        + " |",
    ]
    for name in prompts:
        cells = []
        for w, res in summary["widths"].items():
            if w == "0":
                continue
            item = res[name]
            ratio = (
                item["tokens_per_s_median"] / base[name]["tokens_per_s_median"]
                if base
                else float("nan")
            )
            accept = item["accepted_length_median"]
            cells.append(
                f"{item['tokens_per_s_median']:.1f} ({accept:.2f}, {ratio:.2f}x)"
                if accept is not None
                else f"{item['tokens_per_s_median']:.1f}"
            )
        rows.append(
            f"| {name} | {base[name]['tokens_per_s_median']:.1f} | "
            + " | ".join(cells)
            + " |"
            if base
            else f"| {name} | – | " + " | ".join(cells) + " |"
        )
    table = "\n".join(rows)
    (args.output_dir / "summary.md").write_text(table + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(table)


if __name__ == "__main__":
    main()
