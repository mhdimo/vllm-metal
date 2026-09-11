# SPDX-License-Identifier: Apache-2.0
"""Accepted length per drafting round on the official DSpark evaluation prompts.

The DSpark paper's Table 1 reports the mean accepted length per decoding
round (drafted tokens accepted plus the target's bonus or correction token)
for a fixed block of seven drafts over the DeepSpec evaluation sets. This
tool runs the same prompt sets (first turn, the target's chat template
without thinking) through an offline engine on the pinned Metal pair in the
fixed mode at the given width and reports, per data set, the accepted length
per round from the proposer's own counters, next to the paper's number for
the same target where known. The paper samples at temperature 1.0 with
standard (rejection-sampling) verification; the default here is the same
temperature through the port's exact stochastic verification (seeded
requests), and ``--temperature 0`` measures greedy decoding. The protocol
still differs from the paper's in ways the record states: a subset of
prompts per set, a shorter output budget, and quantized target and drafter
weights (the drafter's precision is the serving knob
``VLLM_METAL_DSPARK_DRAFT_PRECISION``).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Table 1 of arXiv:2607.05147 (DSpark rows; accepted length incl. the bonus token).
PAPER_ACCEPTED_LENGTH: dict[str, dict[str, float]] = {
    "Qwen3-4B": {
        "gsm8k": 6.11,
        "math500": 5.70,
        "aime25": 4.89,
        "mbpp": 5.13,
        "humaneval": 5.38,
        "livecodebench": 4.86,
        "mt-bench": 3.64,
        "alpaca": 3.54,
        "arena-hard-v2": 3.29,
    },
    "Qwen3-8B": {"gsm8k": 6.17, "math500": 5.78, "aime25": 5.01},
}


def load_prompts(path: Path, limit: int, seed: int) -> list[str]:
    """First turn of each record, a deterministic subset of ``limit`` prompts."""
    import random

    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    turns = [rec["turns"][0] for rec in records if rec.get("turns")]
    if limit and len(turns) > limit:
        rng = random.Random(seed)
        turns = rng.sample(turns, limit)
    return turns


def worker(config: dict, output: Path) -> None:
    import mlx.core as mx
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from vllm_metal.v1.dspark_proposer import DSparkProposer

    tokenizer = AutoTokenizer.from_pretrained(config["target"])
    llm = LLM(
        model=config["target"],
        max_model_len=config["max_model_len"],
        max_num_seqs=config["concurrency"],
        max_num_batched_tokens=config["max_num_batched_tokens"],
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        disable_log_stats=True,
        speculative_config={
            "method": "dspark",
            "model": config["draft"],
            "num_speculative_tokens": config["width"],
        },
    )
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    proposer = runner._drafter
    assert isinstance(proposer, DSparkProposer)
    params = SamplingParams(
        temperature=config["temperature"],
        max_tokens=config["output_length"],
        seed=config["seed"] if config["temperature"] > 0 else None,
        ignore_eos=False,
    )
    results = {}
    for name, prompts in config["prompts"].items():
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for prompt in prompts
        ]
        before = proposer.counters.snapshot()
        mx.synchronize()
        started = time.perf_counter()
        outputs = llm.generate(texts, params, use_tqdm=False)
        mx.synchronize()
        wall = time.perf_counter() - started
        after = proposer.counters.snapshot()
        verified = after["verified_requests"] - before["verified_requests"]
        accepted = after["accepted_tokens"] - before["accepted_tokens"]
        scheduled = after["scheduled_tokens"] - before["scheduled_tokens"]
        generated = sum(len(item.outputs[0].token_ids) for item in outputs)
        positions = {
            pos: after["position_acceptances"].get(pos, 0)
            - before["position_acceptances"].get(pos, 0)
            for pos in after["position_opportunities"]
        }
        opportunities = {
            pos: after["position_opportunities"].get(pos, 0)
            - before["position_opportunities"].get(pos, 0)
            for pos in after["position_opportunities"]
        }
        results[name] = {
            "prompts": len(prompts),
            "generated_tokens": generated,
            "verified_rounds": verified,
            "accepted_draft_tokens": accepted,
            "scheduled_draft_tokens": scheduled,
            "accepted_length": (1 + accepted / verified) if verified else None,
            "acceptance_rate": (accepted / scheduled) if scheduled else None,
            "position_acceptance": {
                str(pos): (positions[pos] / opportunities[pos])
                if opportunities[pos]
                else None
                for pos in sorted(opportunities)
            },
            "wall_s": wall,
            "tokens_per_s": generated / wall if wall else None,
            "finished_by_length": sum(
                1 for item in outputs if item.outputs[0].finish_reason == "length"
            ),
        }
        print(
            f"{name}: {len(prompts)} prompts, {generated} tokens, "
            f"accepted length {results[name]['accepted_length']:.2f} "
            f"({verified} rounds), {generated / wall:.1f} tok/s",
            flush=True,
        )
    output.write_text(json.dumps(results, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--datasets-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--datasets", default="gsm8k,math500,humaneval,mbpp,mt-bench,alpaca"
    )
    parser.add_argument("--prompts-per-set", type=int, default=64)
    parser.add_argument("--width", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output-length", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--memory-fraction", default="0.3")
    parser.add_argument("--paper-target", default="Qwen3-4B")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_config:
        worker(
            json.loads(args.worker_config.read_text()),
            args.worker_config.with_suffix(".result.json"),
        )
        return
    if any(
        value is None
        for value in (args.target, args.draft, args.datasets_dir, args.output_dir)
    ):
        parser.error("--target, --draft, --datasets-dir and --output-dir are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = {
        name: load_prompts(
            args.datasets_dir / f"{name}.jsonl", args.prompts_per_set, args.seed
        )
        for name in args.datasets.split(",")
        if name
    }
    config = {
        "target": str(args.target.resolve()),
        "draft": str(args.draft.resolve()),
        "width": args.width,
        "temperature": args.temperature,
        "output_length": args.output_length,
        "concurrency": args.concurrency,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "seed": args.seed,
        "prompts": prompts,
        "draft_precision": os.environ.get(
            "VLLM_METAL_DSPARK_DRAFT_PRECISION", "quantized"
        ),
    }
    path = args.output_dir / "config.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    env = dict(
        os.environ,
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
        VLLM_METAL_USE_PAGED_ATTENTION="1",
        VLLM_METAL_MEMORY_FRACTION=str(args.memory_fraction),
        VLLM_METAL_BUILD_FROM_SOURCE="1",
        HF_HUB_OFFLINE="1",
    )
    with (args.output_dir / "worker.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.dspark_acceptance_eval",
                "--worker-config",
                str(path),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=4 * 3600,
        )
    results = json.loads(path.with_suffix(".result.json").read_text())
    paper = PAPER_ACCEPTED_LENGTH.get(args.paper_target, {})
    rows = [
        "| Data set | Prompts | Accepted length (this port) | Paper (bf16, full set, 2,048 tokens) | Acceptance rate | tok/s |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, item in results.items():
        ref = paper.get(name)
        rows.append(
            f"| {name} | {item['prompts']} | {item['accepted_length']:.2f} | "
            f"{ref:.2f} | {100 * item['acceptance_rate']:.1f}% | {item['tokens_per_s']:.1f} |"
            if ref is not None and item["accepted_length"] is not None
            else f"| {name} | {item['prompts']} | – | – | – | – |"
        )
    table = "\n".join(rows)
    (args.output_dir / "summary.md").write_text(table + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": {k: v for k, v in config.items() if k != "prompts"},
                "results": results,
                "paper": paper,
            },
            indent=2,
        )
        + "\n"
    )
    print(table)


if __name__ == "__main__":
    main()
