# SPDX-License-Identifier: Apache-2.0
"""Probe whether the target-only engine is numerically stable at a divergent prefix.

For every first divergence recorded by ``dspark_memory_check`` (a failure or
classified artifact), the committed prefix up to the divergence is fed to a
fresh target-only engine with no drafter under several execution shapes
(prefill chunk budgets, paged attention on and off, NAX on and off). The
engine prefills all but the last prefix token and decodes twice, so the second
token is a one-row decode at the divergent position. A prefix is ``unstable``
when the shapes disagree on that greedy token; every shape's top logits are
kept so the disagreement is visible, not inferred.

Each shape runs in its own process for clean Metal state. This is a diagnosis
of the target model under bfloat16 execution, not a DSpark check: an unstable
prefix cannot support an exact-token comparison between any two execution
paths, speculative or not.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_SHAPES = (
    # label, extra environment, max_num_batched_tokens
    ("paged-chunk64", {}, 64),
    ("paged-chunk16", {}, 16),
    ("paged-chunk4", {}, 4),
    ("paged-chunk64-nonax", {"VLLM_METAL_DISABLE_NAX": "1"}, 64),
    # The MLX KV-cache path only accepts the automatic memory fraction.
    (
        "nonpaged-chunk64",
        {"VLLM_METAL_USE_PAGED_ATTENTION": "0", "VLLM_METAL_MEMORY_FRACTION": "auto"},
        64,
    ),
)


def divergent_prefixes(artifact: dict) -> list[dict]:
    """First divergences as (request, output position, prefix, both tokens)."""
    if "divergences" in artifact:  # classified.json
        run_dir = Path(artifact["run_dir"])
        width = artifact["width"]
        failure = json.loads((run_dir / f"k{width}.result.failure.json").read_text())
        entries = [
            (d["request"], d["output_position"]) for d in artifact["divergences"]
        ]
    else:  # kN.result.failure.json
        failure = artifact
        entries = []
        for index, (expected, actual) in enumerate(
            zip(failure["expected_tokens"], failure["actual_tokens"], strict=True)
        ):
            for position, (left, right) in enumerate(
                zip(expected, actual, strict=True)
            ):
                if left != right:
                    entries.append((index, position))
                    break
    cases = []
    for index, position in entries:
        prompt = failure["prompt_token_ids"][index]
        expected = failure["expected_tokens"][index]
        cases.append(
            {
                "request": index,
                "output_position": position,
                "prefix": prompt + expected[:position],
                "prompt_length": len(prompt),
                "baseline_token": expected[position],
                "speculative_token": failure["actual_tokens"][index][position],
            }
        )
    return cases


def worker(config: dict, output: Path) -> None:
    from vllm import LLM, SamplingParams

    from tools.dspark_memory_check import top_logit_rows

    llm = LLM(
        model=config["target"],
        max_model_len=config["max_model_len"],
        max_num_seqs=1,
        max_num_batched_tokens=config["max_num_batched_tokens"],
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        disable_log_stats=True,
    )
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    traces: list[list[dict]] = []
    current: list[dict] = []
    real = runner._sample_paged_batch

    def traced(*args, **kwargs):
        state = runner._execute_model_state
        if (
            state is not None
            and state.logits is not None
            and not state.intermediate_only
        ):
            current.extend(top_logit_rows(state, 0, len(current)))
        return real(*args, **kwargs)

    runner._sample_paged_batch = traced
    results = []
    for case in config["cases"]:
        prefix = case["prefix"]
        current = []
        outputs = llm.generate(
            [{"prompt_token_ids": prefix[:-1]}],
            SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True),
            use_tqdm=False,
        )
        tokens = list(outputs[0].outputs[0].token_ids)
        decode_rows = [row for row in current if row["pos"] == len(prefix)]
        results.append(
            {
                "request": case["request"],
                "output_position": case["output_position"],
                "prefill_token": tokens[0],
                "prefill_token_expected": prefix[-1],
                "greedy_token": tokens[1],
                "top": decode_rows[-1]["top"][:5] if decode_rows else None,
                "kinds": [(row["kind"], row["rows"]) for row in current],
            }
        )
        traces.append(current)
    output.write_text(json.dumps(results, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path)
    parser.add_argument(
        "--artifact", type=Path, help="kN.result.failure.json or classified.json"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--memory-fraction", default="0.22")
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_config:
        worker(
            json.loads(args.worker_config.read_text()),
            args.worker_config.with_suffix(".result.json"),
        )
        return
    if args.target is None or args.artifact is None or args.output is None:
        parser.error("--target, --artifact and --output are required")
    if not args.target.is_dir():
        parser.error("--target must be an already downloaded snapshot")
    cases = divergent_prefixes(json.loads(args.artifact.read_text()))
    if not cases:
        parser.error("no first divergence found in the artifact")
    longest = max(len(case["prefix"]) for case in cases)
    if longest + 2 > args.max_model_len:
        parser.error(f"--max-model-len must exceed the longest prefix ({longest}) by 2")
    work = args.output.with_suffix("")
    work.mkdir(parents=True, exist_ok=True)
    base_env = dict(
        os.environ,
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
        VLLM_METAL_USE_PAGED_ATTENTION="1",
        VLLM_METAL_MEMORY_FRACTION=str(args.memory_fraction),
        VLLM_METAL_BUILD_FROM_SOURCE="1",
        HF_HUB_OFFLINE="1",
    )
    shapes = {}
    for label, extra, batch_tokens in DEFAULT_SHAPES:
        config = {
            "target": str(args.target.resolve()),
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": batch_tokens,
            "cases": cases,
        }
        path = work / f"{label}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        with path.with_suffix(".log").open("w") as log:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.dspark_target_stability",
                    "--worker-config",
                    str(path),
                ],
                env={**base_env, **extra},
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1800,
            )
        result_path = path.with_suffix(".result.json")
        if completed.returncode != 0 or not result_path.exists():
            shapes[label] = {
                "error": f"exit {completed.returncode}; see {path.with_suffix('.log')}"
            }
            continue
        shapes[label] = {"results": json.loads(result_path.read_text())}

    verdicts = []
    for index, case in enumerate(cases):
        outcomes = {}
        for label, shape in shapes.items():
            if "results" in shape:
                item = shape["results"][index]
                outcomes[label] = {
                    "greedy_token": item["greedy_token"],
                    "top": item["top"],
                }
        tokens = {item["greedy_token"] for item in outcomes.values()}
        verdicts.append(
            {
                "request": case["request"],
                "output_position": case["output_position"],
                "prefix_length": len(case["prefix"]),
                "baseline_token": case["baseline_token"],
                "speculative_token": case["speculative_token"],
                "engine_outcomes": outcomes,
                "distinct_greedy_tokens": sorted(tokens),
                "stable": len(tokens) == 1,
            }
        )
    summary = {
        "target": str(args.target),
        "artifact": str(args.artifact),
        "shapes": [label for label, _, _ in DEFAULT_SHAPES],
        "errors": {
            label: shape["error"] for label, shape in shapes.items() if "error" in shape
        },
        "cases": verdicts,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    for verdict in verdicts:
        print(
            f"request {verdict['request']} pos {verdict['output_position']}: "
            f"{'stable' if verdict['stable'] else 'UNSTABLE'} "
            f"{verdict['distinct_greedy_tokens']} "
            f"(baseline {verdict['baseline_token']}, speculative {verdict['speculative_token']})"
        )
    if summary["errors"]:
        print("shape errors:", summary["errors"])


if __name__ == "__main__":
    main()
