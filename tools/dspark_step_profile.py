# SPDX-License-Identifier: Apache-2.0
"""In-process per-step cost attribution for DSpark at a fixed K.

Runs an offline engine on natural prompts and times, per scheduler step, the
phases the proposer seam exposes: the target forward with verification and
sampling (from ``execute_model`` entry to the proposer call), feature ingest
(host bookkeeping), context materialization (the proposer's evaluation of the
appended K/V), the batched draft backbone and heads, and the remainder of
``propose``. Phase boundaries are already synchronization points in the
runner (logits evaluation, context evaluation, draft evaluation), so no extra
barriers are added; the numbers are still instrumented, in-process costs and
not the serving benchmark (``tools.dspark_perf_bench``).

K=0 runs the same engine without a drafter and reports the target phase only,
which is the one-row decode cost the K>0 target phase (K+1 rows) compares to.
The decode pipeline is disabled for every width here so each step samples
synchronously; the serving benchmark keeps the production default.
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


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
        return ordered[index]

    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def worker(config: dict, output: Path) -> None:
    import mlx.core as mx
    from vllm import LLM, SamplingParams

    from tools.dspark_memory_check import NATURAL_PROMPTS

    width = config["width"]
    llm = LLM(
        model=config["target"],
        max_model_len=config["max_model_len"],
        max_num_seqs=config["concurrency"],
        max_num_batched_tokens=config["max_num_batched_tokens"],
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
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    phases: dict[str, list[float]] = {
        "target": [],
        "ingest_host": [],
        "context_eval": [],
        "draft": [],
        "propose_other": [],
        "step": [],
    }
    emitted: list[int] = []
    marks: dict[str, float] = {}
    real_execute = runner.execute_model
    real_sample = runner._sample_paged_batch

    def execute(scheduler_output, *args, **kwargs):
        marks["step_start"] = time.perf_counter()
        marks.pop("propose_start", None)
        return real_execute(scheduler_output, *args, **kwargs)

    def sample(*args, **kwargs):
        result = real_sample(*args, **kwargs)
        end = time.perf_counter()
        start = marks.get("step_start")
        if start is not None:
            phases["step"].append(end - start)
            if "propose_start" not in marks:
                phases["target"].append(end - start)
        batch = result[0] if isinstance(result, tuple) else None
        if batch is not None:
            emitted.append(sum(len(row) for row in batch.sampled_tokens))
        return result

    runner.execute_model = execute
    runner._sample_paged_batch = sample

    if width:
        from vllm_metal.v1.dspark_proposer import DSparkProposer

        proposer = runner._drafter
        assert isinstance(proposer, DSparkProposer)
        real_propose = proposer.propose
        real_ingest = proposer._ingest_step
        real_draft = proposer._batch_draft

        def propose(ctx):
            marks["propose_start"] = time.perf_counter()
            start = marks.get("step_start")
            if start is not None:
                phases["target"].append(marks["propose_start"] - start)
            marks["ingest_end"] = None
            marks["draft"] = 0.0
            result = real_propose(ctx)
            end = time.perf_counter()
            total = end - marks["propose_start"]
            ingest = marks.get("ingest_time", 0.0)
            draft = marks.get("draft", 0.0)
            context = 0.0
            if (
                marks.get("ingest_end") is not None
                and marks.get("draft_start") is not None
            ):
                context = marks["draft_start"] - marks["ingest_end"]
            elif marks.get("ingest_end") is not None:
                context = end - marks["ingest_end"]
            phases["ingest_host"].append(ingest)
            phases["context_eval"].append(context)
            phases["draft"].append(draft)
            phases["propose_other"].append(max(0.0, total - ingest - context - draft))
            marks.pop("draft_start", None)
            marks["ingest_time"] = 0.0
            return result

        def ingest(ctx):
            start = time.perf_counter()
            try:
                return real_ingest(ctx)
            finally:
                marks["ingest_end"] = time.perf_counter()
                marks["ingest_time"] = marks["ingest_end"] - start

        def draft(plans):
            marks["draft_start"] = time.perf_counter()
            try:
                return real_draft(plans)
            finally:
                marks["draft"] = time.perf_counter() - marks["draft_start"]

        proposer.propose = propose
        proposer._ingest_step = ingest
        proposer._batch_draft = draft

    prompts = [
        f"Request {index + 1}. {NATURAL_PROMPTS[index % len(NATURAL_PROMPTS)]}"
        for index in range(config["concurrency"])
    ]
    params = SamplingParams(
        temperature=0.0, max_tokens=config["output_length"], ignore_eos=True
    )
    llm.generate(prompts, params, use_tqdm=False)  # warmup, not measured
    for lists in phases.values():
        lists.clear()
    emitted.clear()
    mx.synchronize()
    started = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    mx.synchronize()
    wall = time.perf_counter() - started
    generated = sum(len(item.outputs[0].token_ids) for item in outputs)
    steps = len(phases["step"])
    from vllm_metal.v1.dspark.calibration import CalibrationManifest

    result = {
        "config": config,
        "manifest": (
            CalibrationManifest.from_runner(runner).to_dict() if width else None
        ),
        "device": mx.device_info(),
        "steps": steps,
        "generated_tokens": generated,
        "wall_s": wall,
        "tokens_per_second_instrumented": generated / wall if wall else None,
        "tokens_per_step": generated / steps if steps else None,
        "ms_per_step": {
            name: {
                k: (v * 1000 if isinstance(v, float) else v)
                for k, v in summarize(values).items()
            }
            for name, values in phases.items()
        },
        "share_of_step": {
            name: (
                sum(values) / sum(phases["step"])
                if phases["step"] and sum(phases["step"])
                else None
            )
            for name, values in phases.items()
            if name != "step"
        },
        "peak_mlx_bytes": mx.get_peak_memory(),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--widths", default="0,1,2,4,7")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output-length", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--memory-fraction", default="0.22")
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
    env = dict(
        os.environ,
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
        VLLM_METAL_USE_PAGED_ATTENTION="1",
        VLLM_METAL_MEMORY_FRACTION=str(args.memory_fraction),
        VLLM_METAL_BUILD_FROM_SOURCE="1",
        # Synchronous sampling on every step so K=0 phases are attributable;
        # the serving benchmark keeps the production pipeline default.
        VLLM_METAL_DECODE_PIPELINE="0",
        HF_HUB_OFFLINE="1",
    )
    rows = []
    for width in (int(w) for w in args.widths.split(",") if w):
        config = {
            "target": str(args.target.resolve()),
            "draft": str(args.draft.resolve()),
            "width": width,
            "concurrency": args.concurrency,
            "output_length": args.output_length,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
        }
        path = args.output_dir / f"k{width}-c{args.concurrency}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        with path.with_suffix(".log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.dspark_step_profile",
                    "--worker-config",
                    str(path),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=3600,
            )
        result = json.loads(path.with_suffix(".result.json").read_text())
        rows.append(result)
        ms = result["ms_per_step"]
        print(
            f"K={width} C={args.concurrency}: {result['steps']} steps, "
            f"{result['tokens_per_step']:.2f} tok/step, step p50 {ms['step']['p50']:.1f} ms "
            f"(target {ms['target']['p50']:.1f}, draft {ms['draft'].get('p50', 0.0):.1f}, "
            f"ingest {ms['ingest_host'].get('p50', 0.0):.2f}, context {ms['context_eval'].get('p50', 0.0):.2f}), "
            f"{result['tokens_per_second_instrumented']:.1f} tok/s instrumented",
            flush=True,
        )
    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
