# SPDX-License-Identifier: Apache-2.0
"""Measure DSpark step costs through the serving path and write the cost model.

For every drafted width one ``vllm serve`` runs in the fixed mode, and for
width 0 the same speculative server runs in the bypass mode (drafter loaded,
features captured, no drafts): that is the step the adaptive planner weighs
drafting against. Each server is driven, for every profiled request count
and decode context, by that many concurrent streaming requests of exactly
the context's input length with ``ignore_eos`` and an output budget that grows
with the batch's prefill work and the width, so the requests admitted first
are still decoding when the last prompt finishes prefilling; after one
warmup batch the measured batch records every streamed chunk's arrival time
per request, and the step cost of the cell is the median gap between
consecutive chunks of one request inside the window where every request of
the batch is decoding (its p95 is the uncertainty record). Prefill chunks
are larger than in serving (``--batch-tokens`` 2048) so that window exists;
decode steps are unaffected while the verify rows stay below the budget. The
drafter's own work per step (batched backbone and host bookkeeping) comes
from the in-process step profiler at one width per request level and is
subtracted from the drafted cells' step to give the planner's target cost.
The artifact (``dspark-cost/2``) carries the samples, the model-pair
manifest, the machine identity and the profile settings. Run it alone on an
idle machine.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

from tools.dspark_serving_check import Server
from vllm_metal.v1.dspark.calibration import CalibrationManifest
from vllm_metal.v1.dspark.planner import CostModel, CostSample


def build_prompts(
    tokenizer, repo: Path, input_length: int, count: int
) -> list[list[int]]:
    from tools.dspark_perf_bench import build_prompts as bench_prompts

    return bench_prompts(tokenizer, repo, input_length, count)


async def timed_batch(
    base: str, prompts: list[list[int]], output: int
) -> list[list[float]]:
    """Chunk arrival times per request for one concurrent streaming batch."""

    async def one(client: httpx.AsyncClient, prompt: list[int]) -> list[float]:
        body = {
            "model": "target",
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": output,
            "ignore_eos": True,
            "stream": True,
            "return_token_ids": True,
        }
        arrivals: list[float] = []
        async with client.stream(
            "POST", f"{base}/v1/completions", json=body
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                choice = json.loads(payload)["choices"][0]
                if choice.get("token_ids") or choice.get("text"):
                    arrivals.append(time.perf_counter())
        return arrivals

    async with httpx.AsyncClient(timeout=1800) as client:
        return await asyncio.gather(*(one(client, prompt) for prompt in prompts))


def step_gaps(arrivals: list[list[float]]) -> list[float]:
    """Per-step durations while every request of the batch is decoding."""
    start = max(times[0] for times in arrivals)
    end = min(times[-1] for times in arrivals)
    gaps = []
    for times in arrivals:
        for a, b in zip(times, times[1:], strict=False):
            if a >= start and b <= end:
                gaps.append(b - a)
    return gaps


def in_process_costs(
    args, env: dict[str, str], width: int, requests: int
) -> tuple[float, float, int]:
    """Draft and host milliseconds per step from the in-process profiler."""
    config = {
        "target": str(args.target.resolve()),
        "draft": str(args.draft.resolve()),
        "width": width,
        "concurrency": requests,
        "output_length": args.output_length,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.batch_tokens,
    }
    path = args.output_dir / f"inprocess-k{width}-c{requests}.json"
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
            env=dict(env, VLLM_METAL_DECODE_PIPELINE="0"),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=3600,
        )
    result = json.loads(path.with_suffix(".result.json").read_text())
    ms = result["ms_per_step"]
    host = sum(
        ms[name].get("p50", 0.0)
        for name in ("ingest_host", "context_eval", "propose_other")
    )
    return ms["draft"].get("p50", 0.0), host, result["steps"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--requests", default="1,2,4,8,16")
    parser.add_argument("--widths", default="0,1,2,4,7")
    parser.add_argument(
        "--contexts", default="128,640,1536", help="input tokens per request"
    )
    parser.add_argument("--output-length", type=int, default=64)
    parser.add_argument("--in-process-width", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--batch-tokens", type=int, default=2048)
    parser.add_argument("--memory-fraction", default="0.2")
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument(
        "--async-scheduling",
        action="store_true",
        help="start every server with --async-scheduling (DSpark supports it since M9b)",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    requests = [int(v) for v in args.requests.split(",") if v]
    widths = [int(v) for v in args.widths.split(",") if v]
    contexts = [int(v) for v in args.contexts.split(",") if v]
    if 0 not in widths:
        parser.error("--widths must include 0")
    if max(requests) > args.max_num_seqs:
        parser.error("--requests must not exceed --max-num-seqs")

    def output_budget(count: int, context: int, width: int) -> int:
        """Tokens per request so the first admitted request outlives the batch's prefill."""
        prefill_steps = -(-count * context // args.batch_tokens)
        return args.output_length + 2 * prefill_steps * (width + 1)

    longest = max(
        context + output_budget(count, context, width)
        for context in contexts
        for count in requests
        for width in widths
    )
    if longest > args.max_model_len:
        parser.error(f"the longest cell needs {longest} tokens; raise --max-model-len")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.target))
    env = dict(
        os.environ,
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
        VLLM_METAL_USE_PAGED_ATTENTION="1",
        VLLM_METAL_MEMORY_FRACTION=str(args.memory_fraction),
        VLLM_METAL_BUILD_FROM_SOURCE="1",
        HF_HUB_OFFLINE="1",
    )
    server_args = argparse.Namespace(
        target=args.target,
        draft=args.draft,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        batch_tokens=args.batch_tokens,
        memory_fraction=args.memory_fraction,
    )
    # Drafter work per request level, in process, at one width.
    drafter_costs: dict[int, tuple[float, float, int]] = {}
    for count in requests:
        drafter_costs[count] = in_process_costs(args, env, args.in_process_width, count)
        print(
            f"in-process requests {count}: draft {drafter_costs[count][0]:.2f} ms, "
            f"host {drafter_costs[count][1]:.2f} ms",
            flush=True,
        )
    prompts = {
        (context, count): build_prompts(tokenizer, args.repo, context, count)
        for context in contexts
        for count in requests
    }
    samples: list[CostSample] = []
    manifest = None
    measurements: dict[str, dict] = {}
    for width in widths:
        served_width = width or max(widths)
        mode_env = {"VLLM_METAL_DSPARK_MODE": "bypass"} if width == 0 else None
        server = Server(
            server_args,
            served_width,
            False,
            args.output_dir,
            name=f"k{width}" if width else "bypass",
            async_scheduling=args.async_scheduling,
            env=mode_env,
            expect_in_log="mode=bypass" if width == 0 else "mode=fixed",
        )
        try:
            server.wait_ready(args.startup_timeout)
            if manifest is None:
                manifest = None  # filled from the in-process profile below
            for context in contexts:
                for count in requests:
                    batch = prompts[(context, count)]
                    budget = output_budget(count, context, width)
                    asyncio.run(timed_batch(server.base, batch, budget))  # warmup
                    arrivals = asyncio.run(timed_batch(server.base, batch, budget))
                    gaps = step_gaps(arrivals)
                    if len(gaps) < 8:
                        raise RuntimeError(
                            f"width {width} requests {count} context {context}: only {len(gaps)} "
                            "in-window steps; raise --output-length"
                        )
                    gaps.sort()
                    step_ms = statistics.median(gaps) * 1000
                    p95_ms = gaps[min(len(gaps) - 1, int(0.95 * len(gaps)))] * 1000
                    draft, host, steps = (
                        drafter_costs[count] if width else (0.0, 0.0, 0)
                    )
                    decode_context = context + args.output_length // 2
                    samples.append(
                        CostSample(
                            requests=count,
                            width=width,
                            rows=count * (width + 1),
                            context=decode_context,
                            step_ms=step_ms,
                            target_ms=max(0.0, step_ms - draft - host),
                            draft_ms=draft,
                            host_ms=host,
                            steps=len(gaps),
                            step_p95_ms=p95_ms,
                        )
                    )
                    measurements[f"k{width}/c{count}/ctx{context}"] = {
                        "gaps_ms": [g * 1000 for g in gaps],
                    }
                    print(
                        f"width {width} requests {count} context {context}: step {step_ms:.1f} ms "
                        f"(p95 {p95_ms:.1f}) over {len(gaps)} steps",
                        flush=True,
                    )
        finally:
            server.stop()
    inprocess = json.loads(
        (
            args.output_dir
            / f"inprocess-k{args.in_process_width}-c{requests[0]}.result.json"
        ).read_text()
    )
    manifest = CalibrationManifest.from_dict(inprocess["manifest"])
    model = CostModel(
        manifest=manifest,
        samples=samples,
        machine={
            "platform": platform.platform(),
            "device": inprocess.get("device"),
            "memory_fraction": str(args.memory_fraction),
            "output_length": args.output_length,
            "batch_tokens": args.batch_tokens,
            "contexts": contexts,
            "in_process_width": args.in_process_width,
            "served_widths": widths,
        },
    )
    (args.output_dir / "cost.json").write_text(model.to_json())
    (args.output_dir / "measurements.json").write_text(json.dumps(measurements) + "\n")
    print(f"wrote {args.output_dir / 'cost.json'}")


if __name__ == "__main__":
    main()
