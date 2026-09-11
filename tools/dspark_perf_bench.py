# SPDX-License-Identifier: Apache-2.0
"""Fixed-K serving performance protocol for DSpark over the OpenAI HTTP server.

Workload buckets (input tokens x output tokens) and client concurrencies are
declared up front. For every speculative width the target-only server and the
speculative server are both alive (only one receives requests at a time);
after one warmup repetition per server the bucket is measured in at least
five paired repetitions with alternating order (a pair taken while another
process loaded the machine is discarded and repeated; see ``--max-load``). Every request is streamed
and records time to first token, end-to-end latency, time per output token
and the inter-arrival gaps between streamed chunks (a speculative burst is
one arrival, so gaps are what a reader perceives). Per repetition: output
tokens per second, requests per second, mean TTFT, median TPOT, p95 and p99
gaps, and goodput at the declared SLO. Paired differences carry a bootstrap
95% interval of the median. The proposed release gate from the specification
(at least 10% median benefit with the interval excluding no benefit) is
evaluated per bucket and reported, never asserted.

The target-only reference paired with each width uses the production
default scheduler with the same ``--no-async-scheduling`` the speculative
server needs; ``--async-reference`` also measures a target-only server with
asynchronous scheduling, the best deployable target-only configuration.
Prompts are distinct per request and built from the repository's own
documentation text so long inputs are natural prose, not repeated fixtures.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from tools.dspark_serving_check import Server, metrics

FILLER_DOCS = (
    "docs/design/dspark.md",
    "docs/design/dspark-validation.md",
    "docs/index.md",
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def bootstrap_median_interval(
    differences: list[float], resamples: int = 5000, seed: int = 0
) -> tuple[float, float]:
    rng = random.Random(seed)
    medians = []
    for _ in range(resamples):
        sample = [rng.choice(differences) for _ in differences]
        medians.append(statistics.median(sample))
    return percentile(medians, 0.025), percentile(medians, 0.975)


def build_prompts(
    tokenizer, repo: Path, input_length: int, count: int
) -> list[list[int]]:
    """``count`` distinct prompts of exactly ``input_length`` tokens."""
    from tools.dspark_memory_check import NATURAL_PROMPTS

    filler_text = "\n\n".join((repo / name).read_text() for name in FILLER_DOCS)
    filler = tokenizer.encode(filler_text, add_special_tokens=False)
    prompts = []
    for index in range(count):
        question = tokenizer.encode(
            NATURAL_PROMPTS[index % len(NATURAL_PROMPTS)], add_special_tokens=False
        )
        need = input_length - len(question)
        if need < 0:
            prompts.append(question[:input_length])
            continue
        start = (index * 977) % max(1, len(filler) - need)
        prompts.append(filler[start : start + need] + question)
        assert len(prompts[-1]) == input_length
    return prompts


async def timed_request(
    client: httpx.AsyncClient, base: str, prompt: list[int], output: int
) -> dict:
    body = {
        "model": "target",
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": output,
        "ignore_eos": True,
        "stream": True,
        "return_token_ids": True,
    }
    sent = time.perf_counter()
    arrivals: list[tuple[float, int]] = []
    async with client.stream("POST", f"{base}/v1/completions", json=body) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            choice = json.loads(payload)["choices"][0]
            tokens = len(choice.get("token_ids") or []) or (
                1 if choice.get("text") else 0
            )
            if tokens:
                arrivals.append((time.perf_counter(), tokens))
    if not arrivals:
        raise RuntimeError("stream produced no tokens")
    first = arrivals[0][0]
    last = arrivals[-1][0]
    total = sum(count for _, count in arrivals)
    gaps = [b - a for (a, _), (b, _) in zip(arrivals, arrivals[1:], strict=False)]
    return {
        "tokens": total,
        "ttft": first - sent,
        "e2e": last - sent,
        "tpot": (last - first) / (total - 1) if total > 1 else 0.0,
        "gaps": gaps,
        "arrivals": len(arrivals),
    }


async def run_repetition(
    base: str, prompts: list[list[int]], output: int, slo: dict
) -> dict:
    async with httpx.AsyncClient(timeout=1800) as client:
        started = time.perf_counter()
        results = await asyncio.gather(
            *(timed_request(client, base, prompt, output) for prompt in prompts)
        )
        wall = time.perf_counter() - started
    gaps = [gap for item in results for gap in item["gaps"]]
    good = sum(
        1
        for item in results
        if item["ttft"] <= slo["ttft_s"]
        and (percentile(item["gaps"], 0.95) if item["gaps"] else 0.0)
        <= slo["gap_p95_s"]
    )
    return {
        "wall_s": wall,
        "requests": len(results),
        "output_tokens": sum(item["tokens"] for item in results),
        "output_tokens_per_s": sum(item["tokens"] for item in results) / wall,
        "requests_per_s": len(results) / wall,
        "ttft_mean_s": statistics.fmean(item["ttft"] for item in results),
        "tpot_median_s": statistics.median(item["tpot"] for item in results),
        "gap_p50_s": percentile(gaps, 0.5) if gaps else 0.0,
        "gap_p95_s": percentile(gaps, 0.95) if gaps else 0.0,
        "gap_p99_s": percentile(gaps, 0.99) if gaps else 0.0,
        "tokens_per_arrival": sum(item["tokens"] for item in results)
        / sum(item["arrivals"] for item in results),
        "goodput_requests_per_s": good / wall,
        "slo_met": good,
    }


PAIRED_METRICS = {
    "output_tokens_per_s": "higher",
    "tpot_median_s": "lower",
    "gap_p95_s": "lower",
    "goodput_requests_per_s": "higher",
    "ttft_mean_s": "lower",
}


def paired_summary(reference: list[dict], candidate: list[dict]) -> dict:
    summary = {}
    for metric, better in PAIRED_METRICS.items():
        ref = [rep[metric] for rep in reference]
        cand = [rep[metric] for rep in candidate]
        diffs = [c - r for r, c in zip(ref, cand, strict=True)]
        relative = [(c - r) / r if r else 0.0 for r, c in zip(ref, cand, strict=True)]
        if better == "lower":
            relative = [-value for value in relative]
        low, high = bootstrap_median_interval(relative)
        summary[metric] = {
            "reference_median": statistics.median(ref),
            "candidate_median": statistics.median(cand),
            "median_relative_benefit": statistics.median(relative),
            "benefit_ci95": [low, high],
            "paired_differences": diffs,
            "gate_10pct_ci_excludes_zero": statistics.median(relative) >= 0.10
            and low > 0.0,
        }
    return summary


HEAVY_CPU_PERCENT = 40.0
# The run's own processes (the servers, this client and its children) and the
# desktop's daemons are not outside load; anything else above HEAVY_CPU_PERCENT
# is, another Python process included.
_OWN_PROCESS = re.compile(r"VLLM|vllm|\bzsh\b|\bps\b")
_SYSTEM_PATHS = ("/System", "/Applications", "/Library", "/usr/libexec", "/usr/sbin")


def heavy_other_processes(ps_lines: list[str], own_pids: set[int] | None = None) -> int:
    """Count user-land processes above HEAVY_CPU_PERCENT in ``ps -Ao pid,pcpu,command`` output."""
    count = 0
    own = own_pids or set()
    for line in ps_lines:
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
            cpu = float(parts[1])
        except ValueError:
            continue
        command = parts[2]
        if pid in own or cpu <= HEAVY_CPU_PERCENT or _OWN_PROCESS.search(command):
            continue
        if command.startswith(_SYSTEM_PATHS):
            continue
        count += 1
    return count


class LoadSampler:
    """Sample the 1-minute load average and outside heavy processes while a pair runs."""

    def __init__(self, interval_s: float = 2.0) -> None:
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self) -> dict:
        try:
            lines = subprocess.run(
                ["ps", "-Ao", "pid,pcpu,command", "-r"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.splitlines()[1:]
        except OSError:
            lines = []
        return {
            "t": time.time(),
            "load1": os.getloadavg()[0],
            "heavy_other": heavy_other_processes(lines, {os.getpid()}),
        }

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self._sample())
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join()
        self.samples.append(self._sample())
        return {
            "samples": len(self.samples),
            "load1_max": max(s["load1"] for s in self.samples),
            "load1_mean": sum(s["load1"] for s in self.samples) / len(self.samples),
            "heavy_other_max": max(s["heavy_other"] for s in self.samples),
            "heavy_other_samples": sum(1 for s in self.samples if s["heavy_other"]),
        }


def measure_bucket(
    servers: dict[str, Server],
    order: list[str],
    prompts: list[list[int]],
    output: int,
    repetitions: int,
    slo: dict,
    *,
    max_load: float = 4.0,
    max_retries: int = 5,
    sampler_factory: Callable[[], LoadSampler] = LoadSampler,
) -> tuple[dict[str, list[dict]], list[dict]]:
    """Warm each server once, then measure paired repetitions with alternating order.

    A pair (one run of each server) during which another process ran above
    HEAVY_CPU_PERCENT or the 1-minute load average exceeded ``max_load`` is
    outside load, not a measurement: it is discarded and the pair repeated, up
    to ``max_retries`` extra pairs per bucket; when the retries are spent the
    pair is kept and flagged. Every attempt is returned with its load record.
    """
    for name in order:
        asyncio.run(run_repetition(servers[name].base, prompts, output, slo))
    measured: dict[str, list[dict]] = {name: [] for name in order}
    attempts: list[dict] = []
    retries = 0
    while len(measured[order[0]]) < repetitions:
        kept = len(measured[order[0]])
        sequence = order if kept % 2 == 0 else list(reversed(order))
        sampler = sampler_factory()
        sampler.start()
        results = {
            name: asyncio.run(run_repetition(servers[name].base, prompts, output, slo))
            for name in sequence
        }
        load = sampler.stop()
        contaminated = load["heavy_other_max"] > 0 or load["load1_max"] > max_load
        discard = contaminated and retries < max_retries
        attempts.append(
            {
                "order": sequence,
                "load": load,
                "contaminated": contaminated,
                "kept": not discard,
            }
        )
        if discard:
            retries += 1
            print(
                f"  discarded a pair under outside load (heavy processes "
                f"{load['heavy_other_max']}, load1 max {load['load1_max']:.1f}); "
                f"retry {retries}/{max_retries}",
                flush=True,
            )
            continue
        for name in order:
            measured[name].append(
                dict(results[name], load=load, contaminated=contaminated)
            )
    return measured, attempts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--widths", default="1,2,4,7")
    parser.add_argument("--buckets", default="128x128,1024x128,128x512")
    parser.add_argument("--concurrency", default="1,4")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--max-load",
        type=float,
        default=4.0,
        help="a paired repetition during which the 1-minute load average exceeded this "
        "(or another user-land process ran above 40%% CPU) is discarded and repeated",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="extra paired repetitions allowed per bucket to replace contaminated ones",
    )
    parser.add_argument("--slo-ttft", type=float, default=1.0)
    parser.add_argument("--slo-gap-p95", type=float, default=0.1)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--batch-tokens", type=int, default=512)
    parser.add_argument("--memory-fraction", default="0.2")
    parser.add_argument("--async-reference", action="store_true")
    parser.add_argument(
        "--adaptive-calibration",
        type=Path,
        help="run every speculative server in the adaptive mode with this calibration artifact",
    )
    parser.add_argument("--adaptive-cost-model", type=Path)
    parser.add_argument("--startup-timeout", type=float, default=900)
    args = parser.parse_args()
    if not args.target.is_dir() or not args.draft.is_dir():
        parser.error("target and draft must be already downloaded snapshots")
    if args.repetitions < 5:
        parser.error("the protocol needs at least five paired repetitions")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.target))
    buckets = [
        tuple(int(v) for v in item.split("x"))
        for item in args.buckets.split(",")
        if item
    ]
    concurrencies = [int(v) for v in args.concurrency.split(",") if v]
    widths = [int(v) for v in args.widths.split(",") if v]
    slo = {"ttft_s": args.slo_ttft, "gap_p95_s": args.slo_gap_p95}
    for input_length, output in buckets:
        if input_length + output + 8 > args.max_model_len:
            parser.error(f"bucket {input_length}x{output} exceeds --max-model-len")

    server_args = argparse.Namespace(
        target=args.target,
        draft=args.draft,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        batch_tokens=args.batch_tokens,
        memory_fraction=args.memory_fraction,
    )

    adaptive_env: dict[str, str] | None = None
    if args.adaptive_calibration or args.adaptive_cost_model:
        if not (args.adaptive_calibration and args.adaptive_cost_model):
            parser.error("--adaptive-calibration and --adaptive-cost-model go together")
        adaptive_env = {
            "VLLM_METAL_DSPARK_MODE": "adaptive",
            "VLLM_METAL_DSPARK_CALIBRATION": str(args.adaptive_calibration.resolve()),
            "VLLM_METAL_DSPARK_COST_MODEL": str(args.adaptive_cost_model.resolve()),
        }

    def start(
        width: int, *, name: str | None = None, async_scheduling: bool = False
    ) -> Server:
        env = adaptive_env if width and adaptive_env else None
        server = Server(
            server_args,
            width,
            False,
            args.output_dir,
            name=name or (f"k{width}-adaptive" if env else None),
            async_scheduling=async_scheduling,
            env=env,
            expect_in_log="mode=adaptive" if env else None,
        )
        server.wait_ready(args.startup_timeout)
        return server

    report: dict = {
        "protocol": {
            "adaptive": adaptive_env is not None,
            "buckets": [f"{i}x{o}" for i, o in buckets],
            "concurrency": concurrencies,
            "repetitions": args.repetitions,
            "slo": slo,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "batch_tokens": args.batch_tokens,
            "memory_fraction": args.memory_fraction,
            "gate": "median relative benefit >= 10% with bootstrap 95% CI of the median above 0",
        },
        "results": {},
    }
    reference = start(0)
    try:
        for width in widths:
            candidate = start(width)
            try:
                for input_length, output in buckets:
                    for concurrency in concurrencies:
                        prompts = build_prompts(
                            tokenizer, args.repo, input_length, concurrency
                        )
                        key = f"{candidate.name}/{input_length}x{output}/c{concurrency}"
                        before = metrics(candidate.base)
                        measured, attempts = measure_bucket(
                            {"k0": reference, "k": candidate},
                            ["k0", "k"],
                            prompts,
                            output,
                            args.repetitions,
                            slo,
                            max_load=args.max_load,
                            max_retries=args.max_retries,
                        )
                        after = metrics(candidate.base)
                        entry = {
                            "reference_reps": measured["k0"],
                            "candidate_reps": measured["k"],
                            "attempts": attempts,
                            "paired": paired_summary(measured["k0"], measured["k"]),
                            "spec_metrics_delta": {
                                name: after.get(name, 0.0) - before.get(name, 0.0)
                                for name in (
                                    "vllm:spec_decode_num_draft_tokens_total",
                                    "vllm:spec_decode_num_accepted_tokens_total",
                                )
                            },
                        }
                        report["results"][key] = entry
                        paired = entry["paired"]
                        print(
                            f"{key}: tok/s {paired['output_tokens_per_s']['reference_median']:.1f} -> "
                            f"{paired['output_tokens_per_s']['candidate_median']:.1f} "
                            f"({paired['output_tokens_per_s']['median_relative_benefit'] * 100:+.1f}%, "
                            f"CI {[round(v * 100, 1) for v in paired['output_tokens_per_s']['benefit_ci95']]}); "
                            f"TPOT {paired['tpot_median_s']['reference_median'] * 1000:.1f} -> "
                            f"{paired['tpot_median_s']['candidate_median'] * 1000:.1f} ms; "
                            f"gap p95 {paired['gap_p95_s']['reference_median'] * 1000:.0f} -> "
                            f"{paired['gap_p95_s']['candidate_median'] * 1000:.0f} ms; "
                            f"accepted/drafted {entry['spec_metrics_delta']['vllm:spec_decode_num_accepted_tokens_total']:.0f}/"
                            f"{entry['spec_metrics_delta']['vllm:spec_decode_num_draft_tokens_total']:.0f}",
                            flush=True,
                        )
                        (args.output_dir / "results.json").write_text(
                            json.dumps(report, indent=2) + "\n"
                        )
            finally:
                candidate.stop()
        if args.async_reference:
            asynchronous = start(0, name="k0-async", async_scheduling=True)
            try:
                for input_length, output in buckets:
                    for concurrency in concurrencies:
                        prompts = build_prompts(
                            tokenizer, args.repo, input_length, concurrency
                        )
                        key = f"k0-async/{input_length}x{output}/c{concurrency}"
                        measured, attempts = measure_bucket(
                            {"k0": reference, "async": asynchronous},
                            ["k0", "async"],
                            prompts,
                            output,
                            args.repetitions,
                            slo,
                            max_load=args.max_load,
                            max_retries=args.max_retries,
                        )
                        report["results"][key] = {
                            "reference_reps": measured["k0"],
                            "candidate_reps": measured["async"],
                            "attempts": attempts,
                            "paired": paired_summary(measured["k0"], measured["async"]),
                        }
                        (args.output_dir / "results.json").write_text(
                            json.dumps(report, indent=2) + "\n"
                        )
            finally:
                asynchronous.stop()
    finally:
        reference.stop()
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print("done")


if __name__ == "__main__":
    main()
