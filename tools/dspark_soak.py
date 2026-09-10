# SPDX-License-Identifier: Apache-2.0
"""HTTP soak for DSpark serving: mixed arrivals for a duration and a request count.

Launches one ``vllm serve`` (any DSpark mode through the environment) and
drives it with a closed-loop client pool for at least ``--duration`` seconds
and at least ``--requests`` completed or cancelled requests, whichever is
longer. The mix is drawn from a seeded generator per request: greedy and
sampled requests (temperature 0.8, some seeded), output budgets from 8 to
256 tokens, natural prompts and long documentation windows (several prefill
chunks), a shared prefix on a fraction of requests, streaming and
non-streaming calls, and a fraction of streamed requests the client abandons
mid-way (cancellations). Every request records its outcome, token count,
TTFT and end-to-end latency; the tool samples the server's resident memory
and the ``/metrics`` counters on a fixed cadence, and asserts at the end that
the server still answers, holds no running or waiting request, reported
draft work throughout, and that no request failed. The summary carries the
latency percentiles, throughput, cancellation count, the memory trajectory
(first, peak, last) and the spec-decode counters.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import subprocess
import time
from pathlib import Path

import httpx

from tools.dspark_serving_check import Server, metrics


def rss_bytes(pid: int) -> int | None:
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return int(out) * 1024 if out else None
    except (subprocess.CalledProcessError, ValueError):
        return None


def process_tree_rss(pid: int) -> int:
    """Resident memory of the server process and its children (engine core)."""
    total = 0
    try:
        out = subprocess.run(
            ["pgrep", "-P", str(pid)], capture_output=True, text=True
        ).stdout.split()
    except OSError:
        out = []
    for item in [pid, *(int(v) for v in out)]:
        value = rss_bytes(item)
        if value:
            total += value
    return total


def build_prompts(tokenizer, repo: Path) -> dict[str, list[list[int]]]:
    from tools.dspark_memory_check import NATURAL_PROMPTS
    from tools.dspark_perf_bench import FILLER_DOCS

    natural = [
        tokenizer.encode(text, add_special_tokens=False) for text in NATURAL_PROMPTS
    ]
    text = "\n\n".join(
        (repo / name).read_text(encoding="utf-8") for name in FILLER_DOCS
    )
    filler = tokenizer.encode(text, add_special_tokens=False)
    long_windows = [
        filler[start : start + 700] for start in range(0, len(filler) - 700, 1500)
    ][:8]
    shared = filler[:256]
    return {"natural": natural, "long": long_windows, "shared": [shared]}


def make_request(rng: random.Random, prompts: dict, index: int) -> dict:
    kind = rng.random()
    if kind < 0.6:
        prompt = rng.choice(prompts["natural"])
    elif kind < 0.8:
        prompt = rng.choice(prompts["long"])
    else:
        prompt = prompts["shared"][0] + rng.choice(prompts["natural"])
    body: dict = {
        "model": "target",
        "prompt": prompt,
        "max_tokens": rng.choice([8, 16, 32, 64, 128, 256]),
        "return_token_ids": True,
    }
    sampled = rng.random() < 0.4
    body["temperature"] = 0.8 if sampled else 0.0
    if sampled and rng.random() < 0.5:
        body["seed"] = index
        body["top_p"] = 0.95
    stream = rng.random() < 0.5
    cancel_after = None
    if stream and rng.random() < 0.15:
        cancel_after = rng.choice([1, 2, 4])
    return {"body": body, "stream": stream, "cancel_after": cancel_after}


async def run_one(client: httpx.AsyncClient, base: str, spec: dict) -> dict:
    body = dict(spec["body"])
    sent = time.perf_counter()
    result = {"stream": spec["stream"], "cancelled": False, "tokens": 0, "error": None}
    try:
        if spec["stream"]:
            body["stream"] = True
            first = None
            chunks = 0
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
                    if first is None:
                        first = time.perf_counter()
                    choice = json.loads(payload)["choices"][0]
                    result["tokens"] += len(choice.get("token_ids") or [])
                    chunks += 1
                    if (
                        spec["cancel_after"] is not None
                        and chunks >= spec["cancel_after"]
                    ):
                        result["cancelled"] = True
                        break
            result["ttft"] = (first - sent) if first else None
        else:
            response = await client.post(f"{base}/v1/completions", json=body)
            response.raise_for_status()
            choice = response.json()["choices"][0]
            result["tokens"] = len(choice.get("token_ids") or [])
            result["ttft"] = None
    except Exception as error:  # noqa: BLE001 - every failure is a soak finding
        result["error"] = f"{type(error).__name__}: {error}"[:200]
    result["e2e"] = time.perf_counter() - sent
    return result


async def soak(
    base: str, prompts: dict, args, process: subprocess.Popen, report: dict
) -> None:
    rng = random.Random(args.seed)
    started = time.perf_counter()
    results: list[dict] = []
    samples: list[dict] = []
    counter = 0
    lock = asyncio.Lock()
    server_dead = asyncio.Event()
    pid = process.pid

    async def worker() -> None:
        nonlocal counter
        async with httpx.AsyncClient(timeout=600) as client:
            while not server_dead.is_set():
                async with lock:
                    elapsed = time.perf_counter() - started
                    if elapsed >= args.duration and len(results) >= args.requests:
                        return
                    counter += 1
                    index = counter
                spec = make_request(rng, prompts, index)
                results.append(await run_one(client, base, spec))

    async def sampler() -> None:
        while True:
            # A dead server ends the soak at once, as a failure with the exit
            # code and the moment it happened, instead of a duration's worth
            # of connection errors.
            if process.poll() is not None:
                report["server_exit"] = {
                    "t": time.perf_counter() - started,
                    "returncode": process.returncode,
                }
                server_dead.set()
                return
            state: dict[str, float] = {}
            try:
                state = metrics(base)
            except httpx.HTTPError as error:
                # Sampled state is best effort; a dead server is caught above.
                report.setdefault("sampler_errors", []).append(
                    f"{type(error).__name__}: {error}"[:200]
                )
            samples.append(
                {
                    "t": time.perf_counter() - started,
                    "rss": process_tree_rss(pid),
                    "running": state.get("vllm:num_requests_running", 0.0),
                    "waiting": state.get("vllm:num_requests_waiting", 0.0),
                    "drafts": state.get("vllm:spec_decode_num_draft_tokens_total", 0.0),
                    "accepted": state.get(
                        "vllm:spec_decode_num_accepted_tokens_total", 0.0
                    ),
                    "completed": len(results),
                }
            )
            await asyncio.sleep(args.sample_every)

    task = asyncio.create_task(sampler())
    await asyncio.gather(*(worker() for _ in range(args.concurrency)))
    task.cancel()
    # Close the client sockets of a server that died mid-request.
    await asyncio.sleep(0)
    report["results"] = results
    report["samples"] = samples
    report["elapsed_s"] = time.perf_counter() - started


def summarize(report: dict, base: str) -> tuple[dict, list[str]]:
    results = report["results"]
    failures = []
    errors = [r for r in results if r["error"]]
    cancelled = sum(1 for r in results if r["cancelled"])
    e2e = sorted(r["e2e"] for r in results if not r["error"] and not r["cancelled"])
    ttft = sorted(r["ttft"] for r in results if r.get("ttft"))
    tokens = sum(r["tokens"] for r in results)
    samples = report["samples"]
    rss = [s["rss"] for s in samples if s["rss"]]
    exit_info = report.get("server_exit")
    if exit_info is not None:
        failures.append(
            f"server process exited with code {exit_info['returncode']} "
            f"after {exit_info['t']:.0f} s"
        )
        final: dict[str, float] = {}
    else:
        final = metrics(base)
    if errors:
        failures.append(f"{len(errors)} requests failed, first: {errors[0]['error']}")
    if final.get("vllm:num_requests_running", 0) or final.get(
        "vllm:num_requests_waiting", 0
    ):
        failures.append("server still has running or waiting requests after the soak")
    drafts = final.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    accepted = final.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    if drafts <= 0 or accepted <= 0:
        failures.append("no draft work reported during the soak")
    # Draft work must keep flowing: the last quarter of samples adds drafts.
    if len(samples) >= 8:
        quarter = samples[-len(samples) // 4]
        if samples[-1]["drafts"] <= quarter["drafts"]:
            failures.append("draft work stalled in the last quarter of the soak")

    def pct(values, q):
        return values[min(len(values) - 1, int(q * len(values)))] if values else None

    summary = {
        "requests": len(results),
        "completed": len(results) - cancelled - len(errors),
        "cancelled": cancelled,
        "errors": len(errors),
        "output_tokens": tokens,
        "elapsed_s": report["elapsed_s"],
        "requests_per_s": len(results) / report["elapsed_s"],
        "output_tokens_per_s": tokens / report["elapsed_s"],
        "e2e_p50_s": pct(e2e, 0.5),
        "e2e_p95_s": pct(e2e, 0.95),
        "e2e_p99_s": pct(e2e, 0.99),
        "ttft_p50_s": pct(ttft, 0.5),
        "ttft_p95_s": pct(ttft, 0.95),
        "rss_first_bytes": rss[0] if rss else None,
        "rss_peak_bytes": max(rss) if rss else None,
        "rss_last_bytes": rss[-1] if rss else None,
        "draft_tokens": drafts,
        "accepted_tokens": accepted,
        "acceptance": accepted / drafts if drafts else None,
        "samples": len(samples),
    }
    return summary, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--width", type=int, default=7)
    parser.add_argument("--duration", type=float, default=3600.0)
    parser.add_argument("--requests", type=int, default=10_000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--sample-every", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--batch-tokens", type=int, default=512)
    parser.add_argument("--memory-fraction", default="0.2")
    parser.add_argument("--startup-timeout", type=float, default=900)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.target))
    prompts = build_prompts(tokenizer, args.repo)
    server_args = argparse.Namespace(
        target=args.target,
        draft=args.draft,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        batch_tokens=args.batch_tokens,
        memory_fraction=args.memory_fraction,
    )
    server = Server(
        server_args, args.width, False, args.output_dir, name=f"k{args.width}-soak"
    )
    report: dict = {
        "config": vars(args)
        | {
            "output_dir": str(args.output_dir),
            "target": str(args.target),
            "draft": str(args.draft),
            "repo": str(args.repo),
        }
    }
    try:
        server.wait_ready(args.startup_timeout)
        report["mode_env"] = {
            k: v for k, v in os.environ.items() if k.startswith("VLLM_METAL_DSPARK")
        }
        asyncio.run(soak(server.base, prompts, args, server.process, report))
        summary, failures = summarize(report, server.base)
    finally:
        server.stop()
    summary["failures"] = failures
    (args.output_dir / "soak.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for key, value in summary.items():
        if key != "failures":
            print(f"{key}: {value}")
    if failures:
        print("FAILURES:")
        for failure in failures:
            print(" -", failure)
        raise SystemExit(1)
    print("SOAK PASS")


if __name__ == "__main__":
    main()
