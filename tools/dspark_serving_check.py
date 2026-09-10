# SPDX-License-Identifier: Apache-2.0
"""HTTP serving qualification for DSpark on the OpenAI-compatible server.

A target-only ``vllm serve`` (K=0) establishes reference outputs; then one
server per speculative width runs the same request matrix through the real
HTTP path with vLLM's multiprocess engine core: output limits 1/2/31/128,
natural EOS, stop strings, the platform's ``min_tokens`` rejection, streaming
versus non-streaming,
staggered concurrent arrivals, a client disconnect mid-stream, ``logprobs``
and non-greedy requests (target-only fallback), long prompts that need
several prefill chunks, and, on servers with prefix caching, repeated and
shared prefixes. Servers with prefix caching are optional (``--prefix-widths``).

Parity uses the M4a contract at the HTTP level: a speculative stream must
equal the K=0 stream, or diverge at a position where the K=0 server's own
top logprobs put both tokens within ``TIE_LOGPROB_GAP`` of each other (a
logprob difference equals the logit difference, so 0.5 covers two bfloat16
ULPs up to logit magnitude 64). Anything else fails. Target-instability, the
other admissible class, needs the in-process tools; an HTTP divergence that
is not a tie is reported for them, never waived here.

Every speculative server must show draft and accepted tokens in ``/metrics``
after the matrix, and no running or waiting request after the disconnect.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

TIE_LOGPROB_GAP = 0.5
OUTPUT_LIMITS = (1, 2, 31, 128)
SPEC_METRICS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Server:
    """One ``vllm serve`` process with health wait and clean shutdown."""

    def __init__(
        self,
        args,
        width: int,
        prefix_caching: bool,
        log_dir: Path,
        *,
        name: str | None = None,
        async_scheduling: bool = False,
        extra_args: tuple[str, ...] = (),
    ):
        self.width = width
        self.prefix_caching = prefix_caching
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        name = name or f"k{width}{'-prefix' if prefix_caching else ''}"
        self.name = name
        self.log_path = log_dir / f"server-{name}.log"
        executable = shutil.which("vllm", path=str(Path(sys.executable).parent))
        command = [
            executable or "vllm",
            "serve",
            str(args.target),
            "--port",
            str(self.port),
            "--served-model-name",
            "target",
            "--max-model-len",
            str(args.max_model_len),
            "--max-num-seqs",
            str(getattr(args, "max_num_seqs", 4)),
            "--max-num-batched-tokens",
            str(args.batch_tokens),
            "--async-scheduling" if async_scheduling else "--no-async-scheduling",
            "--seed",
            "0",
            "--enable-prefix-caching"
            if prefix_caching
            else "--no-enable-prefix-caching",
            *extra_args,
        ]
        if width:
            command += [
                "--speculative-config",
                json.dumps(
                    {
                        "method": "dspark",
                        "model": str(args.draft),
                        "num_speculative_tokens": width,
                    }
                ),
            ]
        self.command = command
        env = dict(
            os.environ,
            VLLM_USE_V2_MODEL_RUNNER="0",
            VLLM_METAL_USE_PAGED_ATTENTION="1",
            VLLM_METAL_MEMORY_FRACTION=str(args.memory_fraction),
            VLLM_METAL_BUILD_FROM_SOURCE="1",
            HF_HUB_OFFLINE="1",
            GLOO_SOCKET_IFNAME="lo0",
        )
        self.log = self.log_path.open("w")
        self.process = subprocess.Popen(
            command,
            env=env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        with httpx.Client(timeout=5) as client:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"{self.name} exited with {self.process.returncode}; see {self.log_path}"
                    )
                try:
                    if client.get(f"{self.base}/health").status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(1)
        raise RuntimeError(f"{self.name} did not become healthy; see {self.log_path}")

    def stop(self) -> None:
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=30)
        self.log.close()


def parse_metrics(text: str) -> dict[str, float]:
    """Sum every sample of each Prometheus metric name, ignoring labels."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line:
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        try:
            value = float(line.rsplit(" ", 1)[1])
        except ValueError:
            continue
        values[name] = values.get(name, 0.0) + value
    return values


def metrics(base: str) -> dict[str, float]:
    return parse_metrics(httpx.get(f"{base}/metrics", timeout=30).text)


def token_logprobs(entry: dict) -> dict[int, float]:
    """``token_id:N`` keys of one ``top_logprobs`` entry as ``{id: logprob}``."""
    return {
        int(key.split(":", 1)[1]): float(value)
        for key, value in entry.items()
        if key.startswith("token_id:")
    }


def complete(client: httpx.Client, base: str, prompt: list[int], **params) -> dict:
    body = {
        "model": "target",
        "prompt": prompt,
        "temperature": 0.0,
        "return_token_ids": True,
    }
    body.update(params)
    response = client.post(f"{base}/v1/completions", json=body)
    if response.status_code != 200:
        raise RuntimeError(
            f"{response.status_code} for {json.dumps({k: v for k, v in body.items() if k != 'prompt'})}: "
            f"{response.text[:500]}"
        )
    choice = response.json()["choices"][0]
    return {
        "token_ids": list(choice.get("token_ids") or []),
        "finish_reason": choice.get("finish_reason"),
        "logprobs": choice.get("logprobs"),
    }


def stream_complete(
    client: httpx.Client,
    base: str,
    prompt: list[int],
    stop_after: int | None = None,
    **params,
) -> dict:
    body = {
        "model": "target",
        "prompt": prompt,
        "temperature": 0.0,
        "return_token_ids": True,
        "stream": True,
    }
    body.update(params)
    chunks: list[list[int]] = []
    finish = None
    with client.stream("POST", f"{base}/v1/completions", json=body) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            choice = json.loads(payload)["choices"][0]
            chunks.append(list(choice.get("token_ids") or []))
            finish = choice.get("finish_reason") or finish
            if stop_after is not None and len(chunks) >= stop_after:
                break  # closing the stream mid-way is the disconnect
    return {"chunks": chunks, "finish_reason": finish}


def first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def tie_probe(
    client: httpx.Client, baseline: str, prefix: list[int], a: int, b: int
) -> dict:
    """The K=0 server's own view of the two tokens at the divergent prefix."""
    result = complete(
        client,
        baseline,
        prefix,
        max_tokens=1,
        logprobs=5,
        return_tokens_as_token_ids=True,
        ignore_eos=True,
    )
    top = (result["logprobs"] or {}).get("top_logprobs") or [{}]
    entries = token_logprobs(top[0])
    verdict: dict = {"top": entries, "tokens": [a, b], "gap": None, "tie": False}
    if a in entries and b in entries:
        gap = abs(entries[a] - entries[b])
        verdict["gap"] = gap
        verdict["tie"] = gap <= TIE_LOGPROB_GAP
    return verdict


def compare(client, baseline_base, prompt, expected, actual) -> dict:
    """Equal, or a tie at the first divergence per the K=0 server's logprobs."""
    position = first_divergence(expected, actual)
    if position is None:
        return {"equal": True}
    if position >= min(len(expected), len(actual)):
        return {
            "equal": False,
            "position": position,
            "label": "length",
            "admissible": False,
        }
    probe = tie_probe(
        client,
        baseline_base,
        prompt + expected[:position],
        expected[position],
        actual[position],
    )
    return {
        "equal": False,
        "position": position,
        "expected": expected[position],
        "actual": actual[position],
        "probe": probe,
        "label": "tie" if probe["tie"] else "divergence",
        "admissible": probe["tie"],
    }


def build_matrix(tokenizer, batch_tokens: int) -> dict:
    from tools.dspark_memory_check import NATURAL_PROMPTS

    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)  # noqa: E731
    prompts = [encode(text) for text in NATURAL_PROMPTS]
    long_prompt = encode(" ".join(NATURAL_PROMPTS * 2))
    assert len(long_prompt) > 3 * batch_tokens, "long prompt must need several chunks"
    return {"prompts": prompts, "long": long_prompt}


def run_matrix(
    server: Server, baseline: Server | None, matrix: dict, results: dict
) -> dict:
    """Run every scenario against ``server``; compare with ``baseline`` when given."""
    base = server.base
    client = httpx.Client(timeout=900)
    report: dict = {"scenarios": {}, "failures": []}
    reference = results.get("k0-prefix" if server.prefix_caching else "k0", {})
    baseline_base = baseline.base if baseline else None
    # A server compares its own batched output with its isolated output
    # against its own logprobs; the target-only server probes itself.
    probe_base = baseline_base or base

    def check(key: str, prompt: list[int], actual: list[int]) -> None:
        if baseline is None:
            return
        expected = reference.get(key)
        if expected is None:
            report["failures"].append(f"{key}: no baseline result")
            return
        verdict = compare(client, baseline_base, prompt, expected, actual)
        report["scenarios"][key]["parity"] = verdict
        if not verdict.get("equal") and not verdict.get("admissible"):
            report["failures"].append(f"{key}: {verdict}")

    def record(key: str, value: dict) -> None:
        report["scenarios"][key] = value
        results.setdefault(server.name, {})[key] = value.get(
            "token_ids", value.get("tokens")
        )

    before = metrics(base)
    prompts = matrix["prompts"]

    # 1. Output limits and natural EOS on four prompts.
    for index, prompt in enumerate(prompts[:4]):
        for limit in OUTPUT_LIMITS:
            key = f"limit-{index}-{limit}"
            out = complete(client, base, prompt, max_tokens=limit)
            record(key, out)
            if len(out["token_ids"]) > limit:
                report["failures"].append(
                    f"{key}: {len(out['token_ids'])} tokens exceed {limit}"
                )
            check(key, prompt, out["token_ids"])
        key = f"eos-{index}"
        out = complete(client, base, prompt, max_tokens=256)
        record(key, out)
        if out["finish_reason"] not in ("stop", "length"):
            report["failures"].append(f"{key}: finish_reason {out['finish_reason']}")
        check(key, prompt, out["token_ids"])

    # 2. min_tokens is a logits-processor control the Metal platform rejects on
    #    every server; the speculative server must reject it the same way.
    key = "min-tokens-rejected"
    try:
        complete(client, base, prompts[4], max_tokens=64, min_tokens=40)
    except RuntimeError as error:
        record(key, {"rejected": True, "error": str(error)[:240]})
        if "min_tokens" not in str(error):
            report["failures"].append(f"{key}: unexpected rejection: {error}")
    else:
        report["failures"].append(
            f"{key}: min_tokens accepted although the platform declares it unsupported"
        )
    key = "stop-string"
    out = complete(client, base, prompts[5], max_tokens=128, stop=["\n\n"])
    record(key, out)
    check(key, prompts[5], out["token_ids"])

    # 3. Streaming equals non-streaming.
    key = "stream"
    plain = complete(client, base, prompts[6], max_tokens=64)
    streamed = stream_complete(client, base, prompts[6], max_tokens=64)
    tokens = [token for chunk in streamed["chunks"] for token in chunk]
    record(key, {"token_ids": tokens, "chunks": len(streamed["chunks"])})
    if tokens != plain["token_ids"]:
        # The second request may hit the prefix cache and take a different
        # arithmetic path; that is a tie or a defect, judged like any other.
        verdict = compare(client, probe_base, prompts[6], plain["token_ids"], tokens)
        report["scenarios"][key]["stream_vs_plain"] = verdict
        if not verdict.get("admissible"):
            report["failures"].append(
                f"{key}: streamed tokens differ from non-streamed: {verdict}"
            )
    if len(streamed["chunks"]) < 2:
        report["failures"].append(f"{key}: only {len(streamed['chunks'])} chunk(s)")
    check(key, prompts[6], tokens)

    # 4. Long prompt (several prefill chunks).
    key = "long-prompt"
    out = complete(client, base, matrix["long"], max_tokens=64)
    record(key, out)
    check(key, matrix["long"], out["token_ids"])

    # 5. Staggered concurrent arrivals versus isolated results on this server.
    arrivals = [
        (prompts[0], 64, 0.0),
        (prompts[1], 128, 0.15),
        (prompts[2], 31, 0.4),
        (prompts[3], 96, 0.8),
    ]
    isolated = [
        complete(client, base, prompt, max_tokens=limit)["token_ids"]
        for prompt, limit, _ in arrivals
    ]

    async def arrive():
        async with httpx.AsyncClient(timeout=900) as aclient:

            async def one(prompt, limit, delay):
                await asyncio.sleep(delay)
                body = {
                    "model": "target",
                    "prompt": prompt,
                    "temperature": 0.0,
                    "max_tokens": limit,
                    "return_token_ids": True,
                }
                response = await aclient.post(f"{base}/v1/completions", json=body)
                response.raise_for_status()
                return list(response.json()["choices"][0].get("token_ids") or [])

            return await asyncio.gather(*(one(*item) for item in arrivals))

    concurrent = asyncio.run(arrive())
    for index, ((prompt, _limit, _), alone, together) in enumerate(
        zip(arrivals, isolated, concurrent, strict=True)
    ):
        key = f"arrival-{index}"
        record(key, {"token_ids": together, "isolated": alone})
        if together != alone:
            verdict = compare(client, probe_base, prompt, alone, together)
            report["scenarios"][key]["self_consistency"] = verdict
            if not verdict.get("admissible"):
                report["failures"].append(
                    f"{key}: concurrent result differs from isolated: {verdict}"
                )
        check(key, prompt, together)

    # 6. Client disconnect mid-stream, then the server must be idle and correct.
    key = "disconnect"
    partial = stream_complete(
        client, base, prompts[7], max_tokens=512, ignore_eos=True, stop_after=8
    )
    idle = False
    for _ in range(30):
        state = metrics(base)
        if (
            state.get("vllm:num_requests_running", 0) == 0
            and state.get("vllm:num_requests_waiting", 0) == 0
        ):
            idle = True
            break
        time.sleep(1)
    after = complete(client, base, prompts[7], max_tokens=32)
    record(
        key,
        {
            "chunks_before_disconnect": len(partial["chunks"]),
            "idle": idle,
            "token_ids": after["token_ids"],
        },
    )
    if not idle:
        report["failures"].append(
            f"{key}: requests still running or waiting after disconnect"
        )
    check(key, prompts[7], after["token_ids"])

    # 7. Unsupported sampling on the speculative path: must serve, greedy logprobs must match.
    key = "logprobs-greedy"
    out = complete(client, base, prompts[0], max_tokens=32, logprobs=1)
    record(key, out)
    check(key, prompts[0], out["token_ids"])
    key = "sampled"
    out = complete(
        client,
        base,
        prompts[1],
        max_tokens=32,
        temperature=0.8,
        seed=1,
        ignore_eos=True,
    )
    record(key, {"token_ids": out["token_ids"]})
    if len(out["token_ids"]) != 32:
        report["failures"].append(
            f"{key}: {len(out['token_ids'])} tokens instead of 32"
        )

    # 8. Prefix hits (prefix-caching servers only).
    if server.prefix_caching:
        long = matrix["long"]
        first = complete(client, base, long, max_tokens=48)
        second = complete(client, base, long, max_tokens=48)
        record("prefix-first", first)
        record("prefix-repeat", second)
        if second["token_ids"] != first["token_ids"]:
            verdict = compare(
                client, probe_base, long, first["token_ids"], second["token_ids"]
            )
            if not verdict.get("admissible"):
                report["failures"].append(
                    f"prefix-repeat: differs from first: {verdict}"
                )
        check("prefix-first", long, first["token_ids"])
        shared = long[: len(long) * 2 // 3] + prompts[7]
        out = complete(client, base, shared, max_tokens=48)
        record("prefix-shared", out)
        check("prefix-shared", shared, out["token_ids"])

    after_metrics = metrics(base)
    report["metrics"] = {
        name: after_metrics.get(name, 0.0) - before.get(name, 0.0)
        for name in SPEC_METRICS
    }
    if server.width:
        if report["metrics"]["vllm:spec_decode_num_draft_tokens_total"] <= 0:
            report["failures"].append("no draft tokens reported by /metrics")
        if report["metrics"]["vllm:spec_decode_num_accepted_tokens_total"] <= 0:
            report["failures"].append("no accepted tokens reported by /metrics")
    client.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--widths", default="1,2,4,7", help="speculative widths, comma separated"
    )
    parser.add_argument(
        "--prefix-widths",
        default="2,7",
        help="widths to repeat with prefix caching ('' to skip)",
    )
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--batch-tokens", type=int, default=64)
    parser.add_argument("--memory-fraction", default="0.22")
    parser.add_argument("--startup-timeout", type=float, default=900)
    args = parser.parse_args()
    if not args.target.is_dir() or not args.draft.is_dir():
        parser.error("target and draft must be already downloaded snapshots")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.target))
    matrix = build_matrix(tokenizer, args.batch_tokens)
    widths = [int(w) for w in args.widths.split(",") if w]
    prefix_widths = [int(w) for w in args.prefix_widths.split(",") if w]
    plan = [(0, False)] + [(w, False) for w in widths]
    if prefix_widths:
        plan += [(0, True)] + [(w, True) for w in prefix_widths]

    results: dict = {}
    reports: dict = {}
    baselines: dict[bool, Server] = {}
    failures = 0
    try:
        for width, prefix in plan:
            server = Server(args, width, prefix, args.output_dir)
            print(f"starting {server.name}: {' '.join(server.command)}", flush=True)
            server.wait_ready(args.startup_timeout)
            started = time.monotonic()
            baseline = baselines.get(prefix) if width else None
            report = run_matrix(server, baseline, matrix, results)
            report["elapsed_s"] = time.monotonic() - started
            report["command"] = server.command
            reports[server.name] = report
            failures += len(report["failures"])
            (args.output_dir / f"{server.name}.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(
                f"{server.name}: {len(report['failures'])} failure(s), metrics {report['metrics']}, "
                f"{report['elapsed_s']:.0f}s",
                flush=True,
            )
            for failure in report["failures"]:
                print("   FAIL", failure, flush=True)
            if width == 0:
                baselines[prefix] = server  # keep the baseline alive for tie probes
            else:
                server.stop()
    finally:
        for server in baselines.values():
            server.stop()
    summary = {
        "servers": list(reports),
        "failures": failures,
        "tie_logprob_gap": TIE_LOGPROB_GAP,
        "per_server": {
            name: {
                "failures": report["failures"],
                "metrics": report["metrics"],
                "parity": {
                    key: scenario.get("parity", {}).get("label", "equal")
                    for key, scenario in report["scenarios"].items()
                    if "parity" in scenario
                },
            }
            for name, report in reports.items()
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("GATE PASS" if failures == 0 else f"GATE FAIL: {failures} failure(s)")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
