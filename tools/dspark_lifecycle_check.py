# SPDX-License-Identifier: Apache-2.0
"""Bounded DSpark lifecycle/greedy check using local Qwen3-4B snapshots.

Each baseline/speculative engine runs in a fresh subprocess. This validates
actual proposals and verification, chunked/mixed prefill, repeated prefix-cache
requests and physical context coverage. It is not a production speed benchmark.
"""

from __future__ import annotations

import argparse
import collections
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROMPTS = [
    "The capital of France is",
    "Explain the following algorithm clearly. "
    + "A sorted list can be searched by repeatedly halving the remaining interval. "
    * 7,
    "Write a Python function that adds two numbers.",
    "Explain why leaves are green. "
    + "Plants absorb light and convert its energy into chemical energy. " * 5,
]


def worker(config: dict, output_path: Path) -> None:
    import mlx.core as mx
    from vllm import LLM, SamplingParams

    from vllm_metal.v1.dspark_proposer import DSparkProposer

    width = config["width"]
    llm = LLM(
        model=config["target"],
        max_model_len=256,
        max_num_seqs=config["concurrency"],
        max_num_batched_tokens=32,
        enable_chunked_prefill=True,
        enable_prefix_caching=config["prefix_cache"],
        async_scheduling=False,
        disable_log_stats=False,
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
    counts = collections.Counter()
    fallback_requests = {}
    if width:
        proposer = runner._drafter
        assert isinstance(proposer, DSparkProposer)
        real_propose, real_verify, real_release = (
            proposer.propose,
            runner._spec_decode_controller.verify_greedy,
            proposer.release_requests,
        )

        def propose(ctx):
            result = real_propose(ctx)
            counts["steps"] += 1
            counts["intermediate_prefill_spans"] += sum(
                pr.prompt_len is None for pr in ctx.prefill_reqs
            )
            counts["mixed_steps"] += bool(ctx.decode_reqs and ctx.prefill_reqs)
            for req_id, record in proposer._contexts.items():
                if record.disabled_reason:
                    fallback_requests[req_id] = record.disabled_reason
                    assert not record.caches
                else:
                    assert all(
                        cache.length == record.covered_end for cache in record.caches
                    )
            if result:
                counts["proposed"] += sum(map(len, result.draft_token_ids))
                for req_id in result.req_ids:
                    record = proposer._contexts[req_id]
                    assert record.covered_end == len(record.owner.token_ids) - 1
            return result

        def verify(logits, decode_reqs, decode_segments):
            result = real_verify(logits, decode_reqs, decode_segments)
            for segment, tokens in zip(decode_segments, result, strict=True):
                counts["verified"] += len(segment.draft_token_ids)
                counts["accepted"] += max(0, len(tokens) - 1)
            return result

        def release(req_ids):
            real_release(req_ids)
            assert not req_ids.intersection(proposer._contexts)
            counts["released_requests"] += len(req_ids)

        proposer.propose = propose
        proposer.release_requests = release
        runner._spec_decode_controller.verify_greedy = verify
    rounds = []
    params = SamplingParams(temperature=0.0, max_tokens=24, ignore_eos=True)
    for _ in range(2):
        start = time.perf_counter()
        outputs = llm.generate(PROMPTS, params, use_tqdm=False)
        rounds.append(
            {
                "tokens": [list(output.outputs[0].token_ids) for output in outputs],
                "elapsed_s": time.perf_counter() - start,
            }
        )
    # Offline LLM.generate stops when user-visible outputs finish. Drain the
    # scheduler's remaining finished-ID notification through the normal engine
    # step, exactly as the continuously running server loop does.
    engine = llm.llm_engine
    engine.step()
    lifecycle = {"completion_drained": True}
    if width:
        assert not proposer._contexts
    # Cancel a real request after its first scheduled chunk, then reuse its
    # public ID. No direct proposer cleanup is allowed in this check.
    internal_id = engine.add_request("dspark-cancel-check", PROMPTS[1], params)
    engine.step()
    if width:
        assert internal_id in proposer._contexts
    engine.abort_request(["dspark-cancel-check"])
    engine.step()
    if width:
        assert not proposer._contexts
    engine.add_request("dspark-cancel-check", PROMPTS[0], params)
    while engine.has_unfinished_requests():
        engine.step()
    engine.step()
    if width:
        assert not proposer._contexts
    lifecycle["cancel_and_reuse_drained"] = True
    result = {
        "config": config,
        "rounds": rounds,
        "counts": dict(counts),
        "fallback_requests": fallback_requests,
        "peak_mlx_bytes": mx.get_peak_memory(),
        "lifecycle": lifecycle,
        "runtime": {
            package: importlib.metadata.version(package)
            for package in ("mlx", "mlx-lm", "vllm", "torch")
        },
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--concurrency", type=int, choices=(1, 4), default=4)
    parser.add_argument("--width", type=int, choices=range(1, 8), default=2)
    parser.add_argument("--prefix-cache", action="store_true")
    parser.add_argument("--memory-fraction", default="0.22")
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_config:
        data = json.loads(args.worker_config.read_text())
        worker(data, args.worker_config.with_suffix(".result.json"))
        return
    if any(path is None for path in (args.target, args.draft, args.output_dir)):
        parser.error("--target, --draft and --output-dir are required")
    if not args.target.is_dir() or not args.draft.is_dir():
        parser.error("pinned target and draft snapshots must already exist locally")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
        VLLM_METAL_USE_PAGED_ATTENTION="1",
        VLLM_METAL_MEMORY_FRACTION=args.memory_fraction,
        VLLM_METAL_BUILD_FROM_SOURCE="1",
        HF_HUB_OFFLINE="1",
    )
    results = []
    for width in (0, args.width):
        config = {
            "target": str(args.target.resolve()),
            "draft": str(args.draft.resolve()),
            "width": width,
            "concurrency": args.concurrency,
            "prefix_cache": args.prefix_cache,
            "memory_fraction": args.memory_fraction,
        }
        path = args.output_dir / f"k{width}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        with path.with_suffix(".log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.dspark_lifecycle_check",
                    "--worker-config",
                    str(path),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=600,
            )
        results.append(json.loads(path.with_suffix(".result.json").read_text()))
        print(f"Completed K={width}; results in {args.output_dir}", flush=True)
    baseline, speculative = results
    for base_round, spec_round in zip(
        baseline["rounds"], speculative["rounds"], strict=True
    ):
        assert len(base_round["tokens"]) == len(spec_round["tokens"]) == len(PROMPTS)
        assert all(len(tokens) == 24 for tokens in spec_round["tokens"])
        assert base_round["tokens"] == spec_round["tokens"], "greedy output mismatch"
    for counter in ("proposed", "verified", "accepted", "intermediate_prefill_spans"):
        assert speculative["counts"].get(counter, 0) > 0, (
            f"missing exercised path: {counter}"
        )
    if args.prefix_cache:
        assert (
            "missing target feature prefix" in speculative["fallback_requests"].values()
        )
    print("PASS: greedy identity and exercised DSpark lifecycle checks", flush=True)


if __name__ == "__main__":
    main()
