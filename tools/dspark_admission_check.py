# SPDX-License-Identifier: Apache-2.0
"""Real-engine DSpark admission and per-step fairness qualification.

Distinct natural prompts run through one speculative engine with a chosen
``--max-num-seqs``, context cap (``VLLM_METAL_DSPARK_MAX_CONTEXTS``) and
per-step draft cap (``VLLM_METAL_DSPARK_MAX_DRAFTS_PER_STEP``). The check
records, per request, the steps it held a complete context, the steps it was
drafted, the tokens it proposed and any target-only fallback reason. Gates:

- every request that holds a context is drafted at least once;
- when the per-step cap binds, the drafted share of eligible steps is even
  across context holders (no starvation, no favored request);
- requests scheduled while every slot is held are counted as target-only,
  never hidden; concurrent contexts never exceed the cap, and when observed
  request concurrency stays within the cap no request is excluded.

This measures admission, not token parity or speed; every request must still
complete with the requested output length and the proposer must drain.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


def worker(config: dict, output: Path) -> None:
    import mlx.core as mx
    from vllm import LLM, SamplingParams

    from tools.dspark_memory_check import NATURAL_PROMPTS
    from vllm_metal.v1.dspark_proposer import DSparkProposer

    llm = LLM(
        model=config["target"],
        max_model_len=config["max_model_len"],
        max_num_seqs=config["max_num_seqs"],
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
    engine = llm.llm_engine
    runner = engine.model_executor.driver_worker.model_runner
    proposer = runner._drafter
    assert isinstance(proposer, DSparkProposer)
    prompts = [
        f"Request {index + 1}. {NATURAL_PROMPTS[index % len(NATURAL_PROMPTS)]}"
        for index in range(config["requests"])
    ]
    stats: dict[str, dict] = {}
    peak = {"concurrent_contexts": 0, "concurrent_requests": 0, "steps": 0}
    real_propose = proposer.propose

    def propose(ctx):
        result = real_propose(ctx)
        peak["steps"] += 1
        scheduled = {segment.req_id for segment in ctx.decode_segments}
        scheduled.update(prefill.req_id for prefill in ctx.prefill_reqs)
        peak["concurrent_requests"] = max(peak["concurrent_requests"], len(scheduled))
        for req_id in scheduled:
            entry = stats.setdefault(
                req_id,
                {
                    "context": False,
                    "eligible_steps": 0,
                    "drafted_steps": 0,
                    "proposed": 0,
                },
            )
            record = proposer._contexts.get(req_id)
            if record is None:
                continue
            if record.disabled_reason is not None:
                entry.setdefault("fallback", record.disabled_reason)
                continue
            if record.caches:
                entry["context"] = True
                if (
                    ctx.num_speculative_tokens > 0
                    and record.covered_end == len(record.owner.token_ids) - 1
                ):
                    entry["eligible_steps"] += 1
        if result:
            for req_id, row in zip(result.req_ids, result.draft_token_ids, strict=True):
                stats[req_id]["drafted_steps"] += 1
                stats[req_id]["proposed"] += len(row)
        live = sum(bool(record.caches) for record in proposer._contexts.values())
        peak["concurrent_contexts"] = max(peak["concurrent_contexts"], live)
        return result

    proposer.propose = propose
    params = SamplingParams(
        temperature=0.0, max_tokens=config["output_length"], ignore_eos=True
    )
    outputs = llm.generate(prompts, params, use_tqdm=False)
    assert len(outputs) == config["requests"]
    assert all(
        len(item.outputs[0].token_ids) == config["output_length"] for item in outputs
    )
    engine.step()
    assert not proposer._contexts, "draft context retained after drain"

    holders = {req: entry for req, entry in stats.items() if entry["context"]}
    shares = {
        req: entry["drafted_steps"] / entry["eligible_steps"]
        for req, entry in holders.items()
        if entry["eligible_steps"]
    }
    fallbacks = collections.Counter(
        entry["fallback"] for entry in stats.values() if entry.get("fallback")
    )
    cap_binds = proposer._max_drafts_per_step < peak["concurrent_contexts"]
    result = {
        "config": config,
        "requests": config["requests"],
        "context_holders": len(holders),
        "target_only_requests": config["requests"] - len(holders),
        "fallback_reasons": dict(fallbacks),
        "max_concurrent_contexts": peak["concurrent_contexts"],
        "max_concurrent_requests": peak["concurrent_requests"],
        "context_slots": proposer.memory_plan.max_contexts,
        "drafts_per_step_cap": proposer._max_drafts_per_step,
        "cap_binds": cap_binds,
        "steps": peak["steps"],
        "never_drafted_holders": sorted(
            req for req, entry in holders.items() if entry["drafted_steps"] == 0
        ),
        "drafted_share": {
            "min": min(shares.values()) if shares else None,
            "median": statistics.median(shares.values()) if shares else None,
            "max": max(shares.values()) if shares else None,
        },
        "proposed_total": sum(entry["proposed"] for entry in stats.values()),
        "per_request": stats,
        "peak_mlx_bytes": mx.get_peak_memory(),
        "memory_plan": {
            "max_contexts": proposer.memory_plan.max_contexts,
            "context_bytes": proposer.memory_plan.context_bytes,
            "reserve_bytes": proposer.memory_plan.reserve_bytes,
        },
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    assert not result["never_drafted_holders"], result["never_drafted_holders"]
    assert result["proposed_total"] > 0
    slots = proposer.memory_plan.max_contexts
    assert peak["concurrent_contexts"] <= slots
    if peak["concurrent_requests"] <= slots:
        assert result["target_only_requests"] == 0, result["fallback_reasons"]
    else:
        assert peak["concurrent_contexts"] == slots
        assert result["fallback_reasons"].get("context capacity exhausted", 0) >= 1
        assert len(holders) >= slots
    if cap_binds and shares:
        spread = max(shares.values()) - min(shares.values())
        assert spread <= 0.2, f"uneven drafted share across context holders: {shares}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--max-num-seqs", type=int, default=48)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--width", type=int, choices=range(1, 8), default=7)
    parser.add_argument("--context-cap", type=int, default=32)
    parser.add_argument("--drafts-per-step", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--output-length", type=int, default=128)
    parser.add_argument("--memory-fraction", default="0.22")
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_config:
        worker(
            json.loads(args.worker_config.read_text()),
            args.worker_config.with_suffix(".result.json"),
        )
        return
    if any(path is None for path in (args.target, args.draft, args.output)):
        parser.error("--target, --draft and --output are required")
    if not args.target.is_dir() or not args.draft.is_dir():
        parser.error("target and draft must be already downloaded snapshots")
    config = {
        "target": str(args.target.resolve()),
        "draft": str(args.draft.resolve()),
        "requests": args.requests,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "width": args.width,
        "context_cap": args.context_cap,
        "drafts_per_step": args.drafts_per_step,
        "max_model_len": args.max_model_len,
        "output_length": args.output_length,
        "memory_fraction": args.memory_fraction,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    path = args.output.with_suffix(".config.json")
    path.write_text(json.dumps(config, indent=2) + "\n")
    env = dict(
        os.environ,
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
        VLLM_METAL_USE_PAGED_ATTENTION="1",
        VLLM_METAL_MEMORY_FRACTION=str(args.memory_fraction),
        VLLM_METAL_BUILD_FROM_SOURCE="1",
        VLLM_METAL_DSPARK_MAX_CONTEXTS=str(args.context_cap),
        VLLM_METAL_DSPARK_MAX_DRAFTS_PER_STEP=str(args.drafts_per_step),
        HF_HUB_OFFLINE="1",
    )
    with args.output.with_suffix(".log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.dspark_admission_check",
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
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"PASS: {result['context_holders']} of {result['requests']} requests held a "
        f"context (slots {result['context_slots']}, peak concurrent "
        f"{result['max_concurrent_contexts']}), {result['target_only_requests']} "
        f"target-only, drafts-per-step cap {result['drafts_per_step_cap']} "
        f"{'binding' if result['cap_binds'] else 'not binding'}, drafted share "
        f"min/median/max {result['drafted_share']}, proposed {result['proposed_total']}"
    )


if __name__ == "__main__":
    main()
