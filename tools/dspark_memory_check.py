# SPDX-License-Identifier: Apache-2.0
"""Bounded real-engine DSpark memory soak and recovery, using local snapshots.

Baseline and speculative engines run sequentially in fresh subprocesses. Every
completed token stream must match; target-only fallbacks cannot satisfy the
normal-round speculation gate. Allocation faults are injected deliberately,
without exhausting the host's RAM. This is not the M7 HTTP/production soak.
"""

from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

from tools.dspark_lifecycle_check import PROMPTS

# Non-degenerate workload: varied natural prompts that ask for long answers, so
# a greedy stream is not a repeated sentence with logits near 40. The repeated
# fixtures above drive the target into a bistable regime; this set measures
# parity where the target's greedy decisions are stable.
NATURAL_PROMPTS = [
    "Explain how a hash table handles collisions, and compare chaining with open "
    "addressing in terms of memory use and cache behavior.",
    "Write a short story about a lighthouse keeper who discovers that the light "
    "has started to attract something other than ships.",
    "Describe the steps a compiler takes to turn source code into an executable, "
    "and say what can go wrong at each step.",
    "A recipe for bread calls for 500 grams of flour and 350 grams of water. "
    "Explain what hydration percentage means and how changing it affects the loaf.",
    "Summarize the causes of the 1929 stock market crash for a high school "
    "student, then explain what changed in banking regulation afterwards.",
    "Write a Python class that implements an LRU cache with get and put methods, "
    "then explain the time complexity of each operation.",
    "Why does the sky appear blue during the day but red near the horizon at "
    "sunset? Include the physics of scattering in your answer.",
    "Compare the strengths and weaknesses of trains, buses and bicycles for a "
    "city planner deciding how to reduce traffic in a mid-sized city.",
]


def top_logit_rows(
    paged_state, round_index: int | None, forward_index: int, top_k: int = 8
) -> list[dict]:
    """Top-``top_k`` target logits for every row the runner is about to sample.

    Rows are identified by the runner's own DTOs: decode/verification segments
    carry ``start_row``/``cache_start_pos`` and the request state, prefill
    entries carry ``start_pos``. ``pos`` is the absolute token index the row
    predicts. ``top[0]`` is ``mx.argmax`` of the row, i.e. the greedy token the
    verifier or sampler emits; the remaining entries are sorted descending.
    """
    import mlx.core as mx

    specs: list[tuple] = []
    for (req_id, state), segment in zip(
        paged_state.decode_reqs, paged_state.decode_segments, strict=True
    ):
        kind = "verify" if segment.draft_token_ids else "decode"
        for i in range(segment.num_query_tokens):
            draft = (
                segment.draft_token_ids[i] if i < len(segment.draft_token_ids) else None
            )
            specs.append(
                (
                    segment.start_row + i,
                    req_id,
                    len(state.token_ids) + i,
                    segment.num_query_tokens,
                    kind,
                    draft,
                    segment.cache_start_pos,
                    len(state.token_ids),
                )
            )
    boundaries = list(paged_state.logits_cu_seqlens)
    num_decode = len(paged_state.decode_segments)
    for j, prefill in enumerate(paged_state.prefill_reqs):
        if prefill.prompt_len is None:
            continue  # intermediate chunk: nothing is sampled
        specs.append(
            (
                boundaries[num_decode + j + 1] - 1,
                prefill.req_id,
                prefill.start_pos + len(prefill.token_ids),
                len(prefill.token_ids),
                "prefill",
                None,
                prefill.start_pos,
                prefill.prompt_len,
            )
        )
    if not specs:
        return []
    logits = paged_state.logits[0]
    selected = logits[mx.array([spec[0] for spec in specs])].astype(mx.float32)
    top1 = mx.argmax(selected, axis=-1)
    top1_values = mx.take_along_axis(selected, top1[:, None], axis=-1)[:, 0]
    vocab = mx.arange(selected.shape[-1])[None, :]
    masked = mx.where(vocab == top1[:, None], -mx.inf, selected)
    rest = mx.argpartition(-masked, kth=top_k - 2, axis=-1)[:, : top_k - 1]
    rest_values = mx.take_along_axis(selected, rest, axis=-1)
    order = mx.argsort(-rest_values, axis=-1)
    rest = mx.take_along_axis(rest, order, axis=-1)
    rest_values = mx.take_along_axis(rest_values, order, axis=-1)
    mx.eval(top1, top1_values, rest, rest_values)
    rows = []
    for index, (row, req_id, pos, count, kind, draft, cache_start, length) in enumerate(
        specs
    ):
        top = [[int(top1[index].item()), float(top1_values[index].item())]]
        top.extend(
            [int(token), float(value)]
            for token, value in zip(
                rest[index].tolist(), rest_values[index].tolist(), strict=True
            )
        )
        rows.append(
            {
                "round": round_index,
                "forward": forward_index,
                "req": req_id,
                "row": int(row),
                "pos": int(pos),
                "rows": int(count),
                "kind": kind,
                "draft": draft,
                "cache_start_pos": int(cache_start),
                "state_len": int(length),
                "top": top,
            }
        )
    return rows


def full_capacity_probe(proposer) -> dict:
    """Exercise the full reservation with synthetic features and resident target KV.

    This directly fills every context slot, independent of scheduler arrivals
    that may finish fast requests before every slot reaches maximum length.
    It is an allocation test, not token-parity or workload-performance evidence.
    """
    import mlx.core as mx
    from vllm import SamplingParams

    from vllm_metal.v1.dspark_proposer import _DraftPlan, _RequestContext
    from vllm_metal.v1.model_runner import RequestState

    plan = proposer.memory_plan
    plans = []
    for index in range(plan.max_contexts):
        record = _RequestContext(
            owner=RequestState(
                token_ids=[1] * (plan.max_context_tokens + 1),
                prompt_len=plan.max_context_tokens,
                cache=[],
                sampling_params=SamplingParams(temperature=0.0),
            ),
            caches=proposer._drafter.make_ctx_cache(plan.max_context_tokens),
            covered_end=plan.max_context_tokens,
        )
        width = proposer._config.hidden_size * len(proposer.capture_layer_ids)
        for start in range(0, plan.max_context_tokens, plan.max_step_tokens):
            count = min(plan.max_step_tokens, plan.max_context_tokens - start)
            features = mx.sin(mx.arange(count * width).reshape(1, count, width) * 0.01)
            proposer._drafter.update_context(features, start, record.caches)
            mx.eval([(cache.k, cache.v) for cache in record.caches])
            del features
        plans.append(_DraftPlan(str(index), 1, record, proposer._block_size))
    allocated = sum(c.allocated_bytes for row in plans for c in row.context.caches)
    assert allocated == plan.context_bytes
    _, tokens = proposer._batch_draft(plans)
    assert len(tokens) == plan.max_contexts
    assert all(len(row) == proposer._block_size for row in tokens)
    mx.synchronize()
    return {
        "fixture": "synthetic sinusoidal features; all slots at full capacity",
        "context_bytes": allocated,
        "peak_mlx_bytes": mx.get_peak_memory(),
        "active_plus_cache_bytes": mx.get_active_memory() + mx.get_cache_memory(),
    }


def worker(config: dict, output: Path) -> None:
    import mlx.core as mx
    from vllm import LLM, SamplingParams

    from vllm_metal.v1.dspark_proposer import DSparkProposer

    width = config["width"]
    llm = LLM(
        model=config["target"],
        max_model_len=config["max_model_len"],
        max_num_seqs=4,
        max_num_batched_tokens=64,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        disable_log_stats=True,
        num_gpu_blocks_override=config.get("num_gpu_blocks"),
        speculative_config=(
            {
                "method": config.get("method", "dspark"),
                "model": config["draft"],
                "num_speculative_tokens": width,
            }
            if width
            else None
        ),
    )
    engine = llm.llm_engine
    runner = engine.model_executor.driver_worker.model_runner
    counts = collections.Counter()
    fallbacks = collections.Counter()
    active_samples, total_samples = [], []
    trace_rows: list[dict] = []
    current_round = {"value": None}
    forward_counter = {"value": 0}
    logits_dtype = {"value": None}
    if config.get("trace_logits"):
        # Read the stashed forward state before the runner consumes it, for
        # every engine (baseline included). Identity comes from runner DTOs.
        real_sample_batch = runner._sample_paged_batch

        def traced_sample_batch(*args, **kwargs):
            paged_state = runner._execute_model_state
            if (
                paged_state is not None
                and paged_state.logits is not None
                and not paged_state.intermediate_only
            ):
                logits_dtype["value"] = str(paged_state.logits.dtype)
                trace_rows.extend(
                    top_logit_rows(
                        paged_state, current_round["value"], forward_counter["value"]
                    )
                )
                forward_counter["value"] += 1
            return real_sample_batch(*args, **kwargs)

        runner._sample_paged_batch = traced_sample_batch
    budget = int(
        mx.device_info()["max_recommended_working_set_size"]
        * float(config["memory_fraction"])
    )
    texts = [
        PROMPTS[0],
        PROMPTS[1],
        PROMPTS[2] + "\n" + PROMPTS[1] * 2,
        PROMPTS[3] + "\n" + PROMPTS[1] * 4,
    ]
    if config.get("prompt_set") in ("shared", "shared-short"):
        # Balanced arrivals keep several sequences alive while their shared
        # physical KV allowance fills, forcing genuine scheduler preemption.
        texts = [PROMPTS[0 if config["prompt_set"] == "shared-short" else 1]] * 4
    elif config.get("prompt_set") == "natural":
        texts = list(NATURAL_PROMPTS)
    tokenized = [
        llm.get_tokenizer().encode(text, add_special_tokens=False)[
            : config["max_model_len"] - config["output_length"] - 8
        ]
        for text in texts
    ]
    prompts = [{"prompt_token_ids": row} for row in tokenized]
    params = SamplingParams(
        temperature=0.0, max_tokens=config["output_length"], ignore_eos=True
    )
    if width:
        proposer = runner._drafter
        assert isinstance(proposer, DSparkProposer)
        real_propose = proposer.propose
        real_execute = runner.execute_model
        real_verify = runner._spec_decode_controller.verify_greedy

        def propose(ctx):
            result = real_propose(ctx)
            stored = 0
            for record in proposer._contexts.values():
                if record.disabled_reason:
                    fallbacks[record.disabled_reason] += 1
                    assert not record.caches
                else:
                    for cache in record.caches:
                        assert cache.length == record.covered_end
                        if cache.k is not None:
                            assert (
                                cache.k.dtype
                                == proposer._drafter.hidden_norm.weight.dtype
                            )
                        stored += cache.allocated_bytes
            assert stored <= proposer.memory_plan.context_bytes
            counts["max_context_bytes"] = max(counts["max_context_bytes"], stored)
            if result:
                counts["proposed"] += sum(map(len, result.draft_token_ids))
            counts["max_active_plus_cache_bytes"] = max(
                counts["max_active_plus_cache_bytes"],
                mx.get_active_memory() + mx.get_cache_memory(),
            )
            return result

        def execute(scheduled, *args, **kwargs):
            counts["preemptions"] += len(scheduled.preempted_req_ids or ())
            return real_execute(scheduled, *args, **kwargs)

        def verify(logits, reqs, segments):
            result = real_verify(logits, reqs, segments)
            for (_, state), segment, tokens in zip(reqs, segments, result, strict=True):
                counts["verified"] += len(segment.draft_token_ids)
                counts["accepted"] += max(0, len(tokens) - 1)
                if config.get("diagnose_mismatch") and expected is not None:
                    prompt = state.token_ids[: state.prompt_len]
                    reference = expected[tokenized.index(prompt)]
                    position = len(state.token_ids) - state.prompt_len
                    if state.token_ids[state.prompt_len :] != reference[:position]:
                        continue
                    for offset, token in enumerate(tokens):
                        if token == reference[position + offset]:
                            continue
                        row = logits[0, segment.start_row + offset].astype(mx.float32)
                        top = mx.argsort(row)[-5:][::-1].tolist()
                        output.with_suffix(".trace.json").write_text(
                            json.dumps(
                                {
                                    "prompt_token_ids": prompt,
                                    "prefix_token_ids": state.token_ids
                                    + tokens[:offset],
                                    "output_position": position + offset,
                                    "expected_token": reference[position + offset],
                                    "actual_token": token,
                                    "top_logits": [[i, row[i].item()] for i in top],
                                    "expected_logit": row[
                                        reference[position + offset]
                                    ].item(),
                                    "query_tokens": segment.num_query_tokens,
                                    "draft_index": offset,
                                },
                                indent=2,
                            )
                            + "\n"
                        )
                        raise AssertionError("first divergent target logit recorded")
            return result

        proposer.propose = propose
        runner.execute_model = execute
        runner._spec_decode_controller.verify_greedy = verify

    expected = config.get("expected_tokens")
    mismatched_rounds = []
    begin = time.perf_counter()
    warmup = config.get("warmup_rounds", 4) if width else 0
    for cycle in range(config["rounds"] + warmup):
        before = counts["proposed"]
        # Exercise each fallback once, then require active drafting again on
        # subsequent requests. These are independent from scheduler preemption.
        fault = cycle - warmup if config["faults"] and width else -1
        if width:
            original_plan = proposer.memory_plan
            original_budget = proposer._memory_budget_bytes
            original_update = proposer._drafter.update_context
            if fault == 4:
                proposer._memory_budget_bytes = 0
            elif fault == 8:
                proposer.memory_plan = replace(original_plan, max_contexts=1)
            elif fault == 12:

                def fail_after_write(*args, _update=original_update, **kwargs):
                    _update(*args, **kwargs)
                    proposer._drafter.update_context = _update
                    counts["injected_allocation_failures"] += 1
                    raise MemoryError(
                        "DSpark qualification: injected after context write"
                    )

                proposer._drafter.update_context = fail_after_write
        current_round["value"] = cycle
        try:
            outputs = llm.generate(prompts, params, use_tqdm=False)
        finally:
            if width:
                proposer.memory_plan = original_plan
                proposer._memory_budget_bytes = original_budget
                proposer._drafter.update_context = original_update
        actual = [list(item.outputs[0].token_ids) for item in outputs]
        assert all(len(row) == config["output_length"] for row in actual)
        assert [item.prompt_token_ids for item in outputs] == tokenized
        if expected is None:
            expected = actual
        if actual != expected:
            output.with_suffix(".failure.json").write_text(
                json.dumps(
                    {
                        "round": cycle,
                        "prompt_token_ids": tokenized,
                        "expected_tokens": expected,
                        "actual_tokens": actual,
                        "counts": dict(counts),
                    },
                    indent=2,
                )
                + "\n"
            )
            # Finish normal drain/resource checks and retain their evidence,
            # then fail the process. A numerical divergence must never turn
            # this into a successful exact-token qualification.
            mismatched_rounds.append(cycle)
        counts["completed"] += len(outputs)
        engine.step()  # Normal scheduler finish notification, no private cleanup.
        if width:
            assert not proposer._contexts
            if fault not in (4, 8, 12):
                assert counts["proposed"] > before, (
                    f"no speculation in normal round {cycle}"
                )
                counts["normal_rounds_with_drafts"] += 1

        if width and cycle % 8 == 7:
            current_round["value"] = -1
            public_id = "dspark-memory-reuse"
            internal_id = engine.add_request(public_id, prompts[-1], params)
            engine.step()
            assert internal_id in proposer._contexts
            engine.abort_request([public_id])
            engine.step()
            assert not proposer._contexts
            counts["cancelled"] += 1
            engine.add_request(public_id, prompts[0], params)
            last = None
            while engine.has_unfinished_requests():
                for item in engine.step():
                    last = list(item.outputs[0].token_ids)
            engine.step()
            assert last == expected[0]
            assert not proposer._contexts
            counts["reused"] += 1
        gc.collect()
        mx.synchronize()
        active_samples.append(mx.get_active_memory())
        total_samples.append(mx.get_active_memory() + mx.get_cache_memory())
        if cycle % 8 == 7:
            print(
                f"round={cycle + 1}, completed={counts['completed']}, active={active_samples[-1]}",
                flush=True,
            )

    measured = active_samples[warmup:]
    drift = max(measured) - min(measured)
    assert drift <= 2 * 1024**2, f"retained active memory grows: range={drift}"
    capacity = None
    if width and config.get("capacity_probe"):
        capacity = full_capacity_probe(proposer)
        assert capacity["active_plus_cache_bytes"] <= budget
        gc.collect()
        mx.synchronize()
        assert abs(mx.get_active_memory() - active_samples[-1]) <= 2 * 1024**2
        assert not proposer._contexts
    assert mx.get_peak_memory() <= budget, "peak active allocation exceeds allowance"
    assert max(total_samples + [counts["max_active_plus_cache_bytes"]]) <= budget
    if width:
        assert counts["verified"] > 0 and counts["accepted"] > 0
        if config["faults"]:
            assert fallbacks["context memory budget exhausted"] > 0
            assert fallbacks["context capacity exhausted"] > 0
            assert counts["injected_allocation_failures"] == 1
        if config.get("require_preemption"):
            assert counts["preemptions"] > 0
    result = {
        "config": {
            key: value for key, value in config.items() if key != "expected_tokens"
        },
        "prompt_token_ids": tokenized,
        "tokens": expected,
        "token_sha256": hashlib.sha256(json.dumps(expected).encode()).hexdigest(),
        "greedy_tokens_equal": not mismatched_rounds,
        "mismatched_rounds": mismatched_rounds,
        "resource_checks_passed": True,
        "counts": dict(counts),
        "fallbacks": dict(fallbacks),
        "active_after_drain_bytes": active_samples,
        "active_drift_bytes": drift,
        "peak_mlx_bytes": mx.get_peak_memory(),
        "max_observed_active_plus_cache_bytes": max(
            total_samples
            + [counts["max_active_plus_cache_bytes"]]
            + ([capacity["active_plus_cache_bytes"]] if capacity else [])
        ),
        "budget_bytes": budget,
        "elapsed_s": time.perf_counter() - begin,
        "memory_plan": asdict(proposer.memory_plan) if width else None,
        "full_capacity_probe": capacity,
        "runtime": {
            package: importlib.metadata.version(package)
            for package in ("mlx", "mlx-lm", "vllm", "torch")
        },
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    if config.get("trace_logits"):
        output.with_suffix(".logits.json").write_text(
            json.dumps(
                {
                    "config": result["config"],
                    "prompt_token_ids": tokenized,
                    "logits_dtype": logits_dtype["value"],
                    "rows": trace_rows,
                },
                indent=None,
            )
            + "\n"
        )
    assert not mismatched_rounds, f"greedy mismatch in rounds {mismatched_rounds}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rounds", type=int, default=64)
    parser.add_argument("--warmup-rounds", type=int, default=4)
    parser.add_argument("--width", type=int, choices=range(1, 8), default=7)
    parser.add_argument("--method", choices=("dspark", "draft_model"), default="dspark")
    parser.add_argument(
        "--prompt-set",
        choices=("ragged", "shared", "shared-short", "natural"),
        default="ragged",
    )
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--output-length", type=int, default=64)
    parser.add_argument("--memory-fraction", type=float, default=0.22)
    parser.add_argument("--num-gpu-blocks", type=int)
    parser.add_argument("--require-preemption", action="store_true")
    parser.add_argument("--faults", action="store_true")
    parser.add_argument("--capacity-probe", action="store_true")
    parser.add_argument("--diagnose-mismatch", action="store_true")
    parser.add_argument(
        "--trace-logits",
        action="store_true",
        help="record top-8 target logits for every sampled row of both engines "
        "(kN.result.logits.json) for tools.dspark_divergence_classify",
    )
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_config:
        worker(
            json.loads(args.worker_config.read_text()),
            args.worker_config.with_suffix(".result.json"),
        )
        return
    if (
        args.rounds < (16 if args.faults else 1)
        or args.warmup_rounds < 0
        or not 0 < args.output_length < args.max_model_len - 8
    ):
        parser.error("invalid round count or token limits")
    if not 0 < args.memory_fraction <= 1:
        parser.error("memory fraction must be in (0, 1]")
    if any(path is None for path in (args.target, args.draft, args.output_dir)):
        parser.error("--target, --draft and --output-dir are required")
    if not args.target.is_dir() or not args.draft.is_dir():
        parser.error("target and draft must be already downloaded snapshots")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
        VLLM_METAL_USE_PAGED_ATTENTION="1",
        VLLM_METAL_MEMORY_FRACTION=str(args.memory_fraction),
        VLLM_METAL_BUILD_FROM_SOURCE="1",
        HF_HUB_OFFLINE="1",
    )
    expected = None
    for width in (0, args.width):
        config = {
            "target": str(args.target.resolve()),
            "draft": str(args.draft.resolve()),
            "width": width,
            "method": args.method,
            "prompt_set": args.prompt_set,
            "rounds": args.rounds if width else 1,
            "warmup_rounds": args.warmup_rounds,
            "max_model_len": args.max_model_len,
            "output_length": args.output_length,
            "memory_fraction": args.memory_fraction,
            "num_gpu_blocks": args.num_gpu_blocks,
            "require_preemption": args.require_preemption,
            "faults": args.faults,
            "capacity_probe": args.capacity_probe,
            "diagnose_mismatch": args.diagnose_mismatch,
            "trace_logits": args.trace_logits,
            "expected_tokens": expected,
        }
        path = args.output_dir / f"k{width}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        with path.with_suffix(".log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.dspark_memory_check",
                    "--worker-config",
                    str(path),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=3600,
            )
        expected = json.loads(path.with_suffix(".result.json").read_text())["tokens"]
        print(f"PASS K={width}: {path.with_suffix('.result.json')}", flush=True)


if __name__ == "__main__":
    main()
