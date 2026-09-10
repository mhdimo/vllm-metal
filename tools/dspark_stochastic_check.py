# SPDX-License-Identifier: Apache-2.0
"""Real-engine qualification of DSpark stochastic verification (M5).

Runs the pinned pair offline (in-process engine core) as a speculative engine
(``--width`` > 0) and as a target-only engine (K=0) in two worker processes
and compares them on three scenarios:

1. **Distribution.** ``--samples`` unseeded requests on one natural prompt,
   three output tokens each, at temperature 1.0 and at temperature 0.7 with
   top-p 0.9. The first generated token comes from the prefill sampler on both
   engines; the second and third are verified drafts on the speculative
   engine. Each position's marginal histogram is compared between the engines
   with a two-sample chi-square test over the tokens that reach
   ``--min-count`` observations in either sample (rarer tokens pooled), at
   significance ``--alpha``. Distributional equivalence is the contract; the
   test is predefined (sample size, pooling rule and significance) and its
   sensitivity is reported as the total-variation distance it detects with
   80% power for the observed number of buckets. Positive draft and accepted
   token counts are required for the stochastic requests.
2. **Mixed workload.** Greedy, stochastic, seeded, logprobs, penalty,
   short-budget and stop-token requests in one batch. Greedy outputs must
   equal the target-only outputs or diverge only at a tie (the M4a rule,
   judged from the target-only engine's top logprobs at the divergence);
   logprobs and penalty requests must receive no drafts; every output respects
   ``max_tokens``, and a stop token or EOS that lands inside an accepted draft
   prefix ends the output there (no token after it).
3. **Seeded reproducibility.** The mixed batch runs twice on the speculative
   engine with the same seeds and must produce identical outputs. The
   target-only engine's outputs at the same seeds legitimately differ: same
   distribution, different random stream consumption.

``--control-max-num-seqs`` adds a third, target-only engine with a different
scheduler batch size and reports the same two-sample tests between the two
target-only engines. That control is not a gate: it shows how far the
target's own multi-row arithmetic (the M4a numerics contract) moves the
sampled distributions without any drafting, the floor against which the
speculative engine's statistics should be read.

Exit status is nonzero on any failure; ``summary.json`` records every
verdict, histogram and count.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

MIXED_LIMIT = 48


def pooled_two_sample_chi_square(
    left: dict[int, int], right: dict[int, int], min_count: int
) -> dict:
    """Two-sample chi-square over tokens with ``min_count`` in either sample."""
    from scipy.stats import chi2

    keys = sorted(
        token
        for token in set(left) | set(right)
        if left.get(token, 0) >= min_count or right.get(token, 0) >= min_count
    )
    left_counts = [left.get(token, 0) for token in keys]
    right_counts = [right.get(token, 0) for token in keys]
    left_counts.append(sum(left.values()) - sum(left_counts))
    right_counts.append(sum(right.values()) - sum(right_counts))
    total_left, total_right = sum(left_counts), sum(right_counts)
    statistic = 0.0
    buckets = 0
    for a, b in zip(left_counts, right_counts, strict=True):
        pooled = a + b
        if pooled == 0:
            continue
        buckets += 1
        expected_a = pooled * total_left / (total_left + total_right)
        expected_b = pooled * total_right / (total_left + total_right)
        statistic += (a - expected_a) ** 2 / expected_a
        statistic += (b - expected_b) ** 2 / expected_b
    df = max(1, buckets - 1)
    p_value = float(chi2.sf(statistic, df))
    return {
        "buckets": buckets,
        "df": df,
        "statistic": statistic,
        "p_value": p_value,
        "n_left": total_left,
        "n_right": total_right,
    }


def detectable_total_variation(
    n_left: int, n_right: int, df: int, alpha: float
) -> float:
    """TV distance detected with 80% power by the two-sample test (approximate).

    Uses the noncentral chi-square approximation: with per-sample sizes n_a and
    n_b, the noncentrality for effect size w is w^2 * n_a * n_b / (n_a + n_b),
    and w relates to total variation as w >= 2 * TV (Pearson's phi bound), so
    the reported TV is the smallest distance whose w reaches 80% power.
    """
    from scipy.stats import chi2, ncx2

    critical = chi2.ppf(1 - alpha, df)
    effective = n_left * n_right / (n_left + n_right)
    low, high = 0.0, 4.0
    for _ in range(60):
        mid = (low + high) / 2
        power = float(ncx2.sf(critical, df, mid * mid * effective))
        if power < 0.8:
            low = mid
        else:
            high = mid
    return high / 2


def worker(config: dict, output: Path) -> None:
    from vllm import LLM, SamplingParams

    from tools.dspark_memory_check import NATURAL_PROMPTS

    width = config["width"]
    llm = LLM(
        model=config["target"],
        max_model_len=512,
        max_num_seqs=config["max_num_seqs"],
        max_num_batched_tokens=config["max_num_batched_tokens"],
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        disable_log_stats=True,
        seed=0,
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
    tokenizer = llm.get_tokenizer()
    eos = tokenizer.eos_token_id
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    drafted: Counter = Counter()
    accepted: Counter = Counter()
    if width:
        proposer = runner._drafter
        controller = runner._spec_decode_controller
        real_batch_draft = proposer._batch_draft
        real_verify = controller.verify

        # The engine core suffixes the client request id ("7" -> "7-<hash>").
        def client_id(req_id: str) -> str:
            return req_id.rsplit("-", 1)[0]

        def batch_draft(plans):
            result = real_batch_draft(plans)
            for req_id, row in zip(result[0], result[1], strict=True):
                drafted[client_id(req_id)] += len(row)
            return result

        def verify(logits, decode_reqs, decode_segments, **kwargs):
            outputs = real_verify(logits, decode_reqs, decode_segments, **kwargs)
            for (req_id, _), token_ids in zip(decode_reqs, outputs, strict=True):
                accepted[client_id(req_id)] += len(token_ids) - 1
            return outputs

        proposer._batch_draft = batch_draft
        controller.verify = verify

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    def generate(prompt_lists, params_list):
        prompts = [{"prompt_token_ids": ids} for ids in prompt_lists]
        outputs = llm.generate(prompts, params_list, use_tqdm=False)
        rows = []
        for item in outputs:
            choice = item.outputs[0]
            rows.append(
                {
                    "request_id": item.request_id,
                    "token_ids": list(choice.token_ids),
                    "finish_reason": choice.finish_reason,
                    "drafted": drafted.get(item.request_id, 0),
                    "accepted": accepted.get(item.request_id, 0),
                    "logprobs": (
                        [
                            {str(k): v.logprob for k, v in entry.items()}
                            for entry in choice.logprobs
                        ]
                        if choice.logprobs
                        else None
                    ),
                }
            )
        return rows

    report: dict = {"config": config, "eos_token_id": eos}

    # 1. Distribution samples.
    prompt = encode(NATURAL_PROMPTS[config["prompt_index"]])
    distribution = {}
    for name, params in (
        ("t1.0", {"temperature": 1.0}),
        ("t0.7-p0.9", {"temperature": 0.7, "top_p": 0.9}),
    ):
        count = config["samples"]
        rows = generate(
            [prompt] * count,
            [SamplingParams(max_tokens=3, ignore_eos=True, **params)] * count,
        )
        histograms = []
        for position in range(3):
            histogram: Counter = Counter()
            for row in rows:
                if len(row["token_ids"]) > position:
                    histogram[row["token_ids"][position]] += 1
            histograms.append(dict(histogram))
        distribution[name] = {
            "positions": histograms,
            "drafted": sum(row["drafted"] for row in rows),
            "accepted": sum(row["accepted"] for row in rows),
            "requests": count,
        }
    report["distribution"] = distribution

    # 2. Mixed workload (twice for reproducibility).
    def mixed_batch():
        prompts, params, kinds = [], [], []

        def add(kind, text_index, **kw):
            prompts.append(encode(NATURAL_PROMPTS[text_index % len(NATURAL_PROMPTS)]))
            params.append(SamplingParams(**kw))
            kinds.append(kind)

        for index in range(4):
            add("greedy", index, temperature=0.0, max_tokens=MIXED_LIMIT)
        for index in range(4):
            add(
                "stochastic",
                index + 1,
                temperature=0.8,
                top_p=0.95,
                seed=100 + index,
                max_tokens=MIXED_LIMIT,
            )
        for index in range(2):
            add(
                "seeded-topk",
                index + 2,
                temperature=1.0,
                top_k=40,
                seed=200 + index,
                max_tokens=MIXED_LIMIT,
            )
        for index in range(2):
            add("logprobs", index + 3, temperature=0.0, logprobs=2, max_tokens=24)
        for index in range(2):
            add(
                "penalty",
                index + 4,
                temperature=0.8,
                repetition_penalty=1.2,
                seed=300 + index,
                max_tokens=24,
            )
        for index in range(2):
            add(
                "short-budget",
                index + 5,
                temperature=0.9,
                seed=400 + index,
                max_tokens=5,
            )
        add("stochastic-eos", 6, temperature=0.7, seed=500, max_tokens=256)
        add("greedy-eos", 7, temperature=0.0, max_tokens=256)
        stop = encode(".")[-1]
        for index in range(2):
            add(
                "stop-token",
                index,
                temperature=0.8,
                seed=600 + index,
                max_tokens=64,
                stop_token_ids=[stop],
            )
        rows = generate(prompts, params)
        for row, kind, item, ids in zip(rows, kinds, params, prompts, strict=True):
            row["kind"] = kind
            row["max_tokens"] = item.max_tokens
            row["stop_token_ids"] = list(item.stop_token_ids or [])
            row["prompt_token_ids"] = ids
        return rows

    first = mixed_batch()
    second = mixed_batch()
    report["mixed"] = first
    report["mixed_repeat_identical"] = [row["token_ids"] for row in first] == [
        row["token_ids"] for row in second
    ]

    # 3. Tie probes for the greedy comparison (target-only engine only).
    if not width and config.get("compare_to"):
        other = json.loads(Path(config["compare_to"]).read_text())
        probes = []
        for mine, theirs in zip(report["mixed"], other["mixed"], strict=True):
            if mine["kind"] not in ("greedy", "logprobs", "greedy-eos"):
                continue
            expected, actual = mine["token_ids"], theirs["token_ids"]
            position = next(
                (
                    i
                    for i, (a, b) in enumerate(zip(expected, actual, strict=False))
                    if a != b
                ),
                None,
            )
            if position is None and len(expected) == len(actual):
                probes.append({"kind": mine["kind"], "equal": True})
                continue
            if position is None:
                probes.append(
                    {
                        "kind": mine["kind"],
                        "equal": False,
                        "label": "length",
                        "admissible": False,
                    }
                )
                continue
            prefix = mine["prompt_token_ids"] + expected[:position]
            out = generate(
                [prefix],
                [
                    SamplingParams(
                        temperature=0.0, max_tokens=1, logprobs=5, ignore_eos=True
                    )
                ],
            )[0]
            top = out["logprobs"][0] if out["logprobs"] else {}
            a, b = expected[position], actual[position]
            gap = (
                abs(top[str(a)] - top[str(b)])
                if str(a) in top and str(b) in top
                else None
            )
            probes.append(
                {
                    "kind": mine["kind"],
                    "equal": False,
                    "position": position,
                    "expected": a,
                    "actual": b,
                    "top": top,
                    "gap": gap,
                    "label": "tie" if gap is not None and gap <= 0.5 else "divergence",
                    "admissible": gap is not None and gap <= 0.5,
                }
            )
        report["greedy_parity"] = probes
    output.write_text(json.dumps(report, indent=2) + "\n")


def distribution_tests(
    left: dict, right: dict, alpha: float, min_count: int
) -> dict[str, list[dict]]:
    tests: dict[str, list[dict]] = {}
    for name, item in left["distribution"].items():
        positions = []
        for position in range(3):
            a = {int(k): v for k, v in item["positions"][position].items()}
            b = {
                int(k): v
                for k, v in right["distribution"][name]["positions"][position].items()
            }
            test = pooled_two_sample_chi_square(a, b, min_count)
            test["detectable_tv_80pct_power"] = detectable_total_variation(
                test["n_left"], test["n_right"], test["df"], alpha
            )
            test["pass"] = test["p_value"] >= alpha
            positions.append(test)
        tests[name] = positions
    return tests


def evaluate(
    spec: dict,
    base: dict,
    alpha: float,
    min_count: int,
    control: dict | None = None,
) -> tuple[dict, list[str]]:
    failures: list[str] = []
    summary: dict = {
        "distribution": {},
        "mixed": {},
        "reproducible": spec["mixed_repeat_identical"],
    }
    if control is not None:
        summary["control"] = {
            "max_num_seqs": control["config"]["max_num_seqs"],
            "distribution": distribution_tests(base, control, alpha, min_count),
        }
    tests = distribution_tests(spec, base, alpha, min_count)
    for name, item in spec["distribution"].items():
        positions = tests[name]
        for position, test in enumerate(positions):
            if not test["pass"]:
                failures.append(
                    f"distribution {name} position {position}: "
                    f"p={test['p_value']:.4g} < {alpha}"
                )
        summary["distribution"][name] = {
            "positions": positions,
            "drafted": item["drafted"],
            "accepted": item["accepted"],
        }
        if item["drafted"] <= 0 or item["accepted"] <= 0:
            failures.append(
                f"distribution {name}: no draft work (drafted {item['drafted']}, "
                f"accepted {item['accepted']})"
            )
    if not spec["mixed_repeat_identical"]:
        failures.append(
            "seeded mixed batch is not reproducible on the speculative engine"
        )
    eos = spec["eos_token_id"]
    kinds: dict[str, dict] = {}
    for row in spec["mixed"]:
        kind = row["kind"]
        entry = kinds.setdefault(kind, {"requests": 0, "drafted": 0, "accepted": 0})
        entry["requests"] += 1
        entry["drafted"] += row["drafted"]
        entry["accepted"] += row["accepted"]
        tokens = row["token_ids"]
        if len(tokens) > row["max_tokens"]:
            failures.append(
                f"mixed {kind}: {len(tokens)} tokens exceed {row['max_tokens']}"
            )
        terminal = {eos, *row.get("stop_token_ids", [])}
        if any(token in terminal for token in tokens[:-1]):
            failures.append(f"mixed {kind}: token after EOS or a stop token")
        if kind == "stop-token":
            entry["stopped"] = entry.get("stopped", 0) + int(
                bool(tokens) and tokens[-1] in row["stop_token_ids"]
            )
        if kind in ("logprobs", "penalty") and row["drafted"]:
            failures.append(f"mixed {kind}: drafted although not draftable")
    for kind in (
        "stochastic",
        "seeded-topk",
        "short-budget",
        "stochastic-eos",
        "stop-token",
    ):
        entry = kinds.get(kind)
        if entry is None or entry["drafted"] <= 0 or entry["accepted"] <= 0:
            failures.append(f"mixed {kind}: no draft work ({entry})")
    if kinds.get("stop-token", {}).get("stopped", 0) == 0:
        failures.append("mixed stop-token: no request ended on its stop token")
    summary["mixed"] = kinds
    parity = base.get("greedy_parity", [])
    summary["greedy_parity"] = parity
    for probe in parity:
        if not probe.get("equal") and not probe.get("admissible"):
            failures.append(f"greedy parity: {probe}")
    return summary, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--width", type=int, default=7)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--memory-fraction", default="0.22")
    parser.add_argument(
        "--control-max-num-seqs",
        type=int,
        default=0,
        help="also run a target-only engine at this batch size as a numerics control",
    )
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
        HF_HUB_OFFLINE="1",
    )
    results = {}
    plan = [
        (args.width, args.max_num_seqs, f"k{args.width}"),
        (0, args.max_num_seqs, "k0"),
    ]
    if args.control_max_num_seqs:
        plan.append((0, args.control_max_num_seqs, "k0-control"))
    for width, max_num_seqs, name in plan:
        config = {
            "target": str(args.target.resolve()),
            "draft": str(args.draft.resolve()),
            "width": width,
            "samples": args.samples,
            "prompt_index": args.prompt_index,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
        }
        if name == "k0":
            config["compare_to"] = str(results[f"k{args.width}"])
        path = args.output_dir / f"{name}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        with path.with_suffix(".log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.dspark_stochastic_check",
                    "--worker-config",
                    str(path),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=7200,
            )
        results[name] = path.with_suffix(".result.json")
    spec = json.loads(results[f"k{args.width}"].read_text())
    base = json.loads(results["k0"].read_text())
    control = (
        json.loads(results["k0-control"].read_text())
        if "k0-control" in results
        else None
    )
    summary, failures = evaluate(spec, base, args.alpha, args.min_count, control)
    summary["failures"] = failures
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, item in summary["distribution"].items():
        for position, test in enumerate(item["positions"]):
            print(
                f"{name} position {position}: buckets {test['buckets']}, chi2 {test['statistic']:.1f}, "
                f"p {test['p_value']:.3g}, detectable TV {test['detectable_tv_80pct_power']:.3f}, "
                f"{'PASS' if test['pass'] else 'FAIL'}",
                flush=True,
            )
        print(f"{name}: drafted {item['drafted']} accepted {item['accepted']}")
    for kind, entry in summary["mixed"].items():
        print(f"mixed {kind}: {entry}")
    print("greedy parity:", [p.get("label", "equal") for p in summary["greedy_parity"]])
    print("reproducible:", summary["reproducible"])
    for name, positions in summary.get("control", {}).get("distribution", {}).items():
        for position, test in enumerate(positions):
            print(
                f"control (target-only at {summary['control']['max_num_seqs']} seqs) {name} "
                f"position {position}: buckets {test['buckets']}, chi2 {test['statistic']:.1f}, "
                f"p {test['p_value']:.3g}",
                flush=True,
            )
    if failures:
        print("FAILURES:")
        for failure in failures:
            print(" -", failure)
        sys.exit(1)
    print("GATE PASS")


if __name__ == "__main__":
    main()
