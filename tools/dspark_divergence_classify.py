# SPDX-License-Identifier: Apache-2.0
"""Classify greedy divergences between a target-only and a speculative engine run.

Input is a ``dspark_memory_check --trace-logits`` directory: ``k0.result.json``
and ``k0.result.logits.json`` for the baseline, ``kN.result.json`` (or the
``.failure.json`` written on mismatch) and ``kN.result.logits.json`` for the
speculative engine, plus optionally a ``dspark_target_replay`` result for the
same failure. Every first divergence is labelled from the engines' own logits:

- ``tie``: both engines rank the two tokens first and second and each gap is
  within ``MAX_TIE_ULPS`` bfloat16 ULPs (the criterion upstream's draft-model
  e2e adopted in #524). Summation order between one-row decode and multi-row
  verification or prefill can split such a tie; no state is wrong.
- ``engine-disagreement``: the two engines assign materially different logits
  to the same committed prefix. That is invalid state (KV, slot map, position,
  chunking) in at least one engine, never a benign tie. The native replay, when
  supplied, names which engine agrees with mlx-lm at that prefix.

Identical prompts are also compared within one engine: two requests with the
same committed prefix that receive different logits are reported the same way.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

MAX_TIE_ULPS = 2
DTYPE_MANTISSA_BITS = {"bfloat16": 7, "float16": 10, "float32": 23}


def ulp(value: float, mantissa_bits: int = DTYPE_MANTISSA_BITS["bfloat16"]) -> float:
    """Spacing of the representable values around ``value`` for the given format."""
    magnitude = abs(value)
    if magnitude == 0.0 or math.isinf(magnitude) or math.isnan(magnitude):
        return float("inf") if magnitude else 2.0**-133
    return 2.0 ** (math.floor(math.log2(magnitude)) - mantissa_bits)


def first_divergence(expected: list[int], actual: list[int]) -> int | None:
    for index, (left, right) in enumerate(zip(expected, actual, strict=False)):
        if left != right:
            return index
    return None if len(expected) == len(actual) else min(len(expected), len(actual))


def _rows_by_round(rows: list[dict]) -> dict[int, dict[str, list[dict]]]:
    grouped: dict[int, dict[str, list[dict]]] = {}
    for row in rows:
        grouped.setdefault(row["round"], {}).setdefault(row["req"], []).append(row)
    return grouped


def emitted_tokens(rows: list[dict]) -> dict[int, dict]:
    """Map every committed absolute position to the row that produced it.

    Forwards are replayed in recorded order so a recompute after preemption
    overwrites the earlier record for the same position, as the token stream
    does. Decode and prefill rows commit their argmax. A verification window
    commits its argmax rows up to and including the first draft mismatch (the
    correction) or, when every draft matches, the bonus row; later rows of a
    rejected window never reach the stream and are omitted.
    """
    committed: dict[int, dict] = {}
    by_forward: dict[int, list[dict]] = {}
    for row in rows:
        by_forward.setdefault(row["forward"], []).append(row)
    for forward in sorted(by_forward):
        window = sorted(by_forward[forward], key=lambda r: r["row"])
        if window[0]["kind"] != "verify":
            for row in window:
                committed[row["pos"]] = row
            continue
        for row in window:
            committed[row["pos"]] = row
            if row["draft"] is None or row["top"][0][0] != row["draft"]:
                break
    return committed


def assign_requests(
    grouped: dict[str, list[dict]],
    prompts: list[list[int]],
    streams: list[list[int]],
) -> dict[int, str]:
    """Match traced request ids to prompt indices by the committed token stream.

    Identical prompts make the id order ambiguous, so every assignment is
    validated against the recorded output tokens; the permutation whose traced
    argmax stream reproduces each recorded stream is the only accepted one.
    """
    ids = list(grouped)
    if len(ids) != len(prompts):
        raise ValueError(
            f"trace has {len(ids)} requests in this round, results have {len(prompts)}"
        )
    emitted = {req: emitted_tokens(rows) for req, rows in grouped.items()}

    def matches(req: str, index: int) -> bool:
        prompt_len = len(prompts[index])
        stream = streams[index]
        table = emitted[req]
        for offset, token in enumerate(stream):
            row = table.get(prompt_len + offset)
            if row is None or row["top"][0][0] != token:
                return False
        return True

    for permutation in itertools.permutations(ids):
        if all(matches(req, index) for index, req in enumerate(permutation)):
            return dict(enumerate(permutation))
    raise ValueError("no request assignment reproduces the recorded token streams")


def _lookup(top: list[list], token: int) -> float | None:
    for candidate, value in top:
        if candidate == token:
            return value
    return None


def classify_pair(
    baseline_row: dict,
    speculative_row: dict,
    baseline_token: int,
    speculative_token: int,
    mantissa_bits: int,
) -> dict:
    verdict: dict = {
        "baseline": {
            "top": baseline_row["top"][:4],
            "rows_in_forward": baseline_row["rows"],
            "kind": baseline_row["kind"],
        },
        "speculative": {
            "top": speculative_row["top"][:4],
            "rows_in_forward": speculative_row["rows"],
            "kind": speculative_row["kind"],
            "draft": speculative_row["draft"],
        },
    }
    checks = []
    for name, row, own, other in (
        ("baseline", baseline_row, baseline_token, speculative_token),
        ("speculative", speculative_row, speculative_token, baseline_token),
    ):
        top = row["top"]
        own_value = _lookup(top, own)
        other_value = _lookup(top, other)
        entry = verdict[name]
        entry["own_logit"] = own_value
        entry["other_logit"] = other_value
        entry["argmax_is_own"] = top[0][0] == own
        if own_value is None or other_value is None:
            entry["gap_ulps"] = None
            checks.append(False)
            continue
        gap = own_value - other_value
        entry["gap"] = gap
        entry["gap_ulps"] = gap / ulp(
            max(abs(own_value), abs(other_value)), mantissa_bits
        )
        top_two = {top[0][0], top[1][0]}
        checks.append(
            top_two == {own, other} and abs(entry["gap_ulps"]) <= MAX_TIE_ULPS
        )
    if not (
        verdict["baseline"]["argmax_is_own"] and verdict["speculative"]["argmax_is_own"]
    ):
        verdict["label"] = "trace-inconsistent"
    elif all(checks):
        verdict["label"] = "tie"
    else:
        verdict["label"] = "engine-disagreement"
        b_top = baseline_row["top"][0][1]
        s_top = speculative_row["top"][0][1]
        verdict["top1_logit_difference"] = s_top - b_top
    return verdict


def load_run(run_dir: Path, width: int) -> tuple[dict, dict]:
    result_path = run_dir / f"k{width}.result.json"
    failure_path = run_dir / f"k{width}.result.failure.json"
    trace_path = run_dir / f"k{width}.result.logits.json"
    if not trace_path.exists():
        raise SystemExit(
            f"{trace_path} missing: rerun dspark_memory_check with --trace-logits"
        )
    trace = json.loads(trace_path.read_text())
    if failure_path.exists():
        failure = json.loads(failure_path.read_text())
        result = {
            "prompt_token_ids": failure["prompt_token_ids"],
            "tokens": failure["actual_tokens"],
            "expected_tokens": failure["expected_tokens"],
            "round": failure["round"],
        }
    else:
        result = json.loads(result_path.read_text())
        result.setdefault("round", 0)
    return result, trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--width", type=int, required=True, help="speculative K of the run"
    )
    parser.add_argument(
        "--replay", type=Path, help="dspark_target_replay output for this failure"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline, baseline_trace = load_run(args.run_dir, 0)
    speculative, speculative_trace = load_run(args.run_dir, args.width)
    prompts = baseline["prompt_token_ids"]
    assert prompts == speculative["prompt_token_ids"], "prompt sets differ"
    # Gaps are measured in bfloat16 ULPs whatever the stored logits dtype, the
    # criterion upstream's draft-model e2e adopted in #524.
    mantissa = DTYPE_MANTISSA_BITS["bfloat16"]
    baseline_rounds = _rows_by_round(baseline_trace["rows"])
    speculative_rounds = _rows_by_round(speculative_trace["rows"])
    baseline_map = assign_requests(
        baseline_rounds[baseline["round"]], prompts, baseline["tokens"]
    )
    speculative_map = assign_requests(
        speculative_rounds[speculative["round"]], prompts, speculative["tokens"]
    )
    baseline_emitted = {
        index: emitted_tokens(baseline_rounds[baseline["round"]][req])
        for index, req in baseline_map.items()
    }
    speculative_emitted = {
        index: emitted_tokens(speculative_rounds[speculative["round"]][req])
        for index, req in speculative_map.items()
    }

    replay_cases = {}
    if args.replay is not None:
        for case in json.loads(args.replay.read_text())["cases"]:
            replay_cases[case["case"]] = case

    divergences = []
    expected_streams = speculative.get("expected_tokens", baseline["tokens"])
    for index, prompt in enumerate(prompts):
        position = first_divergence(
            expected_streams[index], speculative["tokens"][index]
        )
        if position is None:
            continue
        absolute = len(prompt) + position
        baseline_row = baseline_emitted[index].get(absolute)
        speculative_row = speculative_emitted[index].get(absolute)
        entry = {
            "request": index,
            "output_position": position,
            "absolute_position": absolute,
            "baseline_token": expected_streams[index][position],
            "speculative_token": speculative["tokens"][index][position],
        }
        if baseline_row is None or speculative_row is None:
            entry["label"] = "untraced"
        else:
            entry.update(
                classify_pair(
                    baseline_row,
                    speculative_row,
                    entry["baseline_token"],
                    entry["speculative_token"],
                    mantissa,
                )
            )
        replay = replay_cases.get(index)
        if replay is not None:
            native = {
                item["chunk_size_after_prompt"]: item["greedy_token"]
                for item in replay["native_target_only"]
            }
            entry["native_greedy_by_chunk"] = native
            agree_baseline = sum(
                token == entry["baseline_token"] for token in native.values()
            )
            agree_speculative = sum(
                token == entry["speculative_token"] for token in native.values()
            )
            entry["native_agrees_with"] = (
                "baseline"
                if agree_baseline > agree_speculative
                else "speculative"
                if agree_speculative > agree_baseline
                else "split"
            )
        divergences.append(entry)

    # Same engine, identical prompts, same prefix, different logits.
    self_divergences = []
    for name, streams, emitted in (
        ("baseline", baseline["tokens"], baseline_emitted),
        ("speculative", speculative["tokens"], speculative_emitted),
    ):
        for left, right in itertools.combinations(range(len(prompts)), 2):
            if prompts[left] != prompts[right]:
                continue
            position = first_divergence(streams[left], streams[right])
            if position is None:
                continue
            absolute = len(prompts[left]) + position
            left_row = emitted[left].get(absolute)
            right_row = emitted[right].get(absolute)
            entry = {
                "engine": name,
                "requests": [left, right],
                "output_position": position,
                "tokens": [streams[left][position], streams[right][position]],
            }
            if left_row is not None and right_row is not None:
                entry.update(
                    classify_pair(
                        left_row,
                        right_row,
                        entry["tokens"][0],
                        entry["tokens"][1],
                        mantissa,
                    )
                )
            else:
                entry["label"] = "untraced"
            self_divergences.append(entry)

    labels = [entry["label"] for entry in divergences]
    summary = {
        "run_dir": str(args.run_dir),
        "width": args.width,
        "logits_dtype": speculative_trace.get("logits_dtype"),
        "max_tie_ulps": MAX_TIE_ULPS,
        "requests": len(prompts),
        "divergent_requests": len(divergences),
        "labels": {label: labels.count(label) for label in sorted(set(labels))},
        "divergences": divergences,
        "self_divergences": self_divergences,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{len(divergences)} divergent request(s): {summary['labels']}")
    for entry in divergences:
        line = (
            f"  request {entry['request']} pos {entry['output_position']}: "
            f"baseline {entry['baseline_token']} vs speculative {entry['speculative_token']} "
            f"-> {entry['label']}"
        )
        if entry.get("native_agrees_with"):
            line += f" (native agrees with {entry['native_agrees_with']})"
        print(line)
    for entry in self_divergences:
        print(
            f"  self-divergence [{entry['engine']}] requests {entry['requests']} "
            f"pos {entry['output_position']}: {entry['tokens']} -> {entry['label']}"
        )


if __name__ == "__main__":
    main()
