# SPDX-License-Identifier: Apache-2.0
"""Statistics and workload helpers of the DSpark performance tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.dspark_acceptance_eval import PAPER_ACCEPTED_LENGTH, load_prompts
from tools.dspark_perf_bench import (
    bootstrap_median_interval,
    build_prompts,
    paired_summary,
    percentile,
)
from tools.dspark_step_profile import summarize


def test_percentile_is_nearest_rank_on_sorted_values() -> None:
    values = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 0.5) == 3.0
    assert percentile(values, 1.0) == 5.0
    assert percentile([7.0], 0.95) == 7.0


def test_bootstrap_interval_brackets_the_median_and_is_deterministic() -> None:
    differences = [0.12, 0.15, 0.11, 0.18, 0.14]
    low, high = bootstrap_median_interval(differences, resamples=2000, seed=1)
    assert low <= 0.14 <= high
    assert low >= min(differences) and high <= max(differences)
    assert bootstrap_median_interval(differences, resamples=2000, seed=1) == (low, high)


def _rep(
    tokens_per_s: float, tpot: float, gap95: float, goodput: float, ttft: float
) -> dict:
    return {
        "output_tokens_per_s": tokens_per_s,
        "tpot_median_s": tpot,
        "gap_p95_s": gap95,
        "goodput_requests_per_s": goodput,
        "ttft_mean_s": ttft,
    }


def test_paired_summary_orients_benefit_and_applies_the_gate() -> None:
    reference = [_rep(100.0, 0.020, 0.030, 1.0, 0.5) for _ in range(5)]
    faster = [_rep(125.0, 0.016, 0.024, 1.2, 0.5) for _ in range(5)]
    summary = paired_summary(reference, faster)
    assert summary["output_tokens_per_s"]["median_relative_benefit"] == pytest.approx(
        0.25
    )
    # Lower is better for TPOT: a 20% reduction is a +20% benefit.
    assert summary["tpot_median_s"]["median_relative_benefit"] == pytest.approx(0.20)
    assert summary["output_tokens_per_s"]["gate_10pct_ci_excludes_zero"]
    assert not summary["ttft_mean_s"]["gate_10pct_ci_excludes_zero"]  # unchanged
    slower = [_rep(90.0, 0.024, 0.040, 0.8, 0.6) for _ in range(5)]
    worse = paired_summary(reference, slower)
    assert worse["output_tokens_per_s"]["median_relative_benefit"] == pytest.approx(
        -0.10
    )
    assert not worse["output_tokens_per_s"]["gate_10pct_ci_excludes_zero"]


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [len(word) for word in text.split()]


def test_build_prompts_are_exact_length_and_distinct(tmp_path: Path) -> None:
    (tmp_path / "docs" / "design").mkdir(parents=True)
    words = " ".join(f"w{i}" for i in range(5000))
    for name in (
        "docs/design/dspark.md",
        "docs/design/dspark-validation.md",
        "docs/index.md",
    ):
        (tmp_path / name).write_text(words)
    prompts = build_prompts(_Tokenizer(), tmp_path, 128, 4)
    assert [len(prompt) for prompt in prompts] == [128] * 4
    assert len({tuple(prompt) for prompt in prompts}) == 4
    short = build_prompts(_Tokenizer(), tmp_path, 4, 1)
    assert len(short[0]) == 4


def test_summarize_reports_percentiles_in_seconds() -> None:
    stats = summarize([0.010, 0.020, 0.030, 0.040, 0.100])
    assert stats["count"] == 5
    assert stats["p50"] == 0.030
    assert stats["p95"] == 0.100
    assert stats["max"] == 0.100
    assert summarize([]) == {"count": 0}


def test_acceptance_eval_prompts_are_first_turns_and_a_stable_subset(
    tmp_path: Path,
) -> None:
    path = tmp_path / "set.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"turns": [f"prompt {i}", "second turn"]}) for i in range(10)
        )
        + "\n"
    )
    prompts = load_prompts(path, 4, seed=1)
    assert len(prompts) == 4 and len(set(prompts)) == 4
    assert all(prompt.startswith("prompt ") for prompt in prompts)
    assert prompts == load_prompts(path, 4, seed=1)
    assert prompts != load_prompts(path, 4, seed=2)
    assert len(load_prompts(path, 0, seed=1)) == 10
    # Table 1 rows carried for the record's comparison column.
    assert PAPER_ACCEPTED_LENGTH["Qwen3-4B"]["gsm8k"] == 6.11


# ---- contamination-aware paired repetitions ---------------------------------


def test_heavy_other_processes_counts_only_outside_user_land_load() -> None:
    from tools.dspark_perf_bench import heavy_other_processes

    lines = [
        "  11  98.6 bun /Users/x/scan.ts",  # another session's work: counts
        "  12  45.0 /System/Library/PrivateFrameworks/SkyLight.framework/WindowServer -daemon",
        "  13  41.2 /Applications/Foo.app/Contents/MacOS/Foo",
        "  14  59.5 VLLM::EngineCore",  # our server
        "  15  80.0 /Users/x/.venv/bin/python -m tools.dspark_perf_bench",  # this client (pid)
        "  16  70.0 /Users/x/.venv/bin/python probe.py",  # another python: counts
        "  17  39.9 node build.js",  # below the threshold
        "  18  55.0 node /Users/x/vitest",  # counts
        "bad line",
    ]
    assert heavy_other_processes(lines, {15}) == 3
    assert heavy_other_processes(lines) == 4


class _FakeSampler:
    def __init__(self, load: dict) -> None:
        self._load = load

    def start(self) -> None:
        pass

    def stop(self) -> dict:
        return self._load


def _quiet() -> dict:
    return {
        "samples": 3,
        "load1_max": 2.0,
        "load1_mean": 1.8,
        "heavy_other_max": 0,
        "heavy_other_samples": 0,
    }


def _loaded() -> dict:
    return {
        "samples": 3,
        "load1_max": 5.1,
        "load1_mean": 4.0,
        "heavy_other_max": 1,
        "heavy_other_samples": 2,
    }


def _measure(monkeypatch, loads: list[dict], repetitions: int, max_retries: int):
    from types import SimpleNamespace

    from tools import dspark_perf_bench as bench

    calls: list[str] = []

    async def fake_run(base, prompts, output, slo):
        calls.append(base)
        return {"output_tokens_per_s": 100.0 + len(calls)}

    monkeypatch.setattr(bench, "run_repetition", fake_run)
    samplers = iter(loads)
    measured, attempts = bench.measure_bucket(
        {"k0": SimpleNamespace(base="k0"), "k": SimpleNamespace(base="k")},
        ["k0", "k"],
        [[1, 2]],
        4,
        repetitions,
        {},
        max_load=4.0,
        max_retries=max_retries,
        sampler_factory=lambda: _FakeSampler(next(samplers)),
    )
    return measured, attempts, calls


def test_measure_bucket_discards_and_repeats_a_loaded_pair(monkeypatch) -> None:
    measured, attempts, calls = _measure(
        monkeypatch, [_loaded(), _quiet(), _quiet()], 2, 5
    )
    assert len(measured["k0"]) == 2 and len(measured["k"]) == 2
    assert [a["kept"] for a in attempts] == [False, True, True]
    assert [a["contaminated"] for a in attempts] == [True, False, False]
    # warm-up (2) + discarded pair (2) + two kept pairs (4)
    assert len(calls) == 8
    # kept pairs alternate their order by the number of pairs kept so far
    assert attempts[1]["order"] == ["k0", "k"] and attempts[2]["order"] == ["k", "k0"]
    assert all(not rep["contaminated"] for rep in measured["k0"] + measured["k"])
    assert measured["k0"][0]["load"] == _quiet()


def test_measure_bucket_keeps_and_flags_when_retries_are_spent(monkeypatch) -> None:
    measured, attempts, calls = _measure(
        monkeypatch, [_loaded(), _loaded(), _loaded()], 2, 1
    )
    assert len(measured["k0"]) == 2
    assert [a["kept"] for a in attempts] == [False, True, True]
    assert all(rep["contaminated"] for rep in measured["k"])
    assert len(calls) == 8
