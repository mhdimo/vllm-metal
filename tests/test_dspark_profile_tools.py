# SPDX-License-Identifier: Apache-2.0
"""The DSpark profiling tools' pure helpers; no engine or model download."""

from __future__ import annotations

import pytest

from tools import dspark_tooling
from tools.dspark_confidence_calibrate import MODES
from tools.dspark_cost_profile import step_gaps
from tools.dspark_step_profile import summarize


class WordTokenizer:
    """Deterministic stand-in: one id per distinct whitespace-separated word."""

    def __init__(self) -> None:
        self.ids: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [self.ids.setdefault(word, len(self.ids)) for word in text.split()]


@pytest.fixture
def repo(tmp_path):
    for index, name in enumerate(dspark_tooling.CORPUS_FILES):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(" ".join(f"w{index}_{n}" for n in range(600)))
    return tmp_path


@pytest.mark.parametrize("length", [3, 64, 300])
def test_exact_length_prompts_have_the_requested_length(repo, length):
    tokenizer = WordTokenizer()
    prompts = dspark_tooling.exact_length_prompts(tokenizer, repo, length, 12)
    assert len(prompts) == 12
    assert all(len(prompt) == length for prompt in prompts)
    again = dspark_tooling.exact_length_prompts(WordTokenizer(), repo, length, 12)
    assert [len(p) for p in again] == [len(p) for p in prompts]


def test_exact_length_prompts_end_with_a_natural_prompt(repo):
    tokenizer = WordTokenizer()
    prompts = dspark_tooling.exact_length_prompts(tokenizer, repo, 200, 3)
    for index, prompt in enumerate(prompts):
        question = tokenizer.encode(dspark_tooling.NATURAL_PROMPTS[index])
        assert prompt[-len(question) :] == question


def test_exact_length_prompts_reject_empty_requests(repo):
    with pytest.raises(ValueError):
        dspark_tooling.exact_length_prompts(WordTokenizer(), repo, 0, 4)


def test_corpus_windows_start_with_natural_prompts_and_respect_bounds(repo):
    tokenizer = WordTokenizer()
    prompts = 40
    windows = dspark_tooling.corpus_windows(
        tokenizer, repo, prompts=prompts, min_tokens=16, max_tokens=64
    )
    assert len(windows) == prompts
    naturals = len(dspark_tooling.NATURAL_PROMPTS)
    assert windows[:naturals] == [
        tokenizer.encode(item) for item in dspark_tooling.NATURAL_PROMPTS
    ]
    assert all(16 <= len(window) <= 64 for window in windows[naturals:])


def test_corpus_windows_reject_a_corpus_shorter_than_the_window(tmp_path):
    for name in dspark_tooling.CORPUS_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("tiny")
    with pytest.raises(ValueError, match="shorter"):
        dspark_tooling.corpus_windows(
            WordTokenizer(), tmp_path, prompts=4, min_tokens=8, max_tokens=64
        )


def test_step_gaps_keep_only_steps_where_every_request_decodes():
    # Request 0 streams from t=0, request 1 from t=2; both end at t=6 and t=5.
    arrivals = [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 3.5, 5.0]]
    gaps = step_gaps(arrivals)
    # Window is [2.0, 5.0]: request 0 contributes 2->3, 3->4, 4->5; request 1 both gaps.
    assert sorted(gaps) == [1.0, 1.0, 1.0, 1.5, 1.5]


def test_summarize_reports_percentiles():
    values = [float(v) for v in range(1, 101)]
    result = summarize(values)
    assert result["count"] == 100
    assert result["p50"] in (50.0, 51.0)
    assert result["p95"] in (95.0, 96.0)
    assert result["max"] == 100.0
    assert summarize([]) == {"count": 0}


def test_calibration_modes_cover_greedy_and_stochastic():
    assert MODES["greedy"] == {"temperature": 0.0}
    assert MODES["stochastic"]["temperature"] > 0
