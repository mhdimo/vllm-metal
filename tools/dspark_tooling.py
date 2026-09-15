# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the DSpark profiling tools.

The adaptive DSpark mode needs two per-pair artifacts: a confidence
calibration (``tools/dspark_confidence_calibrate.py``) and a cost model measured
on the serving machine (``tools/dspark_cost_profile.py``, which uses
``tools/dspark_step_profile.py`` for the drafter's own work). This module holds
what those tools share: the natural prompts, deterministic prompt corpora cut
from the repository's documentation, and a ``vllm serve`` launcher.
"""

from __future__ import annotations

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

# Documentation that ships with the repository; the corpus records its revision.
CORPUS_FILES = (
    "docs/speculative_decoding.md",
    "docs/configuration.md",
    "docs/index.md",
    "docs/supported_models.md",
)


def git_revision(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def corpus_tokens(tokenizer, repo: Path) -> list[int]:
    text = "\n\n".join(
        (repo / name).read_text(encoding="utf-8") for name in CORPUS_FILES
    )
    return tokenizer.encode(text, add_special_tokens=False)


def corpus_windows(
    tokenizer, repo: Path, *, prompts: int, min_tokens: int, max_tokens: int
) -> list[list[int]]:
    """The natural prompts, then deterministic documentation windows of varied length."""
    if not 0 < min_tokens <= max_tokens:
        raise ValueError("require 0 < min_tokens <= max_tokens")
    tokens = corpus_tokens(tokenizer, repo)
    if len(tokens) <= max_tokens:
        raise ValueError("the documentation corpus is shorter than max_tokens")
    windows: list[list[int]] = [
        tokenizer.encode(item, add_special_tokens=False) for item in NATURAL_PROMPTS
    ]
    span = len(tokens) - max_tokens
    step = max(1, span // max(1, prompts))
    for index in range(prompts):
        start = (index * step) % span
        length = min_tokens + (index * 37) % (max_tokens - min_tokens + 1)
        windows.append(tokens[start : start + length])
    return windows[:prompts]


def exact_length_prompts(
    tokenizer, repo: Path, input_length: int, count: int
) -> list[list[int]]:
    """``count`` distinct prompts of exactly ``input_length`` tokens.

    Each ends with a natural prompt and is padded in front with a documentation
    window, so the decode that follows is realistic text continuation.
    """
    if input_length <= 0 or count <= 0:
        raise ValueError("input_length and count must be positive")
    filler = corpus_tokens(tokenizer, repo)
    prompts = []
    for index in range(count):
        question = tokenizer.encode(
            NATURAL_PROMPTS[index % len(NATURAL_PROMPTS)], add_special_tokens=False
        )
        need = input_length - len(question)
        if need <= 0:
            prompts.append(question[:input_length])
            continue
        start = (index * 977) % max(1, len(filler) - need)
        prompts.append(filler[start : start + need] + question)
    return prompts


def model_ref(value: str | Path) -> str:
    """Record a model the way the serving engine will report it.

    A local checkpoint is recorded resolved, so two runs started from different
    working directories agree. A Hugging Face repo id is recorded as given:
    resolving it would manufacture a path under the current directory and bind the
    artifact to a pair no server can ever match, and the manifest check compares
    these strings for equality.
    """
    path = Path(value)
    return str(path.resolve()) if path.exists() else str(value)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Server:
    """One ``vllm serve`` process with a health wait and a clean shutdown."""

    def __init__(
        self,
        *,
        target: Path,
        draft: Path,
        width: int,
        log_dir: Path,
        name: str,
        gpu_memory_utilization: float,
        max_model_len: int,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        async_scheduling: bool = True,
        env: dict[str, str] | None = None,
        expect_in_log: str | None = None,
    ) -> None:
        self.name = name
        self.expect_in_log = expect_in_log
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.log_path = log_dir / f"server-{name}.log"
        executable = shutil.which("vllm", path=str(Path(sys.executable).parent))
        command = [
            executable or "vllm",
            "serve",
            str(target),
            "--port",
            str(self.port),
            "--served-model-name",
            "target",
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--max-model-len",
            str(max_model_len),
            "--max-num-seqs",
            str(max_num_seqs),
            "--max-num-batched-tokens",
            str(max_num_batched_tokens),
            "--async-scheduling" if async_scheduling else "--no-async-scheduling",
            "--no-enable-prefix-caching",
            "--seed",
            "0",
        ]
        if width:
            command += [
                "--speculative-config",
                json.dumps(
                    {
                        "method": "dspark",
                        "model": str(draft),
                        "num_speculative_tokens": width,
                    }
                ),
            ]
        # ``vllm serve`` is a console script: put this checkout first on the
        # path, or an editable install of another checkout would be served.
        repo_root = str(Path(__file__).resolve().parents[1])
        process_env = dict(
            os.environ,
            PYTHONPATH=os.pathsep.join(
                [repo_root, *filter(None, [os.environ.get("PYTHONPATH")])]
            ),
            HF_HUB_OFFLINE=os.environ.get("HF_HUB_OFFLINE", "1"),
            **(env or {}),
        )
        self._log = self.log_path.open("w")
        self.process = subprocess.Popen(
            command,
            env=process_env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        with httpx.Client(timeout=5) as client:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"{self.name} exited with {self.process.returncode}; "
                        f"see {self.log_path}"
                    )
                try:
                    if client.get(f"{self.base}/health").status_code == 200:
                        self._check_log()
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(1)
        raise RuntimeError(f"{self.name} did not become healthy; see {self.log_path}")

    def _check_log(self) -> None:
        """Prove the server runs the expected configuration, not a default."""
        if self.expect_in_log is None:
            return
        self._log.flush()
        text = self.log_path.read_text(errors="replace")
        if self.expect_in_log not in text:
            raise RuntimeError(
                f"{self.name} log lacks {self.expect_in_log!r}; the server did not "
                f"start in the expected mode (see {self.log_path})"
            )

    def stop(self) -> None:
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=30)
        self._log.close()
