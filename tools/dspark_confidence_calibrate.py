# SPDX-License-Identifier: Apache-2.0
"""Record DSpark confidence outcomes on the real pair and fit STS (M6a).

``record`` runs the pinned pair offline at a fixed width on a deterministic
calibration corpus: prompt windows cut from the repository's own documentation
at the recorded git revision plus the natural prompts, generated in one
sampling mode per run (``--mode greedy`` or ``--mode stochastic`` with the
recorded temperature/top-p). Every verified proposal contributes its raw
confidence logits and survival labels for the scheduled positions; clipped
positions are censored and finished requests record nothing. Samples are
split by prompt into calibration and holdout halves (even/odd prompt index),
never by position.

``fit`` fits one temperature per position on the calibration split by
sequential temperature scaling over a fixed grid and reports per-position
ECE (with a bootstrap 95% interval), Brier and reliability bins before and
after fitting on both splits, then writes the artifact with the grid, bin
definition, objective, sample counts, dataset revision, split sizes and the
model-pair manifest. ``report`` prints an existing artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from vllm_metal.v1.dspark.calibration import (
    DEFAULT_BINS,
    DEFAULT_GRID,
    CalibrationArtifact,
    CalibrationManifest,
    ConfidenceRecorder,
    ModeCalibration,
    fit_sequential_temperatures,
    per_position_metrics,
    sample_matrices,
)

CORPUS_FILES = (
    "docs/design/dspark.md",
    "docs/design/dspark-validation.md",
    "docs/design/dspark-handoff.md",
    "docs/speculative_decoding.md",
    "docs/index.md",
    "docs/configuration.md",
)
MODES = {
    "greedy": {"temperature": 0.0},
    "stochastic": {"temperature": 0.8, "top_p": 0.95},
}


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


def build_corpus(
    tokenizer, repo: Path, *, prompts: int, min_tokens: int, max_tokens: int
) -> list[list[int]]:
    """Deterministic prompt windows from the documentation plus natural prompts."""
    from tools.dspark_memory_check import NATURAL_PROMPTS

    text = "\n\n".join(
        (repo / name).read_text(encoding="utf-8") for name in CORPUS_FILES
    )
    tokens = tokenizer.encode(text, add_special_tokens=False)
    windows: list[list[int]] = [
        tokenizer.encode(item, add_special_tokens=False) for item in NATURAL_PROMPTS
    ]
    step = max(1, (len(tokens) - max_tokens) // max(1, prompts))
    for index in range(prompts):
        start = (index * step) % max(1, len(tokens) - max_tokens)
        length = min_tokens + (index * 37) % (max_tokens - min_tokens + 1)
        windows.append(tokens[start : start + length])
    return windows[:prompts]


def record_worker(config: dict, output: Path) -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=config["target"],
        max_model_len=config["max_model_len"],
        max_num_seqs=config["max_num_seqs"],
        max_num_batched_tokens=config["max_num_batched_tokens"],
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        disable_log_stats=True,
        seed=0,
        speculative_config={
            "method": "dspark",
            "model": config["draft"],
            "num_speculative_tokens": config["width"],
        },
    )
    tokenizer = llm.get_tokenizer()
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    proposer = runner._drafter
    manifest = CalibrationManifest.from_runner(runner)
    corpus = build_corpus(
        tokenizer,
        Path(config["repo"]),
        prompts=config["prompts"],
        min_tokens=config["min_prompt_tokens"],
        max_tokens=config["max_prompt_tokens"],
    )
    sampling = MODES[config["mode"]]
    splits = {"calibration": [], "holdout": []}
    for index, prompt in enumerate(corpus):
        splits["calibration" if index % 2 == 0 else "holdout"].append(prompt)
    counts = {}
    for split, prompts in splits.items():
        recorder = ConfidenceRecorder(output.with_name(f"{output.stem}.{split}.jsonl"))
        if recorder.path is not None and recorder.path.exists():
            recorder.path.unlink()
        proposer.recorder = recorder
        params = SamplingParams(
            max_tokens=config["output_length"], ignore_eos=True, **sampling
        )
        llm.generate(
            [{"prompt_token_ids": ids} for ids in prompts],
            [params] * len(prompts),
            use_tqdm=False,
        )
        # Drain: the last proposals of every request were verified at their
        # final step, which never schedules again, so they are censored.
        proposer.recorder = None
        counts[split] = recorder.flush()
    payload = {
        "config": config,
        "manifest": manifest.to_dict(),
        "sampling": sampling,
        "prompts": {split: len(items) for split, items in splits.items()},
        "samples": counts,
        "corpus_files": list(CORPUS_FILES),
        "corpus_revision": git_revision(Path(config["repo"])),
        "corpus_sha256": hashlib.sha256(json.dumps(corpus).encode("utf-8")).hexdigest(),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")


def fit_mode(
    record_json: Path, *, grid: list[float], bins: int
) -> tuple[ModeCalibration, dict]:
    payload = json.loads(record_json.read_text())
    block_size = int(payload["manifest"]["block_size"])
    splits = {
        split: ConfidenceRecorder.load(
            record_json.with_name(f"{record_json.stem}.{split}.jsonl")
        )
        for split in ("calibration", "holdout")
    }
    matrices = {
        split: sample_matrices(samples, block_size) for split, samples in splits.items()
    }
    logits, survival, valid = matrices["calibration"]
    temperatures = fit_sequential_temperatures(logits, survival, valid, grid=grid)
    identity = [1.0] * block_size
    metrics = {}
    for split, (l_, s_, v_) in matrices.items():
        metrics[split] = {
            "before": per_position_metrics(
                l_, s_, v_, identity, bins=bins, bootstrap=True
            ),
            "after": per_position_metrics(
                l_, s_, v_, temperatures, bins=bins, bootstrap=True
            ),
            "samples": int(l_.shape[0]),
        }
    counts = [int(valid[:, position].sum()) for position in range(block_size)]
    mode = ModeCalibration(
        mode=str(payload["config"]["mode"]),
        temperatures=temperatures,
        sample_counts=counts,
        metrics=metrics,
        recorded_sampling=dict(payload["sampling"]),
    )
    return mode, payload


def summarize(artifact: CalibrationArtifact) -> str:
    lines = []
    for name, item in artifact.modes.items():
        lines.append(
            f"mode {name}: temperatures {[round(t, 3) for t in item.temperatures]}, "
            f"samples per position {item.sample_counts}"
        )
        for split in ("calibration", "holdout"):
            block = item.metrics.get(split, {})
            for stage in ("before", "after"):
                rows = block.get(stage, [])
                ece = [
                    f"{row['ece']:.3f}" if row.get("ece") is not None else "-"
                    for row in rows
                ]
                brier = [
                    f"{row['brier']:.3f}" if row.get("brier") is not None else "-"
                    for row in rows
                ]
                lines.append(f"  {split} {stage}: ECE {ece} Brier {brier}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="run the engine and record confidence outcomes")
    rec.add_argument("--target", type=Path, required=True)
    rec.add_argument("--draft", type=Path, required=True)
    rec.add_argument(
        "--output", type=Path, required=True, help="record JSON (splits beside it)"
    )
    rec.add_argument("--mode", choices=sorted(MODES), default="greedy")
    rec.add_argument("--repo", type=Path, default=Path("."))
    rec.add_argument("--width", type=int, default=7)
    rec.add_argument("--prompts", type=int, default=240)
    rec.add_argument("--min-prompt-tokens", type=int, default=32)
    rec.add_argument("--max-prompt-tokens", type=int, default=256)
    rec.add_argument("--output-length", type=int, default=128)
    rec.add_argument("--max-model-len", type=int, default=512)
    rec.add_argument("--max-num-seqs", type=int, default=16)
    rec.add_argument("--max-num-batched-tokens", type=int, default=512)
    rec.add_argument("--memory-fraction", default="0.22")
    rec.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    fit = sub.add_parser(
        "fit", help="fit STS from recorded modes and write the artifact"
    )
    fit.add_argument("--records", type=Path, nargs="+", required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--bins", type=int, default=DEFAULT_BINS)
    fit.add_argument("--grid", default=",".join(str(v) for v in DEFAULT_GRID))
    rep = sub.add_parser("report", help="print an artifact")
    rep.add_argument("artifact", type=Path)
    args = parser.parse_args()

    if args.command == "record":
        if args.worker_config:
            record_worker(json.loads(args.worker_config.read_text()), args.output)
            return
        config = {
            "target": str(args.target.resolve()),
            "draft": str(args.draft.resolve()),
            "repo": str(args.repo.resolve()),
            "mode": args.mode,
            "width": args.width,
            "prompts": args.prompts,
            "min_prompt_tokens": args.min_prompt_tokens,
            "max_prompt_tokens": args.max_prompt_tokens,
            "output_length": args.output_length,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        worker_config = args.output.with_suffix(".config.json")
        worker_config.write_text(json.dumps(config, indent=2) + "\n")
        env = dict(
            os.environ,
            VLLM_USE_V2_MODEL_RUNNER="0",
            VLLM_ENABLE_V1_MULTIPROCESSING="0",
            VLLM_METAL_USE_PAGED_ATTENTION="1",
            VLLM_METAL_MEMORY_FRACTION=str(args.memory_fraction),
            VLLM_METAL_BUILD_FROM_SOURCE="1",
            HF_HUB_OFFLINE="1",
        )
        with args.output.with_suffix(".log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.dspark_confidence_calibrate",
                    "record",
                    "--target",
                    str(args.target),
                    "--draft",
                    str(args.draft),
                    "--output",
                    str(args.output),
                    "--worker-config",
                    str(worker_config),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=7200,
            )
        payload = json.loads(args.output.read_text())
        print(f"recorded {payload['samples']} samples for mode {args.mode}")
        return

    if args.command == "fit":
        grid = [float(v) for v in args.grid.split(",") if v]
        modes = {}
        manifest = None
        dataset: dict = {"records": []}
        for record in args.records:
            mode, payload = fit_mode(record, grid=grid, bins=args.bins)
            current = CalibrationManifest.from_dict(payload["manifest"])
            if manifest is None:
                manifest = current
            elif manifest != current:
                parser.error(f"{record} belongs to another model pair")
            modes[mode.mode] = mode
            dataset["records"].append(
                {
                    "mode": mode.mode,
                    "corpus_files": payload["corpus_files"],
                    "corpus_revision": payload["corpus_revision"],
                    "corpus_sha256": payload["corpus_sha256"],
                    "prompts": payload["prompts"],
                    "samples": payload["samples"],
                    "width": payload["config"]["width"],
                    "output_length": payload["config"]["output_length"],
                }
            )
        assert manifest is not None
        artifact = CalibrationArtifact(
            manifest=manifest, modes=modes, grid=grid, bins=args.bins, dataset=dataset
        )
        artifact.validate(manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(artifact.to_json())
        print(summarize(artifact))
        print(f"wrote {args.output}")
        return

    artifact = CalibrationArtifact.load(args.artifact)
    print(summarize(artifact))


if __name__ == "__main__":
    main()
