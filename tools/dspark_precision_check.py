# SPDX-License-Identifier: Apache-2.0
"""Sequential full-checkpoint DSpark precision check against pinned DeepSpec.

Capture bounded features from the real target, unload it, then run the official
Torch BF16 drafter, MLX BF16 and MLX affine-4/group-64 in separate processes.
This avoids holding target/reference/draft weight copies together. Fixtures are
numerical probes, not calibration data or a representative throughput workload.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from tools.dspark_lifecycle_check import PROMPTS

REFERENCE_SHA = "005e03b81cec38b7da6399833d609ee89a2587f2"
# Freeze before looking at results: five BF16 layers plus projections/norms
# may accumulate rounding error; tiny FP32 comparisons retain their 1e-5 gate.
BF16_RELATIVE_L2_LIMIT = 0.05
MIN_QUANTIZED_ACCEPTANCE_RATIO = 0.8


def snapshot_revision(path: Path) -> str | None:
    # HF cache snapshots carry the resolved commit. An exported local directory
    # name does not establish a revision; retain that fact as unavailable.
    name = path.name
    return (
        name if len(name) == 40 and all(c in "0123456789abcdef" for c in name) else None
    )


def accepted(tokens, expected) -> int:
    for index, (actual, target) in enumerate(zip(tokens, expected, strict=True)):
        if actual != target:
            return index
    return len(tokens)


def native_tokens(model, adapter, ids, count, *, cache=None):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model) if cache is None else cache
    outputs = []
    for _ in range(count):
        result = adapter.target_forward(
            model,
            mx.array(ids)[None],
            cache=cache,
            logits_indices=mx.array([len(ids) - 1]),
        )
        token = int(
            mx.argmax(result.logits.reshape(-1, result.logits.shape[-1])[-1]).item()
        )
        outputs.append(token)
        ids = [token]
    return outputs


def capture(config: dict, folder: Path) -> None:
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model
    from transformers import AutoTokenizer

    from vllm_metal.v1.dspark.config import DSparkConfig
    from vllm_metal.v1.model_adapter import DefaultModelAdapter

    model, _ = load_model(Path(config["target"]), lazy=False)
    tokenizer = AutoTokenizer.from_pretrained(config["target"], local_files_only=True)
    draft = DSparkConfig.from_json(Path(config["draft"]) / "config.json")
    adapter = DefaultModelAdapter()
    cases = []
    for prompt_index, prompt in enumerate(PROMPTS):
        # Fixed lengths exercise single/long contexts and the 256-token buffer
        # boundary. Repetition supplies deterministic stress fixtures only.
        source = tokenizer.encode((prompt + "\n") * 128, add_special_tokens=False)
        for length in (8, 64, 256, 512):
            ids = source[:length]
            cache = make_prompt_cache(model)
            result = adapter.target_forward(
                model,
                mx.array(ids)[None],
                cache=cache,
                logits_indices=mx.array([length - 1]),
                capture_layer_ids=draft.target_layer_ids,
            )
            mx.eval(result.hidden_states, result.logits)
            anchor = int(
                mx.argmax(result.logits.reshape(-1, draft.vocab_size)[-1]).item()
            )
            targets = native_tokens(
                model, adapter, [anchor], draft.block_size, cache=cache
            )
            features = np.asarray(result.hidden_states.astype(mx.float32))[None]
            name = f"p{prompt_index}-n{length}"
            np.savez_compressed(
                folder / f"{name}.features.npz",
                features=features,
                anchor=np.array(anchor),
                targets=np.array(targets),
            )
            cases.append(
                {
                    "name": name,
                    "context_length": length,
                    "feature_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
                }
            )
    baseline = json.loads(Path(config["baseline_result"]).read_text())
    for ids, expected in zip(
        baseline["prompt_token_ids"], baseline["tokens"], strict=True
    ):
        actual = native_tokens(model, adapter, ids, len(expected))
        assert actual == expected, (
            "vLLM baseline differs from native mlx-lm greedy tokens"
        )
    (folder / "fixtures.json").write_text(
        json.dumps(
            {
                "cases": cases,
                "native_target_tokens_equal": True,
                "peak_mlx_bytes": mx.get_peak_memory(),
            },
            indent=2,
        )
        + "\n"
    )


def reference(config: dict, folder: Path) -> None:
    import torch
    from transformers import DynamicCache, Qwen3Config

    sys.path.insert(0, config["deepspec_checkout"])
    from deepspec.modeling.dspark.qwen3.modeling import Qwen3DSparkModel

    torch.set_num_threads(4)
    cfg = Qwen3Config.from_pretrained(config["draft"], local_files_only=True)
    model, info = Qwen3DSparkModel.from_pretrained(
        config["draft"],
        config=cfg,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
        output_loading_info=True,
    )
    assert not any(
        info[key]
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ), info
    model.eval()
    rows = []
    for case in json.loads((folder / "fixtures.json").read_text())["cases"]:
        data = np.load(folder / f"{case['name']}.features.npz", allow_pickle=False)
        features = torch.from_numpy(data["features"]).to(torch.bfloat16)
        anchor = int(data["anchor"])
        ids = torch.tensor([[anchor] + [cfg.mask_token_id] * (cfg.block_size - 1)])
        previous = torch.tensor([[anchor] + data["targets"][:-1].tolist()])
        cache = DynamicCache()
        with torch.inference_mode():
            hidden = model._forward_backbone(
                target_hidden_states=features,
                noise_embedding=model.embed_tokens(ids),
                position_ids=torch.arange(features.shape[1] + cfg.block_size)[None],
                past_key_values=cache,
                use_cache=True,
                is_causal=False,
            )
            cache.crop(features.shape[1])
            logits = model.compute_logits(hidden)
            tokens, _ = model.sample_draft_tokens(
                logits,
                first_prev_token_ids=torch.tensor([anchor]),
                hidden_states=hidden,
                temperature=0.0,
            )
            arrays = {
                "embedding": model.embed_tokens(ids),
                "fused": model.hidden_norm(model.fc(features)),
                "hidden": hidden,
                "logits": logits,
                "corrected": logits
                + model.markov_head.compute_step_bias(previous, None),
                "confidence": model.predict_confidence_step(
                    hidden, prev_token_ids=previous
                ),
            }
            for index, layer in enumerate(cache.layers):
                arrays[f"k{index}"] = layer.keys
                arrays[f"v{index}"] = layer.values
            values = {
                name: value.float().cpu().numpy() for name, value in arrays.items()
            }
        np.savez_compressed(folder / f"{case['name']}.torch.npz", **values)
        proposal = tokens[0].tolist()
        rows.append(
            {
                "case": case["name"],
                "tokens": proposal,
                "accepted": accepted(proposal, data["targets"].tolist()),
            }
        )
    (folder / "torch.json").write_text(
        json.dumps(
            {
                "cases": rows,
                "runtime": {
                    package: importlib.metadata.version(package)
                    for package in ("torch", "transformers", "safetensors")
                },
            },
            indent=2,
        )
        + "\n"
    )


def mlx_check(config: dict, folder: Path, *, quantize: bool) -> None:
    import mlx.core as mx

    from vllm_metal.v1.dspark.loader import load_drafter

    model, cfg = load_drafter(config["draft"], quantize=quantize)
    rows = []
    for case in json.loads((folder / "fixtures.json").read_text())["cases"]:
        data = np.load(folder / f"{case['name']}.features.npz", allow_pickle=False)
        features = mx.array(data["features"]).astype(model.hidden_norm.weight.dtype)
        anchor = int(data["anchor"])
        ids = mx.array([[anchor] + [cfg.mask_token_id] * (cfg.block_size - 1)])
        previous = mx.array([[anchor] + data["targets"][:-1].tolist()])
        cache = model.make_ctx_cache(512)
        # Incremental ingestion also exercises chunk storage and RoPE offsets.
        split = min(17, features.shape[1])
        model.update_context(features[:, :split], 0, cache)
        if split < features.shape[1]:
            model.update_context(features[:, split:], split, cache)
        hidden = model.backbone(model.embed(ids), features.shape[1], cache)
        logits = model.compute_logits(hidden)
        proposal = model.sample_block(logits[0], anchor)
        arrays = {
            "embedding": model.embed(ids),
            "fused": model.fuse_target(features),
            "hidden": hidden,
            "logits": logits,
            "corrected": logits + model.markov_head.step_bias(previous),
            "confidence": model.confidence_logits(hidden, previous),
        }
        for index, layer in enumerate(cache):
            arrays[f"k{index}"] = layer.k
            arrays[f"v{index}"] = layer.v
        mx.eval(arrays, proposal)
        stats = {}
        with np.load(
            folder / f"{case['name']}.torch.npz", allow_pickle=False
        ) as expected:
            for name, value in arrays.items():
                assert value.shape == expected[name].shape, (case["name"], name)
                actual = np.asarray(value.astype(mx.float32)).reshape(-1)
                target = expected[name].reshape(-1)
                assert (
                    actual.shape == target.shape
                    and np.isfinite(actual).all()
                    and np.isfinite(target).all()
                )
                error = actual.astype(np.float64) - target
                relative = float(
                    np.linalg.norm(error)
                    / max(1.0, np.linalg.norm(target.astype(np.float64)))
                )
                stats[name] = {
                    "relative_l2": relative,
                    "max_abs": float(np.abs(error).max()),
                }
                if not quantize:
                    assert relative <= BF16_RELATIVE_L2_LIMIT, (
                        case["name"],
                        name,
                        stats[name],
                    )
        tokens = proposal.tolist()
        rows.append(
            {
                "case": case["name"],
                "tokens": tokens,
                "accepted": accepted(tokens, data["targets"].tolist()),
                "errors_vs_official_bf16": stats,
            }
        )
    (folder / ("affine4.json" if quantize else "bf16.json")).write_text(
        json.dumps(
            {
                "cases": rows,
                "peak_mlx_bytes": mx.get_peak_memory(),
                "runtime": {
                    package: importlib.metadata.version(package)
                    for package in ("mlx", "mlx-lm")
                },
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--deepspec-checkout", type=Path)
    parser.add_argument("--reference-python-path", type=Path)
    parser.add_argument("--baseline-result", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--phase",
        choices=("capture", "torch", "bf16", "affine4"),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.worker_config:
        config = json.loads(args.worker_config.read_text())
        folder = args.worker_config.parent
        if args.phase == "capture":
            capture(config, folder)
        elif args.phase == "torch":
            reference(config, folder)
        else:
            mlx_check(config, folder, quantize=args.phase == "affine4")
        return
    if any(
        path is None
        for path in (
            args.target,
            args.draft,
            args.deepspec_checkout,
            args.baseline_result,
            args.output_dir,
        )
    ):
        parser.error(
            "target, draft, official checkout, baseline result and output directory are required"
        )
    revision = subprocess.check_output(
        ["git", "-C", str(args.deepspec_checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REFERENCE_SHA:
        parser.error(f"official reference must be pinned to {REFERENCE_SHA}")
    subprocess.run(
        [
            "git",
            "-C",
            str(args.deepspec_checkout),
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "*.py",
        ],
        check=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        key: str(getattr(args, key).resolve())
        for key in ("target", "draft", "deepspec_checkout", "baseline_result")
    }
    path = args.output_dir / "config.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    (args.output_dir / "gates.json").write_text(
        json.dumps(
            {
                "bf16_relative_l2_limit": BF16_RELATIVE_L2_LIMIT,
                "minimum_affine4_acceptance_ratio": MIN_QUANTIZED_ACCEPTANCE_RATIO,
                "reference_sha": REFERENCE_SHA,
                "target_revision": snapshot_revision(args.target),
                "draft_revision": snapshot_revision(args.draft),
            },
            indent=2,
        )
        + "\n"
    )
    for phase in ("capture", "torch", "bf16", "affine4"):
        env = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_METAL_BUILD_FROM_SOURCE="1")
        if phase == "torch" and args.reference_python_path:
            env["PYTHONPATH"] = str(args.reference_python_path.resolve())
        with (args.output_dir / f"{phase}.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.dspark_precision_check",
                    "--worker-config",
                    str(path),
                    "--phase",
                    phase,
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1200,
            )
        print(f"PASS: {phase}", flush=True)
    bf16 = json.loads((args.output_dir / "bf16.json").read_text())
    quantized = json.loads((args.output_dir / "affine4.json").read_text())
    full_accept = sum(row["accepted"] for row in bf16["cases"])
    quant_accept = sum(row["accepted"] for row in quantized["cases"])
    assert (
        full_accept > 0 and quant_accept >= MIN_QUANTIZED_ACCEPTANCE_RATIO * full_accept
    ), (full_accept, quant_accept)
    (args.output_dir / "result.json").write_text(
        json.dumps(
            {
                "passed": True,
                "fixture_count": len(bf16["cases"]),
                "bf16_accepted": full_accept,
                "affine4_accepted": quant_accept,
                "acceptance_ratio": quant_accept / full_accept,
                "native_target_tokens_equal": True,
                "reference_sha": REFERENCE_SHA,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "PASS: full-checkpoint numerical and bounded quantization-trace gates",
        flush=True,
    )


if __name__ == "__main__":
    main()
