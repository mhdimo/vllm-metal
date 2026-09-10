# SPDX-License-Identifier: Apache-2.0
"""Compare tiny standalone DSpark models with a pinned DeepSpec checkout.

Run as a module from the repository root. This downloads nothing and does not
exercise a real target model, quantization, or the serving scheduler. See
docs/design/dspark-validation.md for the required larger experiments.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import torch

REFERENCE_COMMIT = "005e03b81cec38b7da6399833d609ee89a2587f2"


def _configuration(family: str):
    from transformers import Gemma4TextConfig, Qwen3Config

    common = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "tie_word_embeddings": False,
        "layer_types": ["full_attention"] * 2,
        "max_position_embeddings": 128,
    }
    if family == "qwen3":
        cfg = Qwen3Config(**common, rope_theta=1_000_000.0)
    else:
        cfg = Gemma4TextConfig(
            **common,
            global_head_dim=8,
            num_global_key_value_heads=1,
            attention_k_eq_v=True,
            enable_moe_block=False,
            hidden_size_per_layer_input=0,
            num_kv_shared_layers=0,
            use_double_wide_mlp=False,
            hidden_activation="gelu_pytorch_tanh",
            final_logit_softcapping=30.0,
            rope_parameters={
                "full_attention": {
                    "rope_type": "proportional",
                    "rope_theta": 1_000_000.0,
                    "partial_rotary_factor": 0.25,
                },
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10_000.0,
                },
            },
        )
    for name, value in {
        "target_layer_ids": [0, 2],
        "num_target_layers": 4,
        "block_size": 7,
        "mask_token_id": 63,
        "num_anchors": 2,
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "markov_rank": 8,
        "markov_head_type": "vanilla",
    }.items():
        setattr(cfg, name, value)
    cfg._attn_implementation = "eager"
    return cfg


def _max_error(reference: torch.Tensor, actual: mx.array) -> float:
    expected = reference.detach().numpy()
    observed = np.asarray(actual)
    if not (np.isfinite(expected).all() and np.isfinite(observed).all()):
        raise AssertionError("Reference comparisons require finite tensors")
    np.testing.assert_allclose(
        observed, expected, atol=1e-5, rtol=1e-5, equal_nan=False
    )
    return float(np.max(np.abs(expected - observed)))


def check_family(family: str) -> dict:
    from deepspec.modeling.dspark.gemma4.modeling import Gemma4DSparkModel
    from deepspec.modeling.dspark.qwen3.modeling import Qwen3DSparkModel
    from transformers import DynamicCache
    from vllm.sampling_params import SamplingParams

    from vllm_metal.v1.dspark.config import DSparkConfig
    from vllm_metal.v1.dspark.memory import DSparkMemoryPlan
    from vllm_metal.v1.dspark.model import DSparkDrafter
    from vllm_metal.v1.dspark_proposer import (
        DSparkProposer,
        _DraftPlan,
        _RequestContext,
    )
    from vllm_metal.v1.model_runner import RequestState
    from vllm_metal.v1.spec_decode import SpeculativeDecodeController

    cfg = _configuration(family)
    torch.manual_seed(7)
    model_class = Qwen3DSparkModel if family == "qwen3" else Gemma4DSparkModel
    reference = model_class(cfg).float().eval()
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "config.json"
        path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
        port_config = DSparkConfig.from_json(path)
    port = DSparkDrafter(port_config)
    port.load_weights(
        [
            (key, mx.array(value.detach().numpy()))
            for key, value in reference.state_dict().items()
        ],
        strict=True,
    )
    port.eval()
    rng = np.random.default_rng(11)
    cases = []
    ids = np.array([[2] + [63] * 6], dtype="int64")
    for length in (1, 5, 9):
        ref_cache = DynamicCache()
        port_cache = port.make_ctx_cache()
        context = rng.normal(size=(1, length, 64)).astype("float32")
        with torch.no_grad():
            hidden = reference._forward_backbone(
                target_hidden_states=torch.from_numpy(context),
                noise_embedding=reference.embed_tokens(torch.from_numpy(ids)),
                position_ids=torch.arange(length + 7)[None],
                past_key_values=ref_cache,
                use_cache=True,
                is_causal=False,
            )
            ref_cache.crop(length)
            logits = reference.compute_logits(hidden)
            tokens, _ = reference.sample_draft_tokens(
                logits,
                first_prev_token_ids=torch.tensor([2]),
                hidden_states=hidden,
                temperature=0.0,
            )
            confidence = reference.predict_confidence_step(
                hidden,
                prev_token_ids=torch.cat([torch.tensor([[2]]), tokens[:, :-1]], dim=1),
            )
        port.update_context(mx.array(context), 0, port_cache)
        actual_hidden = port.backbone(port.embed(mx.array(ids)), length, port_cache)
        actual_logits = port.compute_logits(actual_hidden)
        actual_tokens = port.sample_block(actual_logits[0], 2)
        actual_confidence = port.confidence_logits(
            actual_hidden,
            mx.concatenate([mx.array([[2]]), actual_tokens[None, :-1]], axis=1),
        )
        mx.eval(actual_hidden, actual_logits, actual_tokens, actual_confidence)
        np.testing.assert_array_equal(tokens[0].numpy(), np.asarray(actual_tokens))
        cases.append(
            {
                "context_length": length,
                "hidden_max_abs": _max_error(hidden, actual_hidden),
                "logits_max_abs": _max_error(logits, actual_logits),
                "confidence_max_abs": _max_error(confidence, actual_confidence),
                "draft_tokens_equal": True,
            }
        )
        if length == 5:
            extra = rng.normal(size=(1, 3, 64)).astype("float32")
            with torch.no_grad():
                hidden = reference._forward_backbone(
                    target_hidden_states=torch.from_numpy(extra),
                    noise_embedding=reference.embed_tokens(torch.from_numpy(ids)),
                    position_ids=torch.arange(5, 15)[None],
                    past_key_values=ref_cache,
                    use_cache=True,
                    is_causal=False,
                )
            port.update_context(mx.array(extra), 5, port_cache)
            actual_hidden = port.backbone(port.embed(mx.array(ids)), 8, port_cache)
            mx.eval(actual_hidden)
            cases.append(
                {"incremental_hidden_max_abs": _max_error(hidden, actual_hidden)}
            )

    plans, expected = [], []
    for length in (2, 6):
        cache = port.make_ctx_cache()
        context = mx.array(rng.normal(size=(1, length, 64)).astype("float32"))
        port.update_context(context, 0, cache)
        hidden = port.backbone(port.embed(mx.array(ids)), length, cache)
        tokens = port.sample_block(port.compute_logits(hidden)[0, :2], 2)
        mx.eval(tokens)
        expected.append(tokens.tolist())
        owner = RequestState(
            token_ids=[2] * (length + 1),
            prompt_len=length,
            cache=[],
            sampling_params=SamplingParams(temperature=0.0),
        )
        plans.append(
            _DraftPlan(str(length), 2, _RequestContext(owner, cache, length), 2)
        )
    proposer = DSparkProposer(
        drafter=port,
        config=port_config,
        runner=SimpleNamespace(),
        controller=SpeculativeDecodeController(),
        memory_plan=DSparkMemoryPlan.build(
            port_config,
            itemsize=4,
            max_num_seqs=2,
            max_model_len=16,
            max_num_batched_tokens=16,
        ),
    )
    _, actual = proposer._batch_draft(plans)
    np.testing.assert_array_equal(actual, expected)
    return {
        "family": family,
        "config": cfg.to_dict(),
        "cases": cases,
        "ragged_tokens_equal": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deepspec-checkout", type=Path, required=True)
    parser.add_argument("--family", choices=("qwen3", "gemma4", "all"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = args.deepspec_checkout.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REFERENCE_COMMIT:
        parser.error(
            f"DeepSpec must be checked out at {REFERENCE_COMMIT}; got {revision}"
        )
    dirty = subprocess.check_output(
        ["git", "-C", str(reference), "diff", "HEAD", "--", "deepspec"], text=True
    )
    if dirty:
        parser.error("DeepSpec reference source has local changes")
    if importlib.metadata.version("transformers") != "5.10.2":
        parser.error("Use an isolated Transformers 5.10.2 reference environment")
    sys.path.insert(0, str(reference))
    families = ("qwen3", "gemma4") if args.family == "all" else (args.family,)
    result = {
        "reference_commit": revision,
        "dtype": "float32",
        "numpy_seed": 11,
        "torch_seed": 7,
        "tolerance": {"atol": 1e-5, "rtol": 1e-5},
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "tokenizers", "mlx", "mlx-lm", "vllm")
        },
        "results": [check_family(family) for family in families],
    }
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"PASS: {', '.join(families)} tiny FP32 reference checks; {args.output}")


if __name__ == "__main__":
    main()
