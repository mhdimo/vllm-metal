# SPDX-License-Identifier: Apache-2.0
"""Metal-owned DSpark validation and factory contracts; no model downloads."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import pytest
from transformers import Qwen3Config
from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig, VllmConfig
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)

from tests.stub_runner import make_stub_runner
from vllm_metal.v1.dspark import loader
from vllm_metal.v1.dspark.config import DSparkConfig
from vllm_metal.v1.dspark.contracts import is_dspark_config, validate_dspark_config
from vllm_metal.v1.dspark.model import DSparkDrafter
from vllm_metal.v1.dspark_proposer import DSparkProposer


def draft_hf_config() -> Qwen3Config:
    return Qwen3Config(
        architectures=["Qwen3DSparkModel"],
        hidden_size=32,
        intermediate_size=64,
        vocab_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        block_size=7,
        mask_token_id=63,
        target_layer_ids=[0, 2],
        num_target_layers=4,
        markov_rank=8,
        markov_head_type="vanilla",
        layer_types=["full_attention"] * 2,
    )


def model_config(hf, *, model: str) -> ModelConfig:
    # The hook consumes upstream's resolved DTO, as in test_platform.py. Model
    # resolution itself belongs to vLLM and is not re-tested with fake weights.
    result = object.__new__(ModelConfig)
    result.hf_config = result.hf_text_config = hf
    result.model = model
    result.revision = "a" * 40
    result.model_arch_config = ModelArchConfigConvertorBase(hf, hf).convert()
    return result


@pytest.fixture
def dspark_config(monkeypatch) -> VllmConfig:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    spec = SpeculativeConfig(method="ngram", num_speculative_tokens=2)
    spec.method = "dspark"
    spec.model = "unused-outer-model"
    spec.draft_model_config = model_config(draft_hf_config(), model="resolved-draft")
    target_hf = Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        hidden_size=32,
        vocab_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    result = object.__new__(VllmConfig)
    result.model_config = model_config(target_hf, model="resolved-target")
    result.speculative_config = spec
    result.parallel_config = ParallelConfig()
    result.lora_config = None
    return result


@pytest.mark.parametrize("method", ["dspark", "draft_model"])
@pytest.mark.parametrize("memory_limit", [10_000_000, 10_000_000_000])
def test_canonical_and_alias_install_resolved_draft(
    dspark_config, monkeypatch, method, memory_limit
):
    dspark_config.speculative_config.method = method
    validate_dspark_config(dspark_config, use_paged_attention=True)
    runner = make_stub_runner(tokenizer=object())
    runner.vllm_config = dspark_config
    cfg = DSparkConfig.from_dict(draft_hf_config().to_dict())
    weights = DSparkDrafter(cfg)
    load = Mock(return_value=(weights, cfg))
    runner.model = Mock(parameters=Mock(return_value={}))
    runner.scheduler_config = SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=32)
    runner.cache_config.gpu_memory_utilization = 0.5
    from vllm_metal.config import MetalConfig

    runner.metal_config = MetalConfig(
        memory_fraction=0.5, mlx_device="gpu", use_paged_attention=True
    )
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": memory_limit}
    )
    monkeypatch.setattr(mx, "get_active_memory", lambda: 1000)
    monkeypatch.setattr("vllm_metal.v1.model_runner.load_drafter", load)
    if memory_limit == 10_000_000:
        with pytest.raises(ValueError, match="context/workspace and target profiling"):
            runner._load_dspark_drafter()
        assert runner._drafter is None and runner._dspark_memory_plan is None
        return
    runner._load_dspark_drafter()
    runner.install_drafter(num_blocks=1, block_size=16)
    load.assert_called_once_with(
        "resolved-draft",
        revision="a" * 40,
        quantize=True,
        memory_budget_bytes=4_999_999_000,
        expected_config=cfg,
    )
    assert isinstance(runner._drafter, DSparkProposer)
    assert runner._drafter._drafter is weights


@pytest.mark.parametrize(
    "option,value",
    [
        ("enable_adaptive_verification", True),
        ("draft_sample_method", "probabilistic"),
        ("rejection_sample_method", "synthetic"),
        ("dspark_draft_topk", 8),
        ("quantization", "fp8"),
        ("kv_cache_dtype", "fp8"),
        ("max_model_len", 64),
        ("attention_backend", "FLASH_ATTN"),
        ("draft_load_config", object()),
        ("disable_padded_drafter_batch", True),
        ("use_local_argmax_reduction", True),
        ("draft_tensor_parallel_size", 2),
    ],
)
def test_inert_options_are_rejected(dspark_config, option, value):
    setattr(dspark_config.speculative_config, option, value)
    with pytest.raises(NotImplementedError, match=option):
        validate_dspark_config(dspark_config, use_paged_attention=True)


@pytest.mark.parametrize("value", [None, "1"])
def test_gpu_runner_selection_rejected(dspark_config, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER")
    else:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", value)
    with pytest.raises(NotImplementedError, match="VLLM_USE_V2_MODEL_RUNNER=0"):
        validate_dspark_config(dspark_config, use_paged_attention=True)


@pytest.mark.parametrize("width", [0, 8, 14])
def test_width_cannot_silently_clip(dspark_config, width):
    dspark_config.speculative_config.num_speculative_tokens = width
    with pytest.raises(ValueError, match="num_speculative_tokens"):
        validate_dspark_config(dspark_config, use_paged_attention=True)


@pytest.mark.parametrize("name", ["hidden_size", "vocab_size", "num_hidden_layers"])
def test_pair_geometry_mismatch_rejected(dspark_config, name):
    setattr(dspark_config.model_config.hf_text_config, name, 123)
    with pytest.raises(ValueError, match=name):
        validate_dspark_config(dspark_config, use_paged_attention=True)


def test_unqualified_architecture_rejected(dspark_config):
    dspark_config.speculative_config.draft_model_config.model_arch_config.architectures = [
        "Gemma4DSparkModel"
    ]
    with pytest.raises(NotImplementedError, match="standalone Qwen3DSparkModel"):
        validate_dspark_config(dspark_config, use_paged_attention=True)


def test_nonpaged_and_lora_rejected(dspark_config):
    with pytest.raises(NotImplementedError, match="paged attention"):
        validate_dspark_config(dspark_config, use_paged_attention=False)
    dspark_config.lora_config = object()
    with pytest.raises(NotImplementedError, match="LoRA"):
        validate_dspark_config(dspark_config, use_paged_attention=True)


def test_other_proposers_are_unchanged():
    spec = SpeculativeConfig(method="ngram", num_speculative_tokens=2)
    assert not is_dspark_config(spec)
    validate_dspark_config(
        SimpleNamespace(speculative_config=spec), use_paged_attention=False
    )


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"model_type": "qwen3_5"}, "unsupported drafter family"),
        ({"model_type": "deepseek_v4"}, "unsupported drafter family"),
        ({"markov_head_type": "rnn"}, "vanilla"),
        ({"markov_head_type": "gated"}, "vanilla"),
        ({"log_snr_conditioning": True}, "GIDD"),
        ({"target_layer_ids": [2, 0]}, "strictly increasing"),
        ({"target_layer_ids": [0, 0]}, "strictly increasing"),
        ({"target_layer_ids": [-1, 2]}, "strictly increasing"),
        ({"target_layer_ids": [0, 3]}, "strictly increasing"),
        ({"target_layer_ids": []}, "strictly increasing"),
        ({"mask_token_id": 64}, "vocabulary"),
        ({"block_size": 0}, "positive integer"),
        ({"num_key_value_heads": 0}, "KV heads"),
        ({"num_key_value_heads": 3}, "KV heads"),
        ({"head_dim": 0}, "head dimension"),
        ({"head_dim": 7}, "head dimension"),
        ({"rms_norm_eps": float("nan")}, "finite and positive"),
        ({"use_sliding_window": True}, "sliding or causal"),
        ({"rope_parameters": {"rope_type": "yarn"}}, "full default RoPE"),
    ],
)
def test_unimplemented_or_malformed_checkpoint_rejected(changes, match):
    raw = draft_hf_config().to_dict()
    raw.update(changes)
    with pytest.raises((ValueError, NotImplementedError), match=match):
        DSparkConfig.from_dict(raw)


def test_loader_resolves_requested_revision(monkeypatch, tmp_path):
    resolve = Mock(side_effect=lambda model, **kwargs: model)
    download = Mock(return_value=str(tmp_path))
    monkeypatch.setattr(loader, "get_model_download_path", resolve)
    monkeypatch.setattr(loader, "snapshot_download", download)
    assert loader._resolve("org/draft", revision="b" * 40) == str(tmp_path)
    resolve.assert_called_once_with("org/draft", revision="b" * 40)
    download.assert_called_once_with(
        "org/draft", revision="b" * 40, allow_patterns=["*.json", "*.safetensors"]
    )
    download.reset_mock()
    assert loader._resolve(str(tmp_path), revision="b" * 40) == str(tmp_path)
    download.assert_not_called()


def _loadable_runner(dspark_config, monkeypatch):
    validate_dspark_config(dspark_config, use_paged_attention=True)
    runner = make_stub_runner(tokenizer=object())
    runner.vllm_config = dspark_config
    cfg = DSparkConfig.from_dict(draft_hf_config().to_dict())
    load = Mock(return_value=(DSparkDrafter(cfg), cfg))
    runner.model = Mock(parameters=Mock(return_value={}))
    runner.scheduler_config = SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=32)
    runner.cache_config.gpu_memory_utilization = 0.5
    from vllm_metal.config import MetalConfig

    runner.metal_config = MetalConfig(
        memory_fraction=0.5, mlx_device="gpu", use_paged_attention=True
    )
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 10_000_000_000}
    )
    monkeypatch.setattr(mx, "get_active_memory", lambda: 1000)
    monkeypatch.setattr("vllm_metal.v1.model_runner.load_drafter", load)
    return runner


def test_draft_precision_environment_reaches_the_loader(dspark_config, monkeypatch):
    import vllm_metal.v1.model_runner as model_runner

    runner = _loadable_runner(dspark_config, monkeypatch)
    runner._load_dspark_drafter()
    assert model_runner.load_drafter.call_args.kwargs["quantize"] is True
    monkeypatch.setenv("VLLM_METAL_DSPARK_DRAFT_PRECISION", "source")
    runner = _loadable_runner(dspark_config, monkeypatch)
    runner._load_dspark_drafter()
    assert model_runner.load_drafter.call_args.kwargs["quantize"] is False
    monkeypatch.setenv("VLLM_METAL_DSPARK_DRAFT_PRECISION", "fp8")
    runner = _loadable_runner(dspark_config, monkeypatch)
    with pytest.raises(ValueError, match="DRAFT_PRECISION"):
        runner._load_dspark_drafter()


def test_admission_environment_reaches_plan_and_proposer(dspark_config, monkeypatch):
    runner = _loadable_runner(dspark_config, monkeypatch)
    runner._load_dspark_drafter()
    assert runner._dspark_memory_plan.max_contexts == 4  # min(max_num_seqs, 32)
    assert runner._drafter._max_drafts_per_step == 4
    monkeypatch.setenv("VLLM_METAL_DSPARK_MAX_CONTEXTS", "2")
    monkeypatch.setenv("VLLM_METAL_DSPARK_MAX_DRAFTS_PER_STEP", "1")
    runner = _loadable_runner(dspark_config, monkeypatch)
    runner._load_dspark_drafter()
    assert runner._dspark_memory_plan.max_contexts == 2
    assert runner._drafter._max_drafts_per_step == 1
    monkeypatch.setenv("VLLM_METAL_DSPARK_MAX_CONTEXTS", "0")
    runner = _loadable_runner(dspark_config, monkeypatch)
    with pytest.raises(ValueError, match="MAX_CONTEXTS"):
        runner._load_dspark_drafter()
    assert runner._drafter is None
