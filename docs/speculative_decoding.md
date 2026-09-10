# Speculative Decoding

vllm-metal supports four speculative decoding methods on the paged-attention
path. Use vLLM's [speculative decoding guide](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
for method behavior and configuration details.

| | MTP | DSpark | Draft model | N-gram |
|---|---|---|---|---|
| `--speculative-config` method | `mtp` | `dspark` | `draft_model` | `ngram` |
| Target models | Gemma4 | Qwen3 4B/8B/14B (needs a matched drafter) | Non-hybrid paged-attention models | Non-hybrid paged-attention models |
| Draft source | Matching Gemma4 assistant checkpoint | Parallel backbone and sequential Markov head (consumes target hidden states) | Separate smaller model | Prompt and output token history |
| `num_speculative_tokens` | Configurable (2–3 typical) | Up to checkpoint block size; qualification pending | Configurable (3–5 typical) | Configurable (3–5 typical) |
| Additional model weights | Assistant checkpoint | Drafter checkpoint | Draft model | None |
| Additional KV cache | None; reads target KV | Proposer-owned context KV; currently unbudgeted | Second scheduler-managed cache | None |

All four methods currently have these Metal-specific constraints:

- Only plain greedy requests (`temperature=0`, without penalties, token
  constraints, or sample logprobs) are drafted. Other requests run without
  speculation.
- Scheduling must be synchronous. The Metal platform disables async scheduling
  when speculative decoding is configured.
- Pipeline parallelism is not supported with speculative decoding.
- Hybrid GDN targets and heterogeneous draft vocabularies are not supported.
- `long_prefill_token_threshold`, when set, must be at least
  `1 + num_speculative_tokens`.

## Gemma4 MTP

Follow the upstream [MTP guide](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/)
for Gemma4 assistant behavior. Use matching target and assistant families:

| Target | Assistant |
|---|---|
| Gemma4 E2B-it | Gemma4 E2B-it assistant bf16 |
| Gemma4 E4B-it | Gemma4 E4B-it assistant bf16 |
| Gemma4 31B-it bf16 | Gemma4 31B-it assistant bf16 |

Start with `num_speculative_tokens=3`. On the measured E4B workload, higher
values improved single-stream throughput but reduced saturated throughput.
Benchmark the intended batch shape before changing it.

### Example

```bash
export TARGET=/path/to/gemma-4-E2B-it
export ASSISTANT=/path/to/gemma-4-E2B-it-assistant-bf16

VLLM_METAL_MEMORY_FRACTION=0.5 \
  vllm serve "$TARGET" \
    --max-model-len 1024 \
    --max-num-batched-tokens 1024 \
    --max-num-seqs 4 \
    --no-async-scheduling \
    --speculative-config "{\"method\":\"mtp\",\"model\":\"$ASSISTANT\",\"num_speculative_tokens\":3}"
```

Remote Hugging Face checkpoints are supported. Pin `revision` in
`speculative_config` when publishing benchmark results.

## Draft model

Follow the upstream [draft-model guide](https://docs.vllm.ai/en/latest/features/speculative_decoding/draft_model/)
for configuration details. The draft must use the target vocabulary and full
attention. Sliding-window and hybrid draft models are rejected at startup.
Its committed KV cache shares the Metal KV memory budget with the target.

### Example

```bash
VLLM_METAL_MEMORY_FRACTION=0.55 \
  vllm serve Qwen/Qwen3-8B \
    --max-model-len 2048 \
    --no-async-scheduling \
    --speculative-config '{"method":"draft_model","model":"Qwen/Qwen3-0.6B","num_speculative_tokens":3}'
```

### Dynamic speculative decoding

See the upstream [dynamic speculative decoding guide](https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/)
for configuration details. This example sets `K=3` for one scheduled request
and `K=0` for two:

```bash
VLLM_METAL_MEMORY_FRACTION=0.55 \
  vllm serve Qwen/Qwen3-8B \
    --max-model-len 2048 \
    --max-num-seqs 2 \
    --no-async-scheduling \
    --speculative-config '{
      "method": "draft_model",
      "model": "Qwen/Qwen3-0.6B",
      "num_speculative_tokens": 3,
      "num_speculative_tokens_per_batch_size": [[1, 1, 3], [2, 2, 0]]
    }'
```

## DSpark

The `Dspark` branch currently contains an experimental implementation with open
correctness and serving-integration findings. See the
[implementation specification and roadmap](design/dspark.md) and
[validation/experiment handoff](design/dspark-validation.md) before treating this
path as production-ready. The example below describes the prototype.

DSpark uses a parallel backbone with a sequential prediction head. A small
backbone cross-attends over the *target's* fused intermediate-layer hidden
states (selected by the drafter's `target_layer_ids`), proposes a 7-token
block, and applies a rank-256 Markov head for previous-token correction in the
standalone checkpoints listed below. Correct target verification is required
to preserve output; the current integration has open correctness findings.

A DSpark drafter is **trained per target** — it consumes that target's
hidden states and predicts that target's continuations, so it only works for
models with a published matched drafter:

| Trained target | DSpark drafter |
| --- | --- |
| `Qwen/Qwen3-4B` | `deepseek-ai/dspark_qwen3_4b_block7` |
| `Qwen/Qwen3-8B` | `deepseek-ai/dspark_qwen3_8b_block7` |
| `Qwen/Qwen3-14B` | `deepseek-ai/dspark_qwen3_14b_block7` |

These pairings are from the [official DeepSpec release](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/README.md).
Quantized target/draft combinations need their own correctness and performance
qualification. Gemma4 and integrated DeepSeek-V4 are separate roadmap milestones.

For targets without a matched DSpark drafter, use [N-gram](#n-gram)
(model-agnostic) instead.

### Serve

```bash
# Prototype diagnostic example; not a production-qualified configuration.
# vLLM 0.28 selects its GPU V2 path for DSpark unless explicitly overridden.
VLLM_USE_V2_MODEL_RUNNER=0 \
VLLM_METAL_MEMORY_FRACTION=0.35 \
vllm serve mlx-community/Qwen3-4B-4bit \
  --max-model-len 2048 \
  --max-num-seqs 1 \
  --no-async-scheduling \
  --speculative-config '{"method":"dspark","model":"deepseek-ai/dspark_qwen3_4b_block7","num_speculative_tokens":2}'
```

The drafter can be an HF repo id or a local path. K=2 is an example, not a
hardware-wide optimum. Pin `revision` in the speculative config or use immutable
local snapshots for reproducible experiments. Both `dspark` and upstream's
`draft_model` auto-detection use the resolved draft model and revision.

Startup currently permits matched standalone Qwen3 targets and vanilla Markov
drafters. It rejects unimplemented adaptive/probabilistic/synthetic drafting,
top-k Markov shortcuts, draft quantization/backend/cache overrides, non-paged
attention and LoRA. The prototype's drafter still uses the existing MLX 4-bit
recipe; broader precision/resource qualification is tracked in the roadmap.
Set `VLLM_USE_V2_MODEL_RUNNER=0` explicitly. Request sampling eligibility remains
greedy-only, with unsupported requests following the existing target-only path.

Confirm speculative decoding is active: the server log shows
`DSpark drafter loaded for speculative decoding: <model> (block_size=7,
target_layer_ids=[...])`, and the periodic `SpecDecoding metrics ... Avg
Draft acceptance rate` reflects the live acceptance.

### Characteristics

- **Greedy only**, like every Metal spec-decode method.
- **Batched drafting.** The backbone runs across selected requests with padded
  per-request contexts; the current fixed admission cap is 32 requests.
- **Prefix/chunk handling needs qualification.** Prompt replay and partial
  prefill capture have open position-coverage and performance findings.

### Limitations

- **Matched models required.** Do not infer support for another target from a
  similar model name or tensor shape.
- **Incomplete DSpark features.** Stochastic verification, calibrated confidence
  scheduling and bounded draft memory remain roadmap work.
- **Context grows with generation length.** Padded batch copies add to peak
  memory; a lower memory fraction is not a complete resource budget.
- **Output parity requires validation.** Investigate every divergence, including
  target logits layout and cache state, before attributing it to numerical ties.

---

## N-gram

Follow the upstream [N-gram guide](https://docs.vllm.ai/en/latest/features/speculative_decoding/n_gram/)
for configuration details. N-gram speculation needs no additional model or KV
cache. Its benefit depends on repeated token spans in the request history.

### Example

```bash
VLLM_METAL_USE_PAGED_ATTENTION=1 \
  vllm serve Qwen/Qwen3-8B \
    --max-model-len 2048 \
    --no-async-scheduling \
    --speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_min":2,"prompt_lookup_max":3}'
```

## Benchmarking

Use vLLM's benchmark CLI for serving workloads. For a reproducible Gemma4
target-only versus MTP comparison, use the in-tree benchmark:

```bash
python -m tools.benchmark.gemma4_mtp_benchmark --help
```

`tools/README.md` documents the before-and-after commands and the natural-prompt
dataset used for speculative-decoding measurements.
