# Speculative Decoding

vllm-metal supports four speculative decoding methods on the paged-attention
path. Use vLLM's [speculative decoding guide](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
for method behavior and configuration details.

| | MTP | DSpark | Draft model | N-gram |
|---|---|---|---|---|
| `--speculative-config` method | `mtp` | `dspark` | `draft_model` | `ngram` |
| Target models | Gemma4 | Qwen3 4B/8B/14B (needs a matched drafter) | Non-hybrid paged-attention models | Non-hybrid paged-attention models |
| Draft source | Matching Gemma4 assistant checkpoint | DeepSeek EAGLE3+Markov drafter (consumes target hidden states) | Separate smaller model | Prompt and output token history |
| `num_speculative_tokens` | Configurable (2–3 typical) | 2 (recommended) | Configurable (3–5 typical) | Configurable (3–5 typical) |
| Additional model weights | Assistant checkpoint | Drafter checkpoint | Draft model | None |
| Additional KV cache | None; reads target KV | None; reads target hidden states, plus a small proposer-owned context KV | Second scheduler-managed cache | None |

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

DSpark is DeepSeek's EAGLE-family speculative-decoding drafter. A small
backbone cross-attends over the *target's* fused intermediate-layer hidden
states (selected by the drafter's `target_layer_ids`), proposes a 7-token
block, and applies a rank-256 Markov head for previous-token correction. The
target verifies every token, so output is greedy-identical up to
floating-point tie-breaking.

A DSpark drafter is **trained per target** — it consumes that target's
hidden states and predicts that target's continuations, so it only works for
models with a published matched drafter:

| Target | DSpark drafter |
| --- | --- |
| `mlx-community/Qwen3-4B-4bit` (or any quant) | `deepseek-ai/dspark_qwen3_4b_block7` |
| `mlx-community/Qwen3-8B-8bit` (or any quant) | `deepseek-ai/dspark_qwen3_8b_block7` |
| `mlx-community/Qwen3-14B-8bit` (or any quant) | `deepseek-ai/dspark_qwen3_14b_block7` |

For targets without a matched DSpark drafter, use [N-gram](#n-gram)
(model-agnostic) instead.

### Serve

```bash
# DSpark requires the V1 model runner — Metal has no Triton, and vLLM 0.25.1's
# native DSpark path forces Model Runner V2, which errors without it.
VLLM_USE_V2_MODEL_RUNNER=0 \
VLLM_METAL_MEMORY_FRACTION=0.8 \
vllm serve mlx-community/Qwen3-4B-4bit \
  --max-model-len 2048 \
  --max-num-seqs 1 \
  --no-async-scheduling \
  --speculative-config '{"method":"draft_model","model":"deepseek-ai/dspark_qwen3_4b_block7","num_speculative_tokens":2}'
```

The drafter can be an HF repo id or a local path. `num_speculative_tokens=2`
is the measured optimum on Apple Silicon (the verify cost grows with each
accepted token, so longer blocks rarely pay).

Confirm speculative decoding is active: the server log shows
`DSpark drafter loaded for speculative decoding: <model> (block_size=7,
target_layer_ids=[...])`, and the periodic `SpecDecoding metrics ... Avg
Draft acceptance rate` reflects the live acceptance.

### Characteristics

- **Greedy only**, like every Metal spec-decode method.
- **Prefix-caching compatible.** DSpark re-runs a tapped target forward to
  seed the drafter context at first draft, so it does not require
  `--no-enable-prefix-caching`.
- **Per-request drafting in v1.** Each request's drafter context is drafted
  independently; a batched drafter forward is a follow-up.

### Limitations

- **Per-target drafters.** No drafter is published for Llama, Mistral, Phi, or
  other families — DSpark cannot accelerate them. Use N-gram for those.
- **Floating-point tie-breaking.** On near-ties the multi-token verify
  forward may flip an argmax the single-token baseline would not, so output
  is greedy-*identical up to fp ties*, not bit-identical on every prompt.
- **Drafter context grows with generation length** in v1 (no eviction yet).

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
