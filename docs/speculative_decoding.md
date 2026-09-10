# Speculative Decoding

vllm-metal supports four speculative decoding methods on the paged-attention
path. Use vLLM's [speculative decoding guide](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
for method behavior and configuration details.

| | MTP | DSpark | Draft model | N-gram |
|---|---|---|---|---|
| `--speculative-config` method | `mtp` | `dspark` | `draft_model` | `ngram` |
| Target models | Gemma4 | Matched Qwen3 pair; M0-M3 evidence for pinned 4B only | Non-hybrid paged-attention models | Non-hybrid paged-attention models |
| Draft source | Matching Gemma4 assistant checkpoint | Parallel backbone and sequential Markov head (consumes target hidden states) | Separate smaller model | Prompt and output token history |
| `num_speculative_tokens` | Configurable (2–3 typical) | Up to checkpoint block size; qualification pending | Configurable (3–5 typical) | Configurable (3–5 typical) |
| Additional model weights | Assistant checkpoint | Drafter checkpoint | Draft model | None |
| Additional KV cache | None; reads target KV | Bounded proposer-owned context KV, reserved before target KV allocation | Second scheduler-managed cache | None |

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

The `Dspark` branch contains an experimental implementation with validated
Qwen3 target capture, context lifecycle and bounded resource foundations. See the
[implementation specification and roadmap](design/dspark.md) and
[validation/experiment handoff](design/dspark-validation.md) before treating this
path as production-ready, and the [milestone results](design/dspark-progress.md)
for the bounded 4B checks already completed.
The [development handoff](design/dspark-handoff.md) provides current integration
status, M5 Max setup and the remaining implementation and experiment sequence.

DSpark uses a parallel backbone with a sequential prediction head. A small
backbone cross-attends over the *target's* fused intermediate-layer hidden
states (selected by the drafter's `target_layer_ids`), proposes a 7-token
block, and applies a rank-256 Markov head for previous-token correction in the
standalone checkpoints listed below. Correct target verification is required
to preserve output; model, memory and performance qualification remain necessary.

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
VLLM_METAL_MEMORY_FRACTION=0.22 \
vllm serve mlx-community/Qwen3-4B-4bit \
  --revision 4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25 \
  --max-model-len 256 \
  --max-num-batched-tokens 32 \
  --max-num-seqs 4 \
  --no-async-scheduling \
  --speculative-config '{"method":"dspark","model":"deepseek-ai/dspark_qwen3_4b_block7","revision":"3457dff1417cb84927f6098a5fcb7cee85c934b7","num_speculative_tokens":2}'
```

The drafter can be an HF repo id or a local path. K=2 is an example, not a
hardware-wide optimum. Pin `revision` in the speculative config or use immutable
local snapshots for reproducible experiments. Both `dspark` and upstream's
`draft_model` auto-detection use the resolved draft model and revision.

Startup currently permits matched standalone Qwen3 targets and vanilla Markov
drafters. It rejects unimplemented adaptive/probabilistic/synthetic drafting,
top-k Markov shortcuts, draft quantization/backend/cache overrides, non-paged
attention and LoRA. The prototype's drafter still uses the existing MLX 4-bit
recipe. The loader validates the resolved checkpoint, admits conversion memory,
and materializes draft weights before target KV planning. Broader model-pair
precision qualification is tracked in the roadmap.
Set `VLLM_USE_V2_MODEL_RUNNER=0` explicitly. Request sampling eligibility remains
greedy-only, with unsupported requests following the existing target-only path.

Confirm speculative decoding is active: the server log shows
`DSpark drafter loaded for speculative decoding: <model> (block_size=7,
target_layer_ids=[...])`, and the periodic `SpecDecoding metrics ... Avg
Draft acceptance rate` reflects the live acceptance.

### Characteristics

- **Greedy only**, like every Metal spec-decode method.
- **Batched drafting.** The backbone runs across selected requests with padded
  per-request contexts. Memory is reserved for up to `min(max_num_seqs, 32)`
  complete contexts; excess requests use target-only generation.
- **Contiguous context.** Every prefill chunk contributes features, including
  no-sample steps. Missing features on a target prefix-cache hit use target-only
  generation; the proposer does not replay full prompts or share private KV.

### Limitations

- **Matched models required.** Do not infer support for another target from a
  similar model name or tensor shape.
- **Incomplete DSpark features.** Stochastic verification, calibrated confidence
  scheduling and production serving qualification remain roadmap work.
- **Bounded context and workspace.** The planner subtracts the draft context,
  capture and execution reservation before sizing target KV. Context buffers
  grow in reusable chunks up to the planned model length. Insufficient startup
  capacity fails explicitly; request admission and recoverable draft allocation
  failures fall back to the target. Lower context, sequence and batch-token
  limits to reduce the reservation. The earlier `0.12` memory-fraction example
  is insufficient for the complete 4B load and is now rejected.
- **Output parity is exact up to the target's own numerical stability.** The
  M3 longer-generation and larger-preemption probes fail strict exact-token
  comparison; with both engines' logits traced, every divergence is either a
  tie within two bfloat16 ULPs or a prefix where the target-only engine itself
  returns different greedy tokens under different prefill chunkings. The
  [M4a record](design/dspark-m4-parity.md) defines that contract and the gate
  that enforces it; the [M3 records](design/dspark-m3-validation.md#failed-extended-parity-checks-m4-remains-open)
  keep the strict failures. Degenerate repetitive prompts can flip between
  basins on any execution path, speculative or not.

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
