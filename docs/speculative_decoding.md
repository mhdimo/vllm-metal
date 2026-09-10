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
Set `VLLM_USE_V2_MODEL_RUNNER=0` explicitly. Greedy requests and plain
temperature/top-k/top-p requests are drafted; requests with penalties,
logprobs, allowed or bad token constraints, `min_p`, `logit_bias` or structured
output follow the existing target-only path.

`tools/dspark_serving_check.py` qualifies the HTTP path against a target-only
server (output limits, EOS and stop strings, streaming, long prompts, staggered
arrivals, a mid-stream disconnect, `logprobs` and sampled requests, prefix
repeats) and applies the M4a tie rule to every divergence; see the
[progress record](design/dspark-progress.md). The Metal platform rejects
`min_tokens` on every server, speculative or not.

`VLLM_METAL_DSPARK_MODE=adaptive` (with `VLLM_METAL_DSPARK_CALIBRATION` and
`VLLM_METAL_DSPARK_COST_MODEL` pointing at the pair's calibration artifact
and this machine's cost model, see [configuration](configuration.md)) lets
the proposer plan each request's draft prefix from its calibrated confidence
and the measured step costs, and skip drafting altogether when the bypass
step is predicted to be faster; `fixed`, the default, verifies the configured
width for every eligible request, and `bypass` keeps the drafter loaded but
never drafts. The artifacts come from
`tools/dspark_confidence_calibrate.py` and `tools/dspark_cost_profile.py`;
startup fails with the reason when either is missing or belongs to another
pair. `VLLM_METAL_DSPARK_DRAFT_PRECISION=source` keeps the drafter in the
checkpoint's own precision instead of the qualified affine 4-bit conversion
(more memory, a different cost profile; re-profile the cost model for it);
`tools/dspark_acceptance_eval.py` measures the accepted length per drafting
round on the DeepSpec evaluation prompt sets for either setting.

Fixed-K performance on M5 Max with the 4B pair (see the progress record for
the protocol and every cell): at one request, `num_speculative_tokens` 2-4
raises output tokens per second by 36-49% on 128-token prompts and by 10-14%
on 1,024-token prompts against a target-only server; at four concurrent
requests every width is 4-28% slower than target-only because the multi-row
verification forward is expensive on this path, and the streamed
inter-arrival gap widens because a verified block arrives as one chunk. Use
DSpark for low-concurrency serving with K=2-4 until adaptive planning lands.

Confirm speculative decoding is active: the server log shows
`DSpark drafter loaded for speculative decoding: <model> (block_size=7,
target_layer_ids=[...]) ... mode=<fixed|adaptive|bypass>`, the periodic
`SpecDecoding metrics ... Avg Draft acceptance rate` reflects the live
acceptance, and every 2,000 drafting steps the proposer logs a `DSpark
counters` snapshot (bypass reasons, proposed, scheduled and accepted tokens,
per-position acceptance, planner lengths). `tools/dspark_soak.py` drives a
server with a mixed closed-loop workload for a duration and request count
and reports latency percentiles, throughput, cancellations, the memory
trajectory and the spec-decode counters.

### Characteristics

- **Greedy and stochastic requests.** Greedy requests are verified exactly.
  A plain temperature/top-k/top-p request samples its drafts from exact
  float32 proposal distributions that stay attached to the scheduled proposal,
  and verification accepts each draft with probability `min(1, p/q)`, samples
  the first rejected position from the normalized residual and the bonus
  token from the target distribution, all from the request's own random
  streams (seeded requests reproduce). The emitted distribution equals the
  target's; the tokens at a given seed differ from target-only serving.
- **Verification rows on the small-M kernel.** A verification step runs
  `num_speculative_tokens + 1` rows per request through the target, and the
  drafter's block backbone seven rows per request; on Apple GPUs the stock
  quantized matmul prices six to sixteen rows like a full GEMM tile, so those
  calls go through a kernel that reads each weight group once for every row
  (`VLLM_METAL_SMALL_M_QMM`, measured per weight shape at load; see the
  configuration reference). Outputs differ from the stock kernel only at the
  bfloat16 ULP level, the parity contract's tie class.
- **Batched drafting.** The backbone runs across selected requests, each row
  attending to its own slot of a per-layer context arena (no padding, gather or
  mask), and a step's accepted rows are ingested in one batched pass per layer.
  Memory is reserved for up to
  `min(max_num_seqs, VLLM_METAL_DSPARK_MAX_CONTEXTS)` complete contexts
  (default 32); a request scheduled while every slot is held uses target-only
  generation, and slots are reused as requests finish.
  `VLLM_METAL_DSPARK_MAX_DRAFTS_PER_STEP` bounds the requests drafted per step
  with least-recently-drafted rotation; see [configuration](configuration.md).
- **Contiguous context.** Every prefill chunk contributes features, including
  no-sample steps. Missing features on a target prefix-cache hit use target-only
  generation; the proposer does not replay full prompts or share private KV.
- **Decode pipeline on non-drafting steps.** The proposer can consume a
  pure-decode step whose sampling sync the runner's one-step-ahead pipeline
  defers (the `bypass` mode, or the `adaptive` planner's bypass decision for
  the batch), ingesting that step's target features without the sampled
  token values; drafting and verification steps stay synchronous. The
  pipeline requires asynchronous scheduling, which DSpark servers run by
  default (below); `--no-async-scheduling` keeps every step synchronous.
- **Asynchronous scheduling.** A DSpark server runs vLLM's asynchronous
  scheduler (the production default for target-only serving): the scheduler
  books `num_speculative_tokens` placeholder slots per running request and
  the runner fills them with the drafts it produced at the end of the
  request's previous step, reporting unused slots so the speculative-decode
  metrics count real drafts. The other Metal speculative methods still force
  synchronous scheduling.

### Limitations

- **Matched models required.** Do not infer support for another target from a
  similar model name or tensor shape.
- **Incomplete DSpark features.** Calibrated confidence scheduling and
  production serving qualification remain roadmap work.
- **Bounded context and workspace.** The context arena is allocated once at
  load (every slot holds the planned model length plus the block's scratch
  positions) and is then part of the measured model memory; the target KV
  planner subtracts the capture staging and the per-step workspace that
  remain. Insufficient startup
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
