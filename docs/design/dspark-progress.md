# DSpark implementation progress

The [specification](dspark.md) defines milestone gates; the
[audit and experiment handoff](dspark-validation.md) preserves baseline evidence.
All runtime support remains experimental until the corresponding model, memory
and serving qualification gates pass.

For the verified integration revision, M5 Max setup, exact reproduction commands,
raw-evidence transfer and remaining implementation sequence, start with the
[development handoff](dspark-handoff.md), checked on 2026-09-10. The handoff is
documentation only; M4-M8 remain open and the destination machine is untested.

| Milestone | Status | Change and validation |
| --- | --- | --- |
| M0: Baseline and contract | [Complete: #1](https://github.com/mhdimo/vllm-metal/pull/1) | Startup guards, resolved draft identity/revision, exact source provenance and normal lint coverage. Executable F1/F3 regressions tracked the defects fixed in M1/M2. |
| M1: Target capture | [Complete: #2](https://github.com/mhdimo/vllm-metal/pull/2) | Native Qwen3 capture, selected logits and complete prefill feature spans. |
| M2: Context lifecycle | [Complete: #3](https://github.com/mhdimo/vllm-metal/pull/3) | Exact per-request ingest, physical rollback, lifecycle invalidation and safe prefix-hit behavior. |
| M3: Loading and memory | Complete for the named 4B memory envelope | Deterministic incremental loading, bounded resource planning, precision and recovery checks; [evidence and remaining parity failure](dspark-m3-validation.md). |
| M4: Fixed-greedy serving | Open; parity blocked | Resolve both extended parity failures, qualify fair admission and HTTP semantics, then measure fixed-K performance. |
| M5: Stochastic verification | Planned | Exact proposal-distribution ownership, rejection/bonus sampling and distribution tests. |
| M6: Calibrated adaptive planning | Planned | Recipe-specific confidence calibration, measured cost curves and causal admission/planning. |
| M7: Production qualification | Planned | Profiled serving benefit, packaged deployment and the one-hour/10,000-request HTTP soak. |
| M8: Additional standalone pairs | Planned | Pair-specific target adapters, precision, capacity and serving qualification within 48 GB. |
| Integrated V4 | Deferred | Outside available 32/48 GB hardware; also requires a qualified V4 target backend. |

## M0: Baseline and supported contract

The Metal startup hook rejects unsupported DSpark modes before target weights
load. Both canonical `dspark` and the `draft_model` alias reach the same factory,
which uses vLLM's resolved draft `ModelConfig` and forwards its revision. The
raw checkpoint parser rejects incompatible families, heads, feature selections,
block attention and RoPE variants before allocating drafter weights.

The original model and config were matched byte-for-byte to the independent
MLX port at `9e39ea2fdc6d99d855af2cb7ef9933391c4391db`; the MIT notice is preserved
in the package. The DSpark package now participates in ordinary Ruff checks.
Formatting and local variable renames preserve the reference computations;
layer/cache iteration now rejects unequal lengths.

At M0, three strict expected failures preserved the reproduced F1/F3 contracts.
They were release blockers, not successful correctness tests; M1 and M2 convert
them into passing regressions. Existing runner lifecycle ownership is retained.
F5 revision/support guards and F9 provenance/lint were addressed in M0; full F5
loading and precision qualification followed in M3. No speedup is claimed.

Validation on the M4 32 GB machine:

- Full non-slow suite: 2,217 passed, 15 skipped, 53 deselected, three strict
  expected failures for the remaining M1/M2 defects.
- Ruff check and format, mypy (144 source files), and strict MkDocs build passed.
- The pinned DeepSpec tiny FP32 Qwen3 and Gemma4 reference comparisons passed
  after the package cleanup, including incremental and ragged context cases.
- Actual vLLM engine configuration accepted both method names against the
  pinned Qwen3-4B target metadata and official DSpark draft configuration with
  `VLLM_USE_V2_MODEL_RUNNER=0`. This is a configuration check, not serving evidence.

Each milestone PR records the exact tested head and local check results. Runtime
and reference environments remain separate, and real-model tests must record
immutable target/draft revisions. The available execution machines remain the
M4 32 GB and M5 Max 48 GB RAM / 2 TB storage.

## M1: Native target capture

Capture executes the loaded Qwen3 body's native forward through a shallow body
copy with local layer observers. Weights and attention modules are shared; the
live target's layer list and parameter tree are unchanged. Native embedding,
masking, RoPE, cache writes and final normalization remain intact. Unsupported
body families, unordered/duplicate feature IDs and incomplete cache lists fail
explicitly. Other target families still require a separately qualified adapter.

Feature capture and logits selection are independent. Every packed feature row
is retained, while the head projects exactly the requested logits rows. Pure
intermediate prefill steps can collect features without projecting logits or
sampling. Their proposer handoff uses the existing absolute positions in
`PrefillRequest.start_pos`, `PagedDecodeSegment.cache_start_pos`, and the packed
row boundaries in `ProposeContext.cu_seqlens`. No duplicate position DTO is needed.
Prompt-logprobs requests retain their full-head path and still receive features.
M2 is responsible for ingesting these spans into persistent context.

The F1 strict expected failure is now a passing regression. Tests cover all
capture/hidden/logits-selection combinations, native final normalization,
exception cleanup, multiple no-sample prefill spans and actual paged KV writes.
The full non-slow suite passed 2,233 tests (15 skipped, 53 deselected), with only
the two M2 strict expected failures remaining. Ruff check/format, mypy and the
strict documentation build passed.

The real Qwen3-4B-4bit target at revision
`4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25` passed 12 bit-exact capture-on/off
comparisons on M4 32 GB with MLX 0.32.1 and mlx-lm 0.32.0. These covered native
incremental chunks of 1/7/5/3 tokens and a packed three-token verification
window with two prefill chunks, full/selected logits and both paged window
layouts. Every physical target KV buffer matched. Peak MLX allocation was
2,562,400,880 bytes; this is a capture test, not a serving memory budget or a
speedup result. The official draft configuration was pinned at
`3457dff1417cb84927f6098a5fcb7cee85c934b7`; no drafter weights were loaded.

Repeat with already downloaded snapshots:

```bash
VLLM_METAL_BUILD_FROM_SOURCE=1 python -m tools.dspark_target_check \
  --target /path/to/pinned/Qwen3-4B-4bit \
  --draft-config /path/to/pinned/dspark_qwen3_4b_block7/config.json \
  --output target-capture-results.json
```

## M2: Exact context lifecycle

One context record binds physical KV and contiguous coverage to the runner's
actual `RequestState` object, which identifies the request generation. Every
scheduled prefill/decode span is checked against its absolute position and
committed input tokens before ingestion. Verification contributes only the old
anchor and accepted draft inputs; the correction/bonus becomes the next anchor.
Recomputed overlaps physically trim every layer before replacing the suffix.
Both K and V must cover the same positions.

Ingestion runs before draft eligibility, including intermediate chunks, K=0,
non-greedy requests and requests outside the current draft batch. One evaluation
per step materializes context so it does not retain earlier target activation
graphs. The old pending-feature stash and extra padded full-prompt replay are
removed. Missing prefixes/features release the partial context and use explicit
target-only fallback until a complete recompute starts at position zero.
There is no proposer-private cross-request prefix cache.

The runner retains lifecycle ownership: finish, cancellation, preemption and
resume invalidate context before any same-step ID reuse. A new `RequestState`
also prevents old context reuse. Ingest/draft exceptions discard affected state
before propagating. No logits or random samples are fabricated for intermediate
steps. Drafting preserves the trained seven-position block, handles empty and
unequal contexts, and honors each row's cap with a correction/bonus reservation.
The single-request path avoids copying its entire context into a padded batch.

The context regressions now pass with real tiny drafter projections and KV
tensors. Coverage includes acceptance lengths 0..7, partial/overlapping prefill,
physical rollback, holes, K=0 and clipped schedules, mixed requests, no-sample
steps, lifecycle invalidation/reuse, errors, output limits and ragged/empty
contexts. The updated batching path also passes the pinned official DeepSpec
Qwen3 and Gemma4 tiny FP32 reference checker.
The full non-slow suite passed 2,257 tests (15 skipped, 53 deselected), with no
expected failures remaining. Ruff check/format, mypy (144 source files), and
the strict documentation build passed.

### Real checkpoint checks

The same pinned 4B target/draft pair was tested on M4 32 GB with a 256-token model
limit, 32-token prefill budget, at most four concurrent requests, greedy output
of 24 tokens, and `VLLM_METAL_MEMORY_FRACTION=0.12`. The loader's default
draft recipe is MLX 4-bit, group size 64. Each case runs baseline and DSpark in
separate processes and repeats four prompts twice. All 40 compared request
outputs matched exactly. Positive proposals, verification and acceptance are
required; target-only execution cannot satisfy this check.

| Concurrency | K | Target prefix caching | Verified drafts | Accepted drafts |
| --- | --- | --- | --- | --- |
| 4 | 2 | Off | 138 | 110 |
| 4 | 7 | On | 221 | 101 |
| 1 | 7 | Off | 278 | 138 |
| 1 | 2 | On | 105 | 81 |
| 4, with cleanup checks | 2 | On | 123 | 94 |

The last case additionally cancels a real request after a scheduled chunk,
reuses its public ID, and drains completion notifications through the normal
engine step. No DSpark context remains after those lifecycle notifications.
The runner unit tests separately exercise reuse of the exact internal request
ID and preemption/resume. Prefix-hit cases exercised missing-feature fallback.
Physical cache invariants were checked during actual engine execution.

The [machine-readable results](dspark-lifecycle-results.json) preserve runtime
versions, token-stream hashes, counters, wall times and peak allocation. Peak
MLX allocation, including loading, was at most 6.18 GB in these probes. This
exceeded the target planner's 2.75 GB allowance before M3 implemented complete
DSpark loading and memory accounting. These historical M2 probes fit the machine;
that fraction was not a complete DSpark memory cap and is now rejected at startup.

These short, instrumented runs are correctness probes, not controlled throughput
benchmarks. Several speculative runs were slower than baseline. No speedup or
production serving envelope is claimed. M3 subsequently qualified allocation
and reuse; M4 must qualify batch admission and fixed-K performance before
confidence scheduling is tuned.

```bash
python -m tools.dspark_lifecycle_check \
  --target /path/to/pinned/Qwen3-4B-4bit \
  --draft /path/to/pinned/dspark_qwen3_4b_block7 \
  --concurrency 4 --width 2 --prefix-cache \
  --output-dir /path/to/results
```

### Remaining experiments and handoff

M0-M3 complete the contract, capture, context and resource foundations for their
declared validation envelopes. The M3 record below includes the bounded memory
and precision results and the extended parity failures still open for M4.
M4 must measure fixed K=0/1/2/7
with warmup, repeated trials, context/output-length buckets, TTFT, inter-token
latency and throughput, including prefix-hit and mixed-prefill workloads. The
fixed cap of 32 draft requests still needs fair, measured admission in M4.

Repeat the supplied capture and lifecycle checkers on the M5 Max 48 GB / 2 TB
machine before extending its workload envelope; that machine was not accessed
for this milestone. Use the [existing experiment matrix](dspark-validation.md#required-experiments)
for larger standalone pairs and model-specific gates. The current results do
not qualify Qwen3-8B/14B, Gemma4, stochastic decoding, calibrated confidence or
integrated V4. V4 remains outside both available machines' memory scale.

## M3: Loading and memory

### Step 1: Deterministic checkpoint loading ([PR #4](https://github.com/mhdimo/vllm-metal/pull/4))

The loader follows `model.safetensors.index.json` when present, checking every
shard against its declared tensor ownership. Without an index, exactly one
safetensors file is required. Missing shards, duplicate JSON keys, mismatched
tensor names/shapes, mixed or non-floating dtypes, and prepacked checkpoints
fail before weight materialization. Requested revisions continue through the
existing Hugging Face/ModelScope resolution path.

The serving conversion remains MLX affine 4-bit/group-64 for linear layers and
embeddings, including prediction heads. Conversion now evaluates one tensor at
a time and scopes allocator retention to the load. Real safetensors tests cover
FP32/FP16/BF16 and sharded files, requiring identical converted parameters and
actual draft logits against the previous whole-model MLX recipe. Unquantized
loads preserve source precision for reference checks. This does not by itself
qualify quantization error against official full-precision checkpoint execution.

The pinned Qwen3-4B lifecycle checker passed again at concurrency four/K=2 with
prefix caching: eight baseline/speculative outputs matched, 123 proposed tokens
were verified, 94 were accepted, and normal completion plus cancellation/ID
reuse drained all request context. Peak MLX allocation including startup was
4,115,459,242 bytes. This is a bounded correctness probe, not a speed benchmark
or a complete memory budget; the target planner still omits drafter resources
until step 2.

The next step must load the drafter before target KV sizing and reserve its
persistent context, staging and execution workspace. Loader correctness alone
does not resolve the M2 resource-accounting gap.

### Step 2: Complete resource planning and bounded storage ([PR #5](https://github.com/mhdimo/vllm-metal/pull/5))

DSpark now loads during the runner's model lifecycle, before profiling or any
target KV allocation. A header-derived startup estimate includes final draft
weights, two largest source-tensor buffers for conversion overlap, and 64 MiB
of kernel reserve. It must fit the configured allowance after target weights.
The on-disk config must also agree with the already resolved draft `ModelConfig`.

The normal cache planner measures both loaded models and subtracts a separate
DSpark reservation. This covers up to `min(max_num_seqs, 32)` complete contexts,
feature capture/casts, context growth/copy/padding overlap, attention and head
workspace, and kernel reserve. It uses the actual draft compute dtype and the
configured model/step token limits. DSpark does not register a synthetic
autoregressive cache group. Startup errors include the reservation and practical
ways to reduce it; model-pair qualification must still check observed peaks.

Request context grows in chunks of 256 within a hard capacity. Appends update
existing storage; rollback exposes only the committed prefix and reuses capacity
without admitting stale suffix tokens. Target features are explicitly cast to
draft precision before projection. Capacity and remaining process-budget checks
precede context allocation; an unavailable complete context produces target-only
generation. Recognized MLX allocation failures discard all private context and
preserve the already sampled target output. Other execution errors still fail.
Recovery requires a new request or a complete recomputation from position zero.

Real-buffer regressions cover geometry/precision accounting, allocation bounds,
rollback/reuse, exhausted request slots, insufficient memory, partial-write
allocation failures and recovery. With the pinned 4B pair at concurrency four,
K=7, max length 256, batch-token limit 32 and memory fraction 0.22, all eight
paired outputs matched and cleanup passed. There were 262 proposed tokens,
256 verified and 118 accepted, including the cancellation/reuse probe. Peak MLX
allocation was 4,729,488,612 bytes within the 5.04 GB configured allowance. The
plan included 20.97 MB context, 3.28 MB capture and 194.49 MB workspace. These
remain correctness/resource probes, not serving-speed qualification.

### Step 3: Qualification and failure evidence ([PR #6](https://github.com/mhdimo/vllm-metal/pull/6))

The final loader checks source tensors for NaN/Inf before conversion and restores
the scoped allocator limit on failure. A preflight check rejects impossible
context/workspace and target-logit reservations before target profiling.

The [M3 validation record](dspark-m3-validation.md) and its committed artifacts
cover the full official BF16 checkpoint, the serving affine-4 recipe, sustained
memory reuse, actual scheduler preemption, exhausted admission and injected
allocation failure. The 528-request run retained no active-memory growth after
drain. All four 1,024-token contexts fit simultaneously with the target KV pool
resident; measured allocation stayed within the configured 5.04 GB allowance.
Seven actual preemptions in the short-context case preserved exact token output
and clean context lifecycle.

Extended 900-token generation and a larger preemption case failed exact-token
parity. Native target-only replays reproduce sensitivity to execution chunk size;
the larger target-only preemption baseline itself generates different streams
for identical prompts. These remain explicit M4 qualification failures, with
reproducible prefixes and logits. They are not passed tests or a production
support claim. Quantized confidence also needs separate M6 calibration.

Final local validation: 2,315 non-slow tests passed (15 skipped, 53 deselected),
with no expected failures; Ruff check/format, mypy (145 source files), shellcheck
and the strict MkDocs build passed. The final-source engine run repeated all
controlled faults and full-capacity allocation with 80 matching batch completions,
two cancellations and two ID reuses, and zero retained active-memory drift.
