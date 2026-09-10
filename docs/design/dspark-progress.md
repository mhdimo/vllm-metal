# DSpark implementation progress

The [specification](dspark.md) defines milestone gates; the
[audit and experiment handoff](dspark-validation.md) preserves baseline evidence.
All runtime support remains experimental until the corresponding model, memory
and serving qualification gates pass.

For the verified integration revision, M5 Max setup, exact reproduction commands,
raw-evidence transfer and remaining implementation sequence, start with the
[development handoff](dspark-handoff.md), checked on 2026-09-10. M4, M5 and
M6a are qualified on the M5 Max destination machine (sections below); M6b, M7
and M8 remain open.

| Milestone | Status | Change and validation |
| --- | --- | --- |
| M0: Baseline and contract | [Complete: #1](https://github.com/mhdimo/vllm-metal/pull/1) | Startup guards, resolved draft identity/revision, exact source provenance and normal lint coverage. Executable F1/F3 regressions tracked the defects fixed in M1/M2. |
| M1: Target capture | [Complete: #2](https://github.com/mhdimo/vllm-metal/pull/2) | Native Qwen3 capture, selected logits and complete prefill feature spans. |
| M2: Context lifecycle | [Complete: #3](https://github.com/mhdimo/vllm-metal/pull/3) | Exact per-request ingest, physical rollback, lifecycle invalidation and safe prefix-hit behavior. |
| M3: Loading and memory | Complete for the named 4B memory envelope | Deterministic incremental loading, bounded resource planning, precision and recovery checks; [evidence and remaining parity failure](dspark-m3-validation.md). |
| M4: Fixed-greedy serving | Complete on M5 Max for the pinned 4B pair: M4a parity contract ([record](dspark-m4-parity.md)), M4b admission fairness, M4c HTTP serving semantics, M4d fixed-K performance | Both extended parity failures are ties or target-unstable prefixes under the recorded contract; admission caps measured on a real engine; the HTTP matrix passes with every divergence a tie; fixed K=2-4 gives +36-49% tokens/s at one request on short prompts and loses 4-28% at four concurrent requests (multi-row verification cost), so adaptive bypass is M6 work. |
| M5: Stochastic verification | Complete on M5 Max for the pinned 4B pair | Exact float32 proposal distributions kept with every scheduled draft, rejection/residual/bonus sampling from per-request streams, enumerated oracle and powered distribution tests, real-engine distribution and mixed-workload gate. |
| M6: Calibrated adaptive planning | M6a done on M5 Max (calibration, cost model, planner); M6b serving integration open | Confidence recording with censored labels, sequential temperature scaling with holdout reliability, measured cost model with bounds, causal prefix planner with an oracle; adaptive mode, counters and its evaluation follow. |
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

## M5 Max migration and numerics (2026-09-10)

The destination machine is an Apple M5 Max, 48 GB unified memory, macOS 26.6,
SDK 26.5, Command Line Tools only, MLX 0.32.1 with the Metal recommended
working set at 40,200,896,512 bytes. Shaders compile in-process under
`VLLM_METAL_BUILD_FROM_SOURCE=1`, so no Metal compiler is needed for the
checkers; wheel packaging still needs Xcode and is deferred on this machine.
The NAX prefill kernels load on this GPU, which the M4 did not exercise.

Fresh install at `4af9a92` with the pinned constraints matched all 188
recorded versions (`uv pip check` clean). The interpreter is CPython 3.12.12
because uv 0.9.18 offers no 3.12.13 build.

The not-slow suite gave 2,310 passed and 5 failed at MLX defaults. Every
failure was an FP32 oracle: MLX runs multi-row FP32 matmuls on the M5 tensor
units at TF32 precision (8e-4 relative error measured against float64, versus
4e-7 for one row). `MLX_ENABLE_TF32=0` restores 9e-7 and all tests pass. The
suite and the tiny reference checker now pin that switch, and
`tests/test_metal_numerics.py` guards it. BF16 (9.5e-4) and affine-4 (7.8e-3)
differences between one-row and eight-row matmuls are unaffected by the switch
and are intrinsic kernel behavior; they explain near-tie token flips between
single-row decode and multi-row verification. See the
[handoff numerics section](dspark-handoff.md#m5-max-numerics-tf32-is-the-fp32-gemm-default).


## M4a: divergence diagnostics (tooling)

`tools/dspark_memory_check.py --trace-logits` records, for both the target-only
and the speculative engine, the top-8 logits of every sampled row together with
the runner's request identity, the absolute position the row predicts, the
number of query rows in that forward, the forward sequence number and the
drafted token. `tools/dspark_divergence_classify.py` replays those records into
each engine's committed stream (verification windows commit through the first
draft mismatch; recomputes overwrite), matches traced requests to prompts by
their token streams so identical prompts stay unambiguous, and labels every
first divergence as a `tie` (both engines rank the two tokens first and second
within two bfloat16 ULPs, upstream #524's criterion) or an
`engine-disagreement` (materially different logits for the same prefix, which
is invalid state). Identical prompts are compared within one engine the same
way. Unit tests cover row identity, window replay, stream-based assignment and
both labels. The M5 Max reproduction of the two M3 failures with this tooling
is recorded in the next section.
\n

## M4a: target parity contract (M5 Max, `02f3b2d`)

Both M3 failures reproduce on the M5 Max with deterministic divergence sets.
With the engines' own logits, the 900-token K=7 run has three ties (0 to 1
bfloat16 ULP between one-row decode and eight-row verification) and one
engine-disagreement; the extended preemption run has three engine-disagreements.
`tools/dspark_target_stability.py` shows that the target-only engine, with no
drafter, returns more than one greedy token at every one of those seven
prefixes when only the prefill chunk budget changes; native mlx-lm shows the
same bistability at absolute position 252 under five of nine chunkings. Four
identical prompts diverge from each other in the target-only engine even
without preemption. The repeated-sentence fixtures drive the target into a
bistable regime (argmax logit near 41 versus a junk basin near 12), and the
speculative and target-only engines are two execution shapes of it.

The [M4a record](dspark-m4-parity.md) defines the contract: a divergence is
admissible only as a tie or at a target-unstable prefix; anything else fails.
`tools/dspark_divergence_classify.py --stability ... --gate` enforces it, and
`--prompt-set natural` adds a non-degenerate workload. Under the contract both
M3 failures are admissible (zero inadmissible divergences); the strict
exact-token checkers keep failing on them by design and their M3 records are
unchanged. DSpark behaved correctly in every trace: the drafter proposed the
content-basin token and the target's verification row rejected it.

On the natural prompt set (eight prompts, 256 tokens, C=4) strict exact
parity fails for 8 of 8 requests at K=7 and 5 of 8 at K=2 with four forced
preemptions, and every divergence is a 0 or 1 ULP tie; the gate passes both.
Acceptance there is 1,352 of 4,732 drafted tokens at K=7 (28.6%) and 1,100 of
1,861 at K=2 (59.1%), against 78.7% at K=7 on the repeated fixtures. Both
M3 failures pass the gate (long output: 3 ties, 1 target-unstable; preemption:
3 target-unstable). Validation on M5 Max at `02f3b2d`: 2,324 non-slow tests
passed, ruff, mypy and the strict docs build clean.

## M4b: fixed-K admission (M5 Max, `8196d12`)

Admission has two caps. The context cap, previously the constant 32, is
`VLLM_METAL_DSPARK_MAX_CONTEXTS` (default 32): the planner reserves
`min(max_num_seqs, cap)` complete contexts before target KV allocation, and a
request scheduled while every slot is held uses target-only generation for
its lifetime because a context needs every earlier target feature; slots are
reused as requests finish. The per-step draft cap,
`VLLM_METAL_DSPARK_MAX_DRAFTS_PER_STEP` (default 0, unlimited), replaces the
first-N truncation of the draft batch with least-recently-drafted rotation,
ties broken by batch order; contexts of waiting requests still advance. Each
row keeps its own output/context/scheduler cap; the batched-versus-independent
regression now also covers mixed caps (7/1/3 and 3/7/1) and asserts each row's
length. Unit tests cover the rotation, bookkeeping release, the configured
plan cap and the runner's environment plumbing.

`tools/dspark_admission_check.py` runs distinct natural prompts through a real
engine and records, per request, the steps it held a complete context, the
steps it was drafted and any fallback. Results with the pinned 4B pair, K=7,
128-token outputs, model length 512, 256 batch tokens (peak MLX active bytes in
the last column):

| Requests / `--max-num-seqs` | Context cap | Per-step cap | Peak concurrent requests / contexts | Context holders | Target-only | Drafted share of eligible steps (min / median / max) | Peak bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 48 / 48 | 32 (default) | none | 19 / 19 | 48 | 0 | 0.91 / 0.97 / 0.98 | 6.74 GB |
| 48 / 48 | 48 | none | 19 / 19 | 48 | 0 | 0.91 / 0.97 / 0.98 | 5.84 GB |
| 48 / 48 | 8 | none | 19 / 8 | 26 | 22 (`context capacity exhausted`) | 0.92 / 0.97 / 0.98 | recorded |
| 16 / 16 | 16 | 4 (binding) | 16 / 16 | 16 | 0 | 0.25 / 0.26 / 0.38 | 7.50 GB |

With this scheduler budget the engine ran at most 19 requests concurrently,
so the default cap excluded nobody; the 8-slot run shows the exclusion path
and slot reuse (26 holders over the run, 8 at a time, 22 requests target-only
and counted). Under the binding per-step cap every holder was drafted in about
a quarter of its eligible steps; the spread comes from requests that finished
early. Every context holder drafted at least once in every run and no context
remained after drain. Validation on M5 Max: 2,330 non-slow tests passed, ruff,
mypy and the strict docs build clean.

## M4c: HTTP serving semantics (M5 Max, `c15eeb0`)

`tools/dspark_serving_check.py` qualifies the real serving path rather than the
in-process checkers: it launches a target-only `vllm serve` (K=0), then one
server per speculative width, each with vLLM's multiprocess engine core, model
length 1,024, four sequences, 64 batch tokens, synchronous scheduling, seed 0
and memory fraction 0.22, and drives every server through the same request
matrix over the OpenAI completions API: output limits 1/2/31/128 and natural
EOS on four prompts, `min_tokens`, a stop string, streaming versus
non-streaming, a long prompt that needs more than three prefill chunks, four
staggered concurrent arrivals (0/150/400/800 ms, limits 64/128/31/96), a
client disconnect after eight streamed chunks followed by an idle check and a
fresh request, a `logprobs` request and a sampled request, and, with
`--prefix-widths`, a repeated prompt and a shared two-thirds prefix on servers
with prefix caching. A speculative server's stream must equal the K=0 server's
stream for the same request, or diverge at a position where the K=0 server's
own top logprobs at that prefix put both tokens within 0.5 of each other (a
logprob gap equals the logit gap, so 0.5 covers two bfloat16 ULPs up to logit
magnitude 64: the M4a tie rule at the HTTP level). Concurrent arrivals are also
compared with the same server's isolated results, streamed output with
non-streamed, and a repeated prefix with its first run; those self-comparisons
are judged the same way. Every speculative server must report draft and
accepted tokens in `/metrics`, and no running or waiting request after the
disconnect. Target instability, the other admissible M4a class, is not waived
here; an HTTP divergence that is not a tie fails the gate.

Final run with the pinned 4B pair (`results/m5max-m4c-serving-03-final`,
counters from `/metrics` over the matrix, seconds excluding server startup):

| Server | Failures | Compared with K=0 | Equal | Ties | Largest tie gap (logprob) | Drafts | Draft tokens | Accepted | Acceptance | Seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| k0 | 0 | reference | – | – | – | 0 | 0 | 0 | – | 27 |
| k1 | 0 | 29 | 22 | 7 | 0.125 | 1,465 | 1,465 | 1,043 | 71.2% | 27 |
| k2 | 0 | 29 | 22 | 7 | 0.125 | 1,107 | 2,204 | 1,404 | 63.7% | 24 |
| k4 | 0 | 29 | 20 | 9 | 0.125 | 894 | 3,537 | 1,615 | 45.7% | 21 |
| k7 | 0 | 29 | 21 | 8 | 0.125 | 817 | 5,602 | 1,696 | 30.3% | 23 |
| k0-prefix | 0 | reference | – | – | – | 0 | 0 | 0 | – | 24 |
| k2-prefix | 0 | 31 | 13 | 18 | 0.25 | 815 | 1,622 | 1,097 | 67.6% | 30 |
| k7-prefix | 0 | 31 | 14 | 17 | 0.25 | 587 | 4,040 | 1,328 | 32.9% | 31 |

Every divergence in the matrix is a tie, at most half the admissible gap. The
target-only servers show the same kind of divergence without any drafter: on
`k0`, two of the four concurrent arrivals differ from their isolated results
(gaps 0 and 0.125), and on `k0-prefix` the streamed request differs from the
non-streamed one because the second request hits the prefix cache and takes
a different arithmetic path (gap 0.125). Batching and cache hits, not
speculation, move these ties; the speculative servers' ties are of the same
kind, and the servers with prefix caching show more of them because the
earlier scenarios leave their prompts cached. `min_tokens` is rejected with
HTTP 400 by every server (the platform declares logits-processor controls
unsupported); the stop string, the long prompt, `logprobs` and the repeated
and shared prefixes match the K=0 server (`logprobs` ties on the
prefix-caching servers); the sampled request returns its 32 tokens on the
target-only path; after the disconnect every server is idle within the poll
window and answers the next request correctly. Acceptance per draft token
falls with width as in the in-process natural-workload run. An earlier full
run (`results/m5max-m4c-serving-01`) flagged one strict streamed-versus-plain
comparison on the prefix-caching K=0 server; that comparison now goes through
the same tie judgement (gap 0.125), and the recorded matrix is the rerun with
the final harness. Validation on M5 Max: 2,333 non-slow tests passed, ruff,
mypy and the strict docs build clean.

## M4d: fixed-K performance (M5 Max, `fe44a9d`)

Two tools separate instrumented in-process attribution from timed serving.
`tools/dspark_step_profile.py` runs an offline engine on the natural prompts
with the decode pipeline disabled (every step samples synchronously) and times,
per scheduler step, the target forward with verification and sampling, the
feature ingest, the context materialization, the batched draft backbone and
heads, and the remainder of `propose`; K=0 runs the same engine without a
drafter. 128-token outputs, model length 1,024, 256 batch tokens, memory
fraction 0.22, one warmup generation before the instrumented one. Milliseconds
per step (p50; step p95 in its own column), pinned 4B pair:

C=1:

| K | Steps | Tokens/step | Step p50 ms | Target p50 ms | Draft p50 ms | Ingest p50 ms | Context p50 ms | Other p50 ms | Step p95 ms | Instrumented tok/s | Peak MLX GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 128 | 1.00 | 7.3 | 7.3 | 0.0 | 0.00 | 0.00 | 0.00 | 8.1 | 132.1 | 8.38 |
| 1 | 68 | 1.88 | 10.9 | 8.0 | 2.4 | 0.08 | 0.35 | 0.01 | 12.6 | 165.0 | 8.15 |
| 2 | 52 | 2.46 | 11.7 | 8.6 | 2.5 | 0.08 | 0.36 | 0.01 | 12.4 | 203.5 | 8.16 |
| 4 | 35 | 3.66 | 13.0 | 9.8 | 2.7 | 0.08 | 0.38 | 0.01 | 14.1 | 270.6 | 8.15 |
| 7 | 30 | 4.27 | 17.2 | 13.2 | 3.4 | 0.09 | 0.39 | 0.01 | 19.3 | 242.1 | 8.13 |

C=4:

| K | Steps | Tokens/step | Step p50 ms | Target p50 ms | Draft p50 ms | Ingest p50 ms | Context p50 ms | Other p50 ms | Step p95 ms | Instrumented tok/s | Peak MLX GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 128 | 4.00 | 8.8 | 8.8 | 0.0 | 0.00 | 0.00 | 0.00 | 9.6 | 431.5 | 8.44 |
| 1 | 76 | 6.74 | 20.7 | 13.9 | 5.8 | 0.22 | 0.84 | 0.01 | 21.7 | 330.2 | 7.92 |
| 2 | 55 | 9.31 | 26.6 | 19.1 | 6.2 | 0.24 | 0.93 | 0.01 | 28.3 | 354.7 | 7.92 |
| 4 | 48 | 10.67 | 30.7 | 22.6 | 6.4 | 0.25 | 0.97 | 0.01 | 32.5 | 387.7 | 7.92 |
| 7 | 45 | 11.38 | 30.8 | 23.0 | 5.9 | 0.25 | 0.96 | 0.01 | 32.5 | 400.2 | 7.94 |

Ingest and context costs are below one millisecond per step; the draft
backbone costs 2.4-3.4 ms at one request and about 6 ms for the padded
four-row batch; the target forward that verifies K+1 rows per request is the
dominant term and grows steeply with rows on this Metal path: at C=4 it costs
14-23 ms against 8.8 ms for the one-row-per-request target-only step, so the
6.7-11.4 tokens per step do not repay it, while at C=1 the 4.3 tokens per
step at K=7 cost 17 ms against 7.3 ms (K=2 and K=4 are the better points).

`tools/dspark_perf_bench.py` is the serving protocol: workload buckets (input
x output tokens) 128x128, 1024x128 and 128x512 at client concurrency 1 and 4,
distinct natural-prose prompts of the exact input length built from the
repository documentation, every request streamed with `ignore_eos`. For each
width the target-only server and the speculative server are both alive; after
one warmup repetition per server each bucket is measured in five paired
repetitions with alternating order, recording output tokens per second, mean
TTFT, median TPOT, the p95 and p99 inter-arrival gaps between streamed chunks
(a verified block arrives as one chunk, so gaps are what a reader perceives)
and goodput at the declared SLO (TTFT at most 1 s and gap p95 at most 100 ms).
Paired relative differences carry a bootstrap 95% interval of the median; the
specification's proposed gate (at least 10% median benefit with the interval
above zero) is evaluated per cell and marked ✓, never asserted. Servers: model
length 2,048, 16 sequences, 512 batch tokens, memory fraction 0.2, synchronous
scheduling on both sides (the speculative path needs it); `--async-reference`
also measures a target-only server with asynchronous scheduling, the best
deployable target-only configuration, against the same synchronous reference.
Results (`results/m5max-m4d-performance-01/bench`, medians over the five
repetitions, `reference → DSpark`):

| Width | Bucket | C | Target-only tok/s | DSpark tok/s | Tokens/s benefit (median, bootstrap 95% CI) | TPOT ms | Gap p95 ms | Gap p99 ms | TTFT ms | Goodput req/s | Accepted / drafted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| k1 | 128x128 | 1 | 116.8 | 136.3 | +20.6% [+6.7, +26.0] ✓ | 8.3 → 7.0 | 9 → 14 | 10 → 14 | 43 → 48 | 0.91 → 1.06 | 342 / 414 |
| k1 | 128x128 | 4 | 310.0 | 236.4 | -25.0% [-29.4, -20.1] | 11.4 → 15.1 | 13 → 29 | 13 → 29 | 137 → 154 | 2.42 → 1.85 | 1318 / 1712 |
| k1 | 1024x128 | 1 | 82.7 | 90.8 | +13.6% [-0.1, +14.8] | 9.9 → 8.6 | 11 → 16 | 11 → 16 | 271 → 297 | 0.65 → 0.71 | 336 / 420 |
| k1 | 1024x128 | 4 | 158.6 | 128.0 | -20.7% [-23.3, -17.3] | 17.7 → 23.2 | 17 → 41 | 167 → 196 | 912 → 987 | 0.62 → 0.25 | 1242 / 1788 |
| k1 | 128x512 | 1 | 92.3 | 112.4 | +26.3% [+14.4, +28.3] ✓ | 10.5 → 8.6 | 13 → 17 | 14 → 17 | 129 → 180 | 0.18 → 0.22 | 1386 / 1674 |
| k1 | 128x512 | 4 | 308.9 | 225.4 | -23.9% [-30.7, -21.5] | 12.6 → 17.3 | 14 → 36 | 14 → 38 | 187 → 202 | 0.60 → 0.44 | 5406 / 6846 |
| k2 | 128x128 | 1 | 135.9 | 201.2 | +48.6% [+45.4, +48.8] ✓ | 7.1 → 4.7 | 7 → 12 | 8 → 12 | 36 → 41 | 1.06 → 1.57 | 450 / 618 |
| k2 | 128x128 | 4 | 326.5 | 244.5 | -26.5% [-29.6, -21.4] | 10.9 → 14.9 | 11 → 36 | 11 → 36 | 130 → 147 | 2.55 → 1.91 | 1704 / 2652 |
| k2 | 1024x128 | 1 | 96.6 | 105.2 | +10.3% [+1.0, +21.4] ✓ | 8.5 → 7.2 | 9 → 19 | 10 → 19 | 273 → 304 | 0.75 → 0.82 | 438 / 630 |
| k2 | 1024x128 | 4 | 182.4 | 145.5 | -21.0% [-22.8, -17.6] | 15.4 → 20.2 | 14 → 85 | 140 → 176 | 783 → 935 | 1.07 → 0.28 | 1668 / 2736 |
| k2 | 128x512 | 1 | 119.6 | 162.5 | +35.9% [+30.3, +40.4] ✓ | 8.2 → 5.9 | 9 → 15 | 9 → 15 | 110 → 111 | 0.23 → 0.32 | 1800 / 2520 |
| k2 | 128x512 | 4 | 335.5 | 272.8 | -18.5% [-19.6, -17.1] | 11.6 → 14.0 | 13 → 36 | 13 → 38 | 173 → 188 | 0.66 → 0.53 | 7146 / 10218 |
| k4 | 128x128 | 1 | 130.2 | 176.6 | +42.4% [+32.1, +44.9] ✓ | 7.3 → 5.3 | 8 → 18 | 8 → 18 | 49 → 47 | 1.02 → 1.38 | 522 / 942 |
| k4 | 128x128 | 4 | 341.1 | 282.4 | -18.2% [-19.1, -15.5] | 10.4 → 12.9 | 11 → 38 | 11 → 39 | 123 → 140 | 2.66 → 2.21 | 1992 / 4128 |
| k4 | 1024x128 | 1 | 99.9 | 113.4 | +13.5% [+7.1, +20.1] ✓ | 8.3 → 6.6 | 9 → 21 | 9 → 21 | 243 → 289 | 0.78 → 0.89 | 498 / 1038 |
| k4 | 1024x128 | 4 | 194.6 | 147.9 | -22.8% [-25.0, -20.5] | 14.4 → 19.8 | 13 → 97 | 135 → 169 | 716 → 908 | 1.14 → 0.29 | 1884 / 4596 |
| k4 | 128x512 | 1 | 126.2 | 177.8 | +45.9% [+39.3, +49.3] ✓ | 7.8 → 5.4 | 8 → 18 | 9 → 18 | 106 → 102 | 0.25 → 0.35 | 2070 / 3966 |
| k4 | 128x512 | 4 | 346.2 | 283.5 | -17.7% [-21.4, -15.8] | 11.2 → 13.1 | 12 → 41 | 13 → 42 | 174 → 183 | 0.68 → 0.55 | 7992 / 17028 |
| k7 | 128x128 | 1 | 131.6 | 149.8 | +13.7% [+7.9, +21.4] ✓ | 7.3 → 6.4 | 8 → 25 | 8 → 25 | 48 → 47 | 1.03 → 1.17 | 540 / 1524 |
| k7 | 128x128 | 4 | 353.3 | 292.0 | -18.8% [-20.7, -16.8] | 10.1 → 11.9 | 11 → 37 | 11 → 38 | 116 → 140 | 2.76 → 2.28 | 2034 / 6936 |
| k7 | 1024x128 | 1 | 99.2 | 84.0 | -15.3% [-24.5, -11.9] | 8.3 → 10.1 | 9 → 27 | 9 → 27 | 240 → 285 | 0.78 → 0.66 | 456 / 2070 |
| k7 | 1024x128 | 4 | 200.5 | 141.1 | -28.3% [-29.6, -27.2] | 14.0 → 21.2 | 13 → 136 | 124 → 188 | 695 → 900 | 1.17 → 0.00 | 1872 / 7992 |
| k7 | 128x512 | 1 | 127.1 | 151.5 | +25.8% [+15.3, +32.7] ✓ | 7.8 → 6.4 | 8 → 24 | 9 → 25 | 105 → 88 | 0.25 → 0.30 | 2202 / 6048 |
| k7 | 128x512 | 4 | 345.2 | 328.1 | -4.2% [-8.0, -1.1] | 11.3 → 10.0 | 12 → 41 | 13 → 42 | 175 → 187 | 0.67 → 0.64 | 8964 / 22956 |
| k0-async | 128x128 | 1 | 127.3 | 144.8 | +12.1% [+11.5, +16.4] ✓ | 7.6 → 6.6 | 8 → 7 | 9 → 7 | 42 → 48 | 0.99 → 1.13 | – |
| k0-async | 128x128 | 4 | 336.8 | 369.9 | +11.3% [+4.6, +12.7] ✓ | 10.6 → 9.5 | 11 → 10 | 11 → 10 | 128 → 161 | 2.63 → 2.89 | – |
| k0-async | 1024x128 | 1 | 101.2 | 108.6 | +10.8% [+1.5, +11.9] ✓ | 8.2 → 7.2 | 9 → 8 | 9 → 8 | 214 → 217 | 0.79 → 0.85 | – |
| k0-async | 1024x128 | 4 | 194.1 | 202.8 | +5.1% [+2.0, +7.1] | 14.4 → 12.4 | 14 → 12 | 135 → 133 | 736 → 850 | 1.14 → 0.79 | – |
| k0-async | 128x512 | 1 | 124.9 | 138.2 | +10.7% [+0.0, +14.7] ✓ | 7.9 → 7.0 | 9 → 8 | 9 → 8 | 117 → 131 | 0.24 → 0.27 | – |
| k0-async | 128x512 | 4 | 322.1 | 347.3 | +6.6% [+4.9, +19.5] | 12.1 → 11.2 | 13 → 13 | 14 → 15 | 192 → 232 | 0.63 → 0.68 | – |

At one request DSpark meets the gate in every 128-token-input bucket at every
width, and in the 1,024-token-input bucket at K=2 and K=4; K=2 and K=4 are the
strongest fixed widths (+36% to +49% tokens/s on short prompts, +10% to +14%
on the long prompt) and K=7 loses on the long prompt (-15%). At four
concurrent requests every width loses 4% to 28%, and on the long-prompt
bucket the p95 gap exceeds the 100 ms SLO from K=2 upward, so SLO goodput
collapses. TPOT improves wherever throughput does, but the streamed gap p95
widens in every cell because a verified block arrives as one chunk (3.4
tokens per arrival at K=7). The asynchronous target-only server is 5% to 12%
faster than the synchronous reference; at one request DSpark still exceeds
it (K=2, 128x128: 201 against 145 tokens/s, unpaired medians) and at four
requests it falls further behind. Per-repetition spread is small (for
example K=7, 128x128, C=1: 129-134 against 142-156 tokens/s). Consequence:
fixed-K speculation is a one-request (or low-concurrency) optimization on this
platform with K=2-4; the M6 planner must bypass drafting from about four
concurrent requests and the multi-row target verification cost is the M7
profiling target. Validation on M5 Max: the tool statistics helpers are unit
tested, 2,338 non-slow tests passed, ruff, mypy and the strict docs build
clean.

## M5: exact stochastic verification (M5 Max, `ac114d7`)

DSpark now drafts plain temperature/top-k/top-p requests and verifies them by
exact rejection sampling; greedy requests keep the argmax draft chain and the
exact greedy verifier, and the two kinds mix freely in one batch.
`SpeculativeDecodeController.draft_mode` classifies every request once:
penalties, logprobs and allowed/bad-token constraints exclude a request from
drafting in every mode (target-only with the existing observable reason),
greedy drafting needs plain argmax sampling (vLLM normalizes a greedy
request's top-k/top-p away), and stochastic drafting additionally excludes
structured output (its grammar state would have to evolve along the drafted
prefix), `min_p` and `logit_bias` (the platform rejects both anyway).

`vllm_metal/v1/dspark/sampling.py` owns the arithmetic. A stochastic row's
proposal distribution at block position k is the float32 softmax of the
drafter's logits plus the Markov step bias of the previous drafted token,
divided by the request's temperature and masked to its top-k/top-p candidate
set with the Metal sampler's own mask function (vLLM's tie and boundary
semantics), so `q` is exactly what a target-only server would sample from at
that request's settings; the token is drawn by inverse CDF over the float32
cumulative sum against a float64 uniform rounded to float32 (a zero-mass token
can never be selected; a cumulative sum that falls short of the uniform by
rounding selects the last token with positive mass). The proposer keeps every
sampled row in a `DSparkProposal` record attached to the scheduled proposal:
the request generation (`RequestState` identity), the anchor position and
token, the drafted tokens, the `[K, vocab]` float32 rows, the transforms, the
precision statement and the request's random streams. Verification builds the
target distribution `p` of every verify row from the target logits with the
same transforms, accepts draft `x_k` with probability `min(1, p_k(x_k) /
q_k(x_k))`, samples the first rejected position from the normalized positive
residual `max(p_k - q_k, 0)` (falling back to `p_k` when the float32 residual
mass is at most 1e-8, the reference evaluator's rule) and samples the bonus
token from the last row after full acceptance. Every uniform comes from the
owning request's own streams (proposal, acceptance, target: three PCG64
generators spawned from the request seed, or from the engine seed and the
request's admission ordinal, or from entropy when the engine has no seed);
the acceptance stream advances by the scheduled width every step, so another
request's scheduling, cancellation or reordering never consumes this
request's draws, and a seeded request reproduces. A record is spent the next
time the scheduler schedules the request (verified, clipped to a prefix, or
dropped for a prefill chunk: vLLM clears a request's drafts on every
scheduling) and released with the request; drafts on a stochastic request
whose record does not match the request generation, anchor, tokens,
transforms or vocabulary are invalid state and raise rather than being
verified against the wrong `q`. The memory plan reserves the stored rows and
the per-position temporaries (`rows x (block + 6) x vocab x 4` bytes, 253 MB
for the 4B pair at 32 contexts). The runner passes the proposer's records and
the vocabulary size to the controller's `verify`, which dispatches per
request; the other Metal proposers stay greedy-only. The drafter's dead
`sample_block_probs` (an import of a module that never existed) is removed.

Tests. `tests/test_dspark_sampling.py`: the transform matches the sampler's
mask semantics (top-k ties survive, top-p keeps the leader, padded lm_head
columns carry no mass), batched rows equal rows computed alone, inverse-CDF
edges (leading zeros, boundaries, a uniform rounded to 1.0, unnormalized
rows, zero-mass tokens never selected over a dense grid), residual clipping
and fallback, zero-proposal-mass reporting, an enumerated two-position
tiny-vocabulary oracle that walks every branch of `verify_rows` with uniforms
chosen inside each interval and exact interval weights (the emitted first
token matches `p_0` and the second, given acceptance, matches `p_1`, to 1e-9),
a predefined 20,000-draw chi-square gate against the target with real request
streams (statistic 15.7 against the 0.001 critical value 20.5) whose negative
control, verifying against a proposal distribution the drafts were not
sampled from, scores above 100, zero-support and truncated proposals, and
stream seeding and isolation. `tests/test_dspark_proposer.py`: stochastic
requests draft with records whose rows sum to one, carry the truncation's
zeros and reproduce the drafted tokens when the request's proposal stream is
replayed from its seed; records are spent on the next schedule and released
with the request; mixed greedy and stochastic rows drafted in one batch equal
the rows drafted alone (tokens exactly, rows within summation order);
creating and releasing another request leaves a request's draws unchanged;
undraftable parameters keep their context without drafting.
`tests/test_spec_decode_metadata.py`: mixed-mode dispatch, residual sampling
at the first rejection, scheduler-clipped drafts using the record prefix,
fail-closed records (missing, other owner, other anchor, other tokens, other
transforms, other vocabulary), drafts on undraftable parameters, and the mode
classification.

Real-engine gate. `tools/dspark_stochastic_check.py` runs the pinned pair
offline as a K=7 engine, a target-only engine and (with
`--control-max-num-seqs`) a second target-only engine at another scheduler
batch size, in separate worker processes with the multiprocess-free engine
core (`results/m5max-m5-stochastic-04`; model length 512, 64 sequences, 512
batch tokens, engine seed 0). Distribution: 8,000 unseeded requests on one
natural prompt, three output tokens each, at temperature 1.0 and at
temperature 0.7 with top-p 0.9; the first token comes from the prefill
sampler on every engine, the second and third are verified drafts on the K=7
engine. Each position's marginal histogram is compared between engines with a
two-sample chi-square over the tokens holding at least ten observations in
either sample (rarer tokens pooled) at significance 0.001, a test fixed
before the run; the last column repeats the same test between the two
target-only engines, which differ only in batch composition:

| Scenario | Position | Buckets | Chi-square | p-value | TV detectable at 80% power | Control chi-square / p (target-only, 16 vs 64 sequences) |
| --- | --- | --- | --- | --- | --- | --- |
| t1.0 | 0: prefill sampler, not drafted (built-in control) | 45 | 22.6 | 0.997 | 0.055 | 26.1 / 0.992 |
| t1.0 | 1: first verified position | 103 | 96.6 | 0.633 | 0.065 | 97.1 / 0.563 |
| t1.0 | 2: second verified position | 99 | 136.2 | 0.006 | 0.065 | 130.6 / 0.016 |
| t0.7-p0.9 | 0: prefill sampler, not drafted (built-in control) | 5 | 3.4 | 0.491 | 0.038 | 0.9 / 0.927 |
| t0.7-p0.9 | 1: first verified position | 38 | 43.9 | 0.201 | 0.054 | 28.7 / 0.802 |
| t0.7-p0.9 | 2: second verified position | 41 | 36.6 | 0.624 | 0.054 | 57.5 / 0.022 |

Every comparison passes the predefined gate. The two smallest p-values sit at
the second verified position, and the target-only control reproduces them
(0.016 and 0.022 against 0.006 and 0.624): the target's own multi-row
arithmetic (the M4a numerics contract) moves the sampled distributions by
about that much without any drafting, so the speculative engine is
indistinguishable from a target-only engine at the sensitivity the run
affords (a total-variation distance of 0.04 to 0.07 at 80% power). Draft
work in those runs: t1.0: 7,880 draft tokens, 4,050 accepted (51.4%); t0.7-p0.9: 7,985 draft tokens, 5,803 accepted (72.7%).

Mixed workload on the K=7 engine (20 requests in one batch,
run twice with the same seeds):

| Request kind | Requests | Draft tokens | Accepted | Settings and outcome |
| --- | --- | --- | --- | --- |
| greedy | 4 | 418 | 124 | temperature 0; parity with the target-only engine judged by the M4a rule |
| stochastic | 4 | 428 | 119 | temperature 0.8, top-p 0.95, seeded |
| seeded-topk | 2 | 212 | 63 | temperature 1.0, top-k 40, seeded |
| logprobs | 2 | 0 | 0 | greedy with `logprobs`: target-only |
| penalty | 2 | 0 | 0 | repetition penalty 1.2: target-only |
| short-budget | 2 | 9 | 3 | temperature 0.9, `max_tokens` 5 |
| stochastic-eos | 1 | 708 | 153 | temperature 0.7, 256-token budget |
| greedy-eos | 1 | 562 | 172 | temperature 0, 256-token budget |
| stop-token | 2 | 133 | 25 | temperature 0.8, `stop_token_ids` on the period token; both ended on the stop token |

Greedy outputs against the target-only engine: 4 equal and
3 ties (logprob gaps 0.125, 0.125, 0.125); logprobs and penalty
requests received no drafts; every output respected its budget; no token
followed an EOS or stop token, and both stop-token requests ended on the stop
token after drafting; the repeated batch reproduced every output token for
token. Validation on M5 Max: 2,377 non-slow tests passed, ruff, mypy and the
strict docs build clean.

## M6a: confidence calibration, cost model and planner (M5 Max, `8b3d3ec`)

The drafter's confidence head emits one raw logit per block position;
`sigmoid` of it estimates the probability that the draft at that position is
accepted given every earlier draft was, so the cumulative product is the
survival probability of each prefix. `vllm_metal/v1/dspark/calibration.py`
owns the pieces that make those estimates usable for planning. The
proposer now keeps a proposal record for every drafted row (greedy rows too)
with the raw confidence logits of the drafted positions, and when the request
is scheduled again it derives the verification outcome from the committed
tokens (the accepted count is the distance from the anchor to the new pending
token minus one) and hands the recorder the logits and survival labels of
the scheduled positions: `1` while the accepted prefix continues, `0` from
the first rejection on; positions the scheduler clipped are censored, and a
request that finished during verification is never scheduled again and
records nothing (positions past an EOS or an output limit are unobservable).
Sequential temperature scaling fits one positive temperature per position on
a fixed grid (`2^(i/4)` from 0.25 to 8) by the binary cross-entropy of the
cumulative survival prediction against the survival label, position by
position with earlier temperatures held fixed; a position without
observations keeps temperature 1. Reliability is reported per position as
expected calibration error over 15 equal-width bins with a bootstrap 95%
interval, Brier score and per-bin averages. The artifact
(`dspark-confidence-calibration/1`) stores the temperatures per recorded
sampling mode with the grid, bin definition, objective, sample counts, corpus
files and revision, split sizes and metrics before and after fitting on both
splits, and the model-pair manifest (target and draft identity and revision,
block size, target layer ids, Markov rank, confidence inputs, draft dtype);
loading rejects another schema, a wrong block size, non-finite or
non-positive temperatures and a manifest mismatch.

`tools/dspark_confidence_calibrate.py record` runs the pinned pair at K=7 on
a deterministic corpus (prompt windows of 32 to 256 tokens cut from the
repository documentation at the recorded git revision, plus the natural
prompts; 240 prompts, 128 output tokens each, split by prompt index into
calibration and holdout halves) in one sampling mode per run; `fit` fits the
temperatures on the calibration split and evaluates both splits
(`results/m5max-m6a-calibration-01`, corpus revision `8b3d3ec`; greedy: 120/120 prompts, 4,395/4,375 proposals; stochastic: 120/120 prompts, 6,122/6,072 proposals):

| Mode | Position | Samples | Temperature | Holdout survival (label mean) | Predicted (raw → calibrated) | Holdout ECE raw [95% CI] | Holdout ECE calibrated [95% CI] | Holdout Brier raw → calibrated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| greedy | 0 | 4,375 | 1.19 | 0.697 | 0.729 → 0.714 | 0.033 [0.027, 0.048] | 0.031 [0.025, 0.045] | 0.157 → 0.157 |
| greedy | 1 | 4,357 | 1 | 0.514 | 0.525 → 0.514 | 0.022 [0.018, 0.037] | 0.021 [0.020, 0.037] | 0.162 → 0.161 |
| greedy | 2 | 4,335 | 1 | 0.391 | 0.380 → 0.372 | 0.026 [0.022, 0.040] | 0.030 [0.024, 0.042] | 0.135 → 0.135 |
| greedy | 3 | 4,308 | 0.841 | 0.303 | 0.284 → 0.282 | 0.025 [0.021, 0.038] | 0.029 [0.023, 0.040] | 0.106 → 0.106 |
| greedy | 4 | 4,285 | 0.841 | 0.242 | 0.218 → 0.221 | 0.031 [0.024, 0.042] | 0.031 [0.025, 0.041] | 0.088 → 0.088 |
| greedy | 5 | 4,263 | 1 | 0.197 | 0.172 → 0.174 | 0.032 [0.027, 0.042] | 0.030 [0.026, 0.040] | 0.077 → 0.077 |
| greedy | 6 | 4,232 | 1 | 0.158 | 0.134 → 0.136 | 0.032 [0.026, 0.041] | 0.033 [0.027, 0.041] | 0.066 → 0.066 |
| stochastic | 0 | 6,072 | 1.19 | 0.600 | 0.647 → 0.636 | 0.048 [0.039, 0.060] | 0.036 [0.028, 0.049] | 0.193 → 0.192 |
| stochastic | 1 | 6,028 | 1.19 | 0.367 | 0.405 → 0.391 | 0.038 [0.030, 0.048] | 0.027 [0.021, 0.039] | 0.176 → 0.175 |
| stochastic | 2 | 5,983 | 1.41 | 0.228 | 0.249 → 0.235 | 0.022 [0.018, 0.033] | 0.017 [0.014, 0.029] | 0.125 → 0.124 |
| stochastic | 3 | 5,937 | 1.19 | 0.138 | 0.155 → 0.143 | 0.017 [0.014, 0.026] | 0.011 [0.010, 0.021] | 0.080 → 0.079 |
| stochastic | 4 | 5,899 | 1.19 | 0.088 | 0.099 → 0.089 | 0.014 [0.012, 0.021] | 0.010 [0.009, 0.017] | 0.051 → 0.051 |
| stochastic | 5 | 5,856 | 1 | 0.059 | 0.065 → 0.058 | 0.011 [0.008, 0.016] | 0.009 [0.007, 0.015] | 0.036 → 0.036 |
| stochastic | 6 | 5,813 | 1.41 | 0.040 | 0.043 → 0.037 | 0.006 [0.005, 0.011] | 0.005 [0.004, 0.010] | 0.024 → 0.024 |

The head is already well calibrated for this pair: the fitted temperatures
stay within 0.84 to 1.41, the raw ECE is 2% to 5% at every position, and
scaling lowers the holdout ECE at the early positions of the stochastic mode
(0.048 to 0.036, 0.038 to 0.027, 0.022 to 0.017) while leaving the greedy
mode within the bootstrap intervals; the objective is the survival
likelihood, not the ECE, so a position's ECE can move either way by a few
thousandths. Calibration is specific to the recorded sampling settings
(greedy; temperature 0.8 with top-p 0.95), which the artifact records.

`vllm_metal/v1/dspark/planner.py` holds the cost model and the planner.
`tools/dspark_cost_profile.py` measures, alone on the machine and with the
decode pipeline disabled, the per-step target forward with verification, the
batched draft backbone and the host bookkeeping over a grid of active decode
requests and drafted widths through the step profiler, and writes the cost
artifact (`dspark-cost/1`: median and p95 per cell, steps, machine and MLX
identity, profiled context length, manifest). The model interpolates the
target cost piecewise-linearly in verify rows between the two profiled
request levels that bracket the batch (at equal rows per request), the draft
and host costs in the request count, and refuses queries outside the
profiled bounds. The planner maximizes expected useful emitted tokens per
step time for one drafting batch: every active request emits at least one
token and needs one target input, so the batch starts at `R` rows and `R`
tokens; the next position of each request is a candidate scored by its
calibrated survival; candidates are admitted in descending score with
deterministic ties (earlier position, then lower batch index), each frozen
before the next is examined, and the walk stops at the first extension that
does not raise the ratio or at a hard row cap. Survival decreases along a
request, so admission is prefix-closed; caps from the output and model
budgets bound each request. `should_draft` takes the draft-or-not decision
before the backbone runs from the survival the previous block predicted (or
the calibration prior for a request without history): it drafts only when
the planned ratio strictly exceeds the target-only ratio inside the profiled
bounds, and reports the reason otherwise (`no-requests`,
`outside-cost-bounds`, `planner-empty`, `target-only-faster`). Tests compare
the planner with a brute-force oracle over every prefix-closed allocation on
convex cost curves, check causality, prefix closure, deterministic ties,
caps and the row cap, show strict early stopping on a non-unimodal curve
(where the oracle finds a better far point the planner does not claim), and
cover the decision's reasons, the model's interpolation and bounds, and both
artifacts' validation. Integration of the planner into serving (adaptive
mode, counters, bypass) is M6b.

The cost artifact for this machine (`results/m5max-m6-cost-01/cost.json`;
Apple M5 Max, MLX 0.32.1, decode pipeline disabled, 128-token outputs on the
natural prompts, memory fraction 0.22) covers 1, 2, 4, 8 and 16 active
requests at widths 0, 1, 2, 3, 4, 5 and 7. Target forward with verification
in milliseconds per step, median (p95), with the draft backbone and host
bookkeeping per step averaged over the drafted widths:

| Active requests | K=0 | K=1 | K=2 | K=3 | K=4 | K=5 | K=7 | Draft ms | Host ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 6.7 (7.1) | 7.4 (7.7) | 8.1 (8.3) | 8.6 (9.1) | 9.3 (9.5) | 11.0 (11.2) | 12.5 (12.9) | 2.7 | 0.40 |
| 2 | 7.2 (7.7) | 8.5 (8.8) | 11.0 (11.2) | 12.5 (12.9) | 14.0 (14.3) | 16.8 (17.1) | 19.2 (19.6) | 4.1 | 0.54 |
| 4 | 8.3 (8.6) | 12.7 (13.2) | 17.0 (17.3) | 19.3 (19.8) | 20.3 (21.0) | 21.6 (23.0) | 21.1 (22.0) | 5.6 | 0.93 |
| 8 | 12.6 (13.2) | 19.6 (20.3) | 23.0 (24.3) | 22.5 (24.0) | 32.9 (35.0) | 33.9 (35.7) | 33.6 (36.2) | 9.7 | 1.79 |
| 16 | 21.1 (22.1) | 23.3 (24.6) | 33.9 (36.4) | 35.0 (38.1) | 60.6 (62.4) | 56.6 (67.6) | 45.6 (54.1) | 15.9 | 4.03 |

The undrafted step costs 6.7 ms at one request and 21.1 ms at sixteen; each
verify row adds about 0.8 ms at one request but the curve steepens with
concurrency (four requests at width 7 verify 32 rows for 21 ms, sixteen
requests at width 4 verify 80 rows for 61 ms), which is the shape behind the
M4d result. A few cells dip below a narrower neighbour (for example sixteen
requests at width 7 against width 4 or 5, and eight requests at width 3
against width 2); the p95 columns show those cells are also the noisiest, and
the cost model makes each level's curve non-decreasing in rows before
interpolating, so a dip can never make a longer prefix look cheaper. The
draft backbone grows from 2.7 ms for one row to 15.9 ms for sixteen, host
costs stay below 4 ms. Validation on M5 Max: 2,406 non-slow tests passed,
ruff, mypy and the strict docs build clean.
