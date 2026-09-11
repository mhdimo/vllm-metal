# DSpark implementation progress

The [specification](dspark.md) defines milestone gates; the
[audit and experiment handoff](dspark-validation.md) preserves baseline evidence.
All runtime support remains experimental until the corresponding model, memory
and serving qualification gates pass.

For the verified integration revision, M5 Max setup, exact reproduction commands,
raw-evidence transfer and remaining implementation sequence, start with the
[development handoff](dspark-handoff.md), checked on 2026-09-10. M4, M5 and
M6 are qualified on the M5 Max destination machine (sections below); M7 and
M8 remain open.

| Milestone | Status | Change and validation |
| --- | --- | --- |
| M0: Baseline and contract | [Complete: #1](https://github.com/mhdimo/vllm-metal/pull/1) | Startup guards, resolved draft identity/revision, exact source provenance and normal lint coverage. Executable F1/F3 regressions tracked the defects fixed in M1/M2. |
| M1: Target capture | [Complete: #2](https://github.com/mhdimo/vllm-metal/pull/2) | Native Qwen3 capture, selected logits and complete prefill feature spans. |
| M2: Context lifecycle | [Complete: #3](https://github.com/mhdimo/vllm-metal/pull/3) | Exact per-request ingest, physical rollback, lifecycle invalidation and safe prefix-hit behavior. |
| M3: Loading and memory | Complete for the named 4B memory envelope | Deterministic incremental loading, bounded resource planning, precision and recovery checks; [evidence and remaining parity failure](dspark-m3-validation.md). |
| M4: Fixed-greedy serving | Complete on M5 Max for the pinned 4B pair: M4a parity contract ([record](dspark-m4-parity.md)), M4b admission fairness, M4c HTTP serving semantics, M4d fixed-K performance | Both extended parity failures are ties or target-unstable prefixes under the recorded contract; admission caps measured on a real engine; the HTTP matrix passes with every divergence a tie; fixed K=2-4 gives +36-49% tokens/s at one request on short prompts and loses 4-28% at four concurrent requests (multi-row verification cost), so adaptive bypass is M6 work. |
| M5: Stochastic verification | Complete on M5 Max for the pinned 4B pair | Exact float32 proposal distributions kept with every scheduled draft, rejection/residual/bonus sampling from per-request streams, enumerated oracle and powered distribution tests, real-engine distribution and mixed-workload gate. |
| M6: Calibrated adaptive planning | Complete on M5 Max for the pinned 4B pair | Confidence recording with censored labels, sequential temperature scaling with holdout reliability, measured cost model with bounds, causal prefix planner with an oracle, and the adaptive serving mode with bypass, history reset and counters, evaluated in the paired protocol. |
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

The in-process cost artifact recorded here was superseded in M6b by a
serving-path profile (`dspark-cost/2`, see the M6b section) after the first
adaptive evaluation showed it under-predicts the served step; the numbers
stay as the attribution record. The cost artifact for this machine
(`results/m5max-m6-cost-01/cost.json`;
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
## M6b: adaptive serving mode (M5 Max, `adc00d8`)

`VLLM_METAL_DSPARK_MODE=adaptive` binds the calibration artifact and the
cost model (`VLLM_METAL_DSPARK_CALIBRATION`, `VLLM_METAL_DSPARK_COST_MODEL`,
both validated against the served pair's manifest at startup; a missing or
mismatched artifact or an unknown mode fails startup with the reason, and
`fixed`, the default, keeps the M4 behaviour) to the proposer through
`vllm_metal/v1/dspark/adaptive.py`. A third mode, `bypass`, keeps the
drafter loaded and every context advancing but never drafts: it is the
step the planner weighs drafting against, and serves profiling and A/B
comparisons. Each step the proposer first selects the eligible, capped plans
as before, counts the requests the next target step will decode or verify
(drafted or not, one row each at least) and their mean committed length,
and asks the planner whether drafting the batch pays: the expected survival
of every candidate is the calibrated survival its previous block predicted,
or the calibration prior (mean survival per position on the calibration
split) for a request without history, and `should_draft` compares the
planned ratio of expected emitted tokens per step time with the bypass
ratio inside the profiled cost bounds. A bypass skips the backbone entirely
(its cost is not recoverable afterwards), counts its reason and leaves the
contexts advancing. When drafting proceeds, the backbone runs once for the
batch, the fresh confidence logits are calibrated with the mode's
temperatures, the planner allocates a causal prefix per request, and every
row and its proposal record (tokens, distributions of a stochastic row,
confidence) is cut to that length; rows planned at zero are not handed to
the scheduler. History is per request generation: it is dropped when the
request is released (finish, cancel, preemption) and when a scheduled
request goes a step without a draft, so a resumed or idle request starts
again from the prior. `DSparkCounters` accumulates steps, drafting steps,
bypass reasons, proposed, scheduled and accepted tokens, verified requests,
correction or bonus tokens, per-position opportunities and acceptances,
planner lengths and the last planned and bypass ratios, from the same
outcome derivation the recorder uses; nothing is logged per step.

The first evaluation exposed the in-process cost model of M6a as
insufficient. Its target cost was the runner's own phase (execute to
sample), and even the whole engine-core step measured in process differed
from it by less than a millisecond, yet the paired HTTP benchmark at four
concurrent requests lost 20% where the model predicted a small gain: the
step a served request experiences is longer than the in-process step, the
gap grows with drafted tokens, and the model ignored the decode context,
which the 1,024-token buckets showed to matter. `tools/dspark_cost_profile.py`
therefore measures through the serving path: for every drafted width one
`vllm serve` runs in the fixed mode and for width 0 the same speculative
server runs in the bypass mode, each driven, per profiled request count and
context, by that many concurrent streaming requests of exactly the
context's length; the step cost of a cell is the median gap between
consecutive streamed chunks of one request while every request of the
batch is decoding (p95 kept as the uncertainty), and the drafter's own work
per step comes from the in-process profiler at one width per request level
and is subtracted to give the planner's target cost. The artifact
(`dspark-cost/2`) adds the decode context as a dimension; the model
interpolates in rows, then request level, then context, and treats a batch
beyond the longest profiled context as outside its bounds. Serving-path step costs in milliseconds, median (p95), of the 4B pair on this
machine (`results/m5max-m6-cost-03`; memory fraction 0.4 so sixteen long requests
fit, 64-token outputs, the drafter's work from the in-process profiler at width 4):

| Context (decode tokens) | Requests | bypass | K=1 | K=2 | K=4 | K=7 | Draft ms | Host ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 160 | 1 | 7.4 (7.8) | 10.7 (12.1) | 11.2 (11.6) | 14.3 (15.0) | 19.4 (20.5) | 2.6 | 0.42 |
| 160 | 2 | 8.2 (9.0) | 13.3 (13.7) | 20.3 (22.3) | 25.9 (26.4) | 31.5 (32.1) | 4.2 | 0.58 |
| 160 | 4 | 9.5 (9.8) | 19.4 (19.7) | 30.3 (30.6) | 40.8 (42.6) | 35.2 (36.0) | 5.7 | 0.92 |
| 160 | 8 | 14.5 (14.8) | 31.5 (32.5) | 41.4 (45.6) | 56.1 (59.9) | 54.4 (57.6) | 9.5 | 1.62 |
| 160 | 16 | 24.5 (28.6) | 42.9 (45.7) | 60.6 (65.3) | 87.2 (88.6) | 68.8 (74.7) | 14.3 | 3.77 |
| 672 | 1 | 8.1 (8.4) | 11.7 (12.0) | 13.8 (14.2) | 17.2 (20.6) | 24.3 (24.8) | 2.6 | 0.42 |
| 672 | 2 | 8.9 (9.3) | 16.0 (16.4) | 22.8 (24.2) | 28.8 (30.4) | 34.9 (38.0) | 4.2 | 0.58 |
| 672 | 4 | 12.0 (13.1) | 25.4 (26.3) | 33.7 (35.8) | 41.1 (44.8) | 43.0 (46.8) | 5.7 | 0.92 |
| 672 | 8 | 18.8 (20.8) | 40.5 (42.1) | 48.2 (50.8) | 64.2 (67.1) | 66.8 (71.3) | 9.5 | 1.62 |
| 672 | 16 | 28.5 (29.3) | 55.8 (59.1) | 74.9 (80.4) | 102.5 (108.7) | 94.7 (100.0) | 14.3 | 3.77 |
| 1312 | 1 | 8.3 (8.7) | 12.9 (13.8) | 16.7 (18.1) | 19.4 (20.9) | 26.0 (27.7) | 2.6 | 0.42 |
| 1312 | 2 | 10.4 (12.0) | 18.8 (20.5) | 25.7 (28.6) | 32.2 (33.5) | 39.3 (42.9) | 4.2 | 0.58 |
| 1312 | 4 | 12.7 (13.8) | 27.9 (30.1) | 42.0 (46.9) | 47.1 (51.1) | 50.2 (54.6) | 5.7 | 0.92 |
| 1312 | 8 | 20.7 (22.1) | 47.7 (49.9) | 64.1 (68.5) | 76.5 (80.5) | 82.8 (95.0) | 9.5 | 1.62 |
| 1312 | 16 | 34.4 (35.7) | 71.5 (78.3) | 98.3 (105.5) | 132.4 (142.0) | 137.6 (148.1) | 14.3 | 3.77 |

Evaluation (`results/m5max-m6b-adaptive-03`), the M4d protocol with the
speculative server in the adaptive mode (K=7 configured, both artifacts
bound), against the synchronous target-only server, five paired
repetitions per cell:

| Server | Bucket | C | Target-only tok/s | Candidate tok/s | Tokens/s benefit (median, 95% CI) | TPOT ms | Gap p95 ms | Goodput req/s | Accepted / drafted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| k7-adaptive | 128x128 | 1 | 136.6 | 197.8 | +40.5% [+35.2, +45.5] ✓ | 7.0 → 4.7 | 7 → 18 | 1.07 → 1.55 | 510 / 690 |
| k7-adaptive | 128x128 | 4 | 364.6 | 329.8 | -11.0% [-11.5, -6.6] | 9.8 → 11.0 | 10 → 11 | 2.85 → 2.58 | 0 / 12 |
| k7-adaptive | 128x128 | 8 | 439.1 | 412.5 | -6.4% [-9.7, -5.4] | 16.4 → 17.8 | 17 → 18 | 3.43 → 3.22 | 0 / 12 |
| k7-adaptive | 1024x128 | 1 | 106.0 | 106.2 | +1.5% [-4.7, +6.7] | 7.8 → 7.5 | 8 → 19 | 0.83 → 0.83 | 360 / 576 |
| k7-adaptive | 1024x128 | 4 | 200.1 | 182.5 | -8.2% [-9.1, -4.4] | 14.2 → 15.6 | 13 → 15 | 1.17 → 1.07 | 6 / 12 |
| k7-adaptive | 1024x128 | 8 | 215.3 | 172.1 | -20.1% [-20.7, -18.7] | 26.3 → 25.8 | 59 → 79 | 0.00 → 0.00 | 27 / 33 |
| k7-adaptive | 128x512 | 1 | 126.4 | 168.4 | +33.3% [+28.5, +38.7] ✓ | 7.7 → 5.8 | 8 → 22 | 0.25 → 0.33 | 2040 / 2964 |
| k7-adaptive | 128x512 | 4 | 329.7 | 306.9 | -7.1% [-7.6, -6.8] | 11.8 → 12.7 | 13 → 14 | 0.64 → 0.60 | 0 / 12 |
| k7-adaptive | 128x512 | 8 | 417.9 | 389.9 | -6.4% [-6.8, -6.3] | 18.7 → 20.0 | 20 → 22 | 0.82 → 0.76 | 0 / 12 |

Fixed K=2 and K=7 at eight clients under the same protocol
(`results/m5max-m6b-adaptive-01/bench-fixed-c8`), for comparison:

| Server | Bucket | C | Target-only tok/s | Candidate tok/s | Tokens/s benefit (median, 95% CI) | TPOT ms | Gap p95 ms | Goodput req/s | Accepted / drafted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| k2 | 128x128 | 8 | 423.8 | 382.2 | -10.0% [-12.5, -7.4] | 17.1 → 18.3 | 17 → 43 | 3.31 → 2.99 | 3478 / 5167 |
| k2 | 1024x128 | 8 | 225.6 | 166.2 | -26.3% [-27.9, -26.0] | 25.2 → 23.2 | 57 → 155 | 0.00 → 0.00 | 3430 / 5296 |
| k2 | 128x512 | 8 | 434.6 | 405.3 | -6.8% [-6.9, -6.1] | 18.0 → 18.1 | 20 → 48 | 0.85 → 0.79 | 14391 / 20241 |
| k7 | 128x128 | 8 | 389.9 | 348.2 | -12.6% [-15.9, -10.5] | 18.7 → 17.6 | 20 → 61 | 3.05 → 2.72 | 4116 / 13410 |
| k7 | 1024x128 | 8 | 212.3 | 134.3 | -37.0% [-43.5, -35.7] | 25.7 → 30.9 | 62 → 215 | 0.00 → 0.00 | 3975 / 14503 |
| k7 | 128x512 | 8 | 413.5 | 385.7 | -8.1% [-10.9, -4.2] | 18.9 → 16.7 | 21 → 67 | 0.81 → 0.75 | 17536 / 48718 |

At one request the adaptive mode gains +40.5% on the 128x128 bucket and
+33.3% on 128x512 (fixed K=2 gave +48.6% and +35.9% under the same protocol
in M4d, fixed K=7 +13.7% and +25.8%) with acceptance of 69% to 74% per draft
token, because the planner keeps prefixes short (three to four positions);
on the 1,024-token bucket it drafts but the end-to-end gain is within noise
(+1.5%), since the per-step ratio it optimizes is diluted by the prompt's
prefill in this bucket's wall time. At four and eight requests the planner
bypasses drafting in every bucket except a few steps on the 1,024-token
bucket at eight requests (33 draft tokens over the five repetitions), and
the server then sits on the bypass floor: 6% to 11% below the target-only
server on the 128-token buckets and 20% below it on the 1,024-token bucket
at eight requests, where fixed K=7 lost 8% to 37% and fixed K=2 7% to 26%
(and 18% to 28% at four requests in M4d).
That floor is not the planner's cost: with speculative decoding configured
the decode pipeline is off and target features are captured every step,
which is why a bypassing speculative server cannot match a target-only one;
it is recorded as the first M7 optimization target. The serving-path cost
model explains the M4d and first-evaluation results that the in-process
model contradicted: through `vllm serve`, a drafted step at sixteen
requests and 1,300 tokens of context costs 138 ms against 34 ms for the
bypass step, and the planner therefore drafts only at one or two
concurrent requests (the cost table above). The M6a in-process table's
40 ms for that width and request count was measured at the natural
prompts' short contexts, not at 1,300 tokens; the M7 probe measured the
same cell in process at 114 ms, so the serving path adds about a fifth
and context length accounts for the rest. The specification's adaptive criterion (goodput and p95 within
5% of target-only, or an explicit bypass) is met by the explicit bypass at
four and eight requests with the floor recorded, and by the one-request
gains that exceed the 10% gate with the interval above zero.

Gates in the adaptive mode. The HTTP serving matrix (K=0 and K=7 servers
with and without prefix caching, the mode bound to the K=7 servers) passed
with zero failures. The stochastic distribution gate needs the planner to
draft, which it declines at two or more concurrent requests on this cost
model, so it ran at one sequence: every request drafted (4,000 drafts per
scenario, 54% and 76% accepted at temperature 1.0 and at 0.7 with top-p
0.9), the mixed batch drafted every draftable kind and none of the
undraftable ones, greedy outputs matched the target-only engine (five equal,
two ties), budgets, EOS and stop tokens held and the seeded repeat was
identical; at temperature 1.0 every position passed, and at temperature
0.7 with top-p 0.9 the second verified position failed the predefined test
(chi-square 129 over 26 buckets). The histograms show a nucleus-boundary
flip rather than a verifier defect: one token (`1246`) carries 1.6% of the
mass on the target-only engine and none on the speculative engine, and its
neighbour (`4263`) the reverse, because at one sequence the target-only
engine decodes single-row logits while verification computes two-row
logits, the two arithmetic paths of the M4a numerics contract, and top-p
truncation turns their sub-ULP logit difference into a token that is inside
the nucleus on one path and outside on the other. Without truncation
(temperature 1.0) the same run passes at every position, and the fixed-mode
gate at sixty-four sequences (M5) passed both scenarios because both
engines then batch. The numerics control confirms
it (`results/m5max-m6b-adaptive-03/stochastic-adaptive-1seq-control`): the
same target-only engine run at sixty-four sequences against itself at one
sequence, with no drafting anywhere, differs at temperature 0.7 with top-p
0.9 at every position (chi-square 257, 217 and 179; p below 1e-23, position
0 included) and agrees at temperature 1.0 at every position (p 0.12 to
0.79). The speculative engine's one failing position is a smaller version
of the target's own disagreement with itself across the single-row and
batched paths; the verifier and the adaptive mode add nothing to it, and
the gate's own control column is the way to read such a result.

## M7: production hardening (M5 Max, `5b2bd1e`)

Profile before optimization. The M4d step profile, the M6 cost model and the
M6b evaluation locate DSpark's serving cost in three places, and the two
probes recorded here separate them. First, context: the in-process cost
table of M6a was taken at the natural prompts' short contexts, and its
40 ms engine step for K=7 at sixteen requests became 114 ms at 1,300 tokens
of context in process (`results/m5max-m7-served-gap-01`, the engine-core
step timed over the all-decoding half of the run), against 138 ms through
`vllm serve` at the same cell; the M6b record's sentence that put the
served step at about three times the in-process one compared cells at
different contexts and is corrected here: the serving path adds about a
fifth, and the multiprocess engine core and request statistics add nothing
(decode throughput 338, 346 and 347 tokens per second with the core in
process, in its own process, and with statistics on). Second, the target
forward that verifies `K+1` rows per request: the matmul probe
(`tools/dspark_matmul_rows_probe.py`) times the pinned target's own affine-4
linear layers at 1 to 32 query rows. The affine-4 linear layers are memory-bound only up to about four rows: one decoder layer's MLP takes 0.23 ms for one row, 0.26 for four, 0.42 for eight and 0.58 for thirty-two (2.5 times the one-row cost for thirty-two times the rows), the attention projections follow the same shape, and the language-model head goes from 0.61 ms at one row through 1.04 at eight to 1.45 at thirty-two. Past four rows each extra row adds a near-constant cost, about 7 µs per row for the MLP, so a K=7 verification (eight rows per request) costs the target roughly twice a single-row decode per request, and sixteen such requests (128 rows) sit well inside the compute-bound regime of the quantized matmul: the multi-row verification cost the M4d and M6 profiles measured is the matmul path, not attention, and it is inherent to this kernel rather than to the port. The probe's first pass showed the one-row path 2.6 times slower than two rows until its kernel variant had been compiled once (`results/m5max-m7-matmul-01`); the recorded numbers (`results/m5max-m7-matmul-02`) are a second pass after warming every row count:

| Rows | MLP ms | MLP ratio | QKV ms | O ms | LM head ms | LM head ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.230 | 1.00 | 0.181 | 0.174 | 0.610 | 1.00 |
| 2 | 0.240 | 1.04 | 0.188 | 0.168 | 0.582 | 0.95 |
| 4 | 0.263 | 1.14 | 0.188 | 0.172 | 0.810 | 1.33 |
| 8 | 0.416 | 1.81 | 0.241 | 0.224 | 1.040 | 1.71 |
| 12 | 0.441 | 1.91 | 0.223 | 0.200 | 1.555 | 2.55 |
| 16 | 0.536 | 2.33 | 0.269 | 0.225 | 1.390 | 2.28 |
| 24 | 0.536 | 2.33 | 0.290 | 0.268 | 1.465 | 2.40 |
| 32 | 0.578 | 2.51 | 0.347 | 0.270 | 1.452 | 2.38 |

Third, the
structural cost of a speculative server that does not draft: with
speculative decoding configured the Metal decode pipeline is disabled and
target features are captured and ingested every step, so the bypass step
runs 6% to 11% below a target-only server on the 128-token buckets and 20%
below it at eight requests on the 1,024-token bucket (M6b). That bypass
floor, the affine-4 multi-row matmul path and the fifth the serving path
adds are the optimization targets recorded here; none is changed in this
milestone, which qualifies the serving path as it is.

`tools/dspark_soak.py` is the HTTP soak. It launches one `vllm serve` (any
mode through the environment) and drives it with a closed-loop client pool
for at least the requested duration and request count: greedy and sampled
requests (temperature 0.8, half of them seeded with top-p 0.95), output
budgets from 8 to 256 tokens, natural prompts, long documentation windows
that need several prefill chunks, a shared prefix on a fifth of the
requests, streaming and non-streaming calls, and streamed requests the
client abandons after one to four chunks (cancellations). It samples the
server process tree's resident memory and the `/metrics` counters every
fifteen seconds and requires at the end that no request failed, the server
holds no running or waiting request, draft work was reported and kept
flowing through the last quarter of the run, and records latency
percentiles, throughput, the cancellation count, the memory trajectory and
the spec-decode counters. The proposer now logs its counters snapshot once
per 2,000 drafting steps (bypass reasons, proposed, scheduled and accepted
tokens, per-position acceptance, planner lengths) so an operator can read
the mode's behaviour from the server log without per-step logging.

The soak runs the fixed mode at K=7, the demanding path (every step
drafts and verifies eight rows per request, proposal records and their
distributions turn over on every stochastic request, cancellations land
mid-draft); in the adaptive mode at eight clients the planner bypasses
nearly every step (M6b), so a soak there would exercise only the bypass
path. The first attempt (`results/m5max-m7-soak-01`) lost its server after
51 minutes and 10,264 requests to a Metal command-buffer OOM
(`kIOGPUCommandBufferCallbackErrorOutOfMemory`) at the moment a full test
suite with end-to-end engine tests ran on the same GPU (the suite logged the
same error); the soak tool now ends the run as soon as the server process
exits, with the exit code and the moment, and the recorded run below had the
GPU to itself. Result (`results/m5max-m7-soak-02`, eight concurrent
clients, 4B pair):

| Duration | Requests (completed / cancelled / errors) | Output tokens | Requests/s | Tokens/s | TTFT p50 / p95 s | E2E p50 / p95 / p99 s | Draft / accepted tokens | RSS first → peak → last |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 65.0 min | 13,289 (12,220 / 1,069 / 0) | 1,009,629 | 3.40 | 258.7 | 0.200 / 0.455 | 1.57 / 7.75 / 9.44 | 2,303,046 / 649,888 (28.2%) | 4.38 GB → 4.38 GB → 4.28 GB |

The 260 fifteen-second samples put the draft-token counter at 0, 1,162,997 and 2,300,418 tokens at the start, middle and end of the run (never idle), at most 8 requests running and 2 waiting at any sample, and the failure list empty: none.

Packaged deployment (`scripts/test.sh` wheel build) needs Xcode's Metal
compiler, which this machine does not have; the source-built kernels
(`VLLM_METAL_BUILD_FROM_SOURCE=1`) served every run in this record. Packaging
validation stays deferred to a machine with Xcode and is listed as such in the
handoff.

## M9: performance parity work (M5 Max, `fe09b88`)

The official DSpark runners (vLLM's GPU speculator and SGLang's) keep the
drafter's context K/V in a paged pool that the attention kernel reads through
block tables, so a drafting step touches each request's context once and pays
nothing for the other rows of the batch; their planners (vLLM's adaptive
verification manager, SGLang's planner) choose the drafted prefix per request
from confidences against measured draft and verify cost curves, the objective
being emitted tokens per unit of step cost. The Metal port had the second
half (M6) but not the first: every draft context was a private buffer, and a
batched drafting step padded every request's context to the longest one and
concatenated the copies, per layer, per step (two operations per request and
layer), and the per-step feature ingest ran the fuse, projection, RoPE and
two writes per request and layer. The M4d and M6 profiles recorded the result:
the draft phase grew from 2.6 ms at one request to 14 ms at sixteen, and the
ingest and context evaluation with it. This milestone removes those costs and
measures what remains between this port and a speculative server that never
loses to target-only serving.

### Context arena and per-row attention

`vllm_metal/v1/dspark/model.py` now holds one `ContextArena` per drafter layer:
a `[slots, kv_heads, capacity + block, head_dim]` K and V pair reserved once
at load for `max_contexts` slots (the memory plan's `context_bytes` now counts
the block's scratch positions). A request's context is an `ArenaCache` over one
slot; appends write in place, and the slot returns to the free list when the
request finishes, is preempted or loses its prefix. On a drafting step the
block's own keys and values go to the scratch positions right after each
row's context (one scatter per tensor and layer for the whole batch), and each
row attends to a strided view of its own slot, `[0, length + block)`, with no
gather, padding or mask: a batch costs the sum of its rows and a short context
never pays for the longest one. Measured in process on the real 4B pair
with synthetic contexts, all draft phases in one evaluation, the old and the
new path back to back on an idle GPU (`results/m5max-m9-ab-02`,
`draft-breakdown-old` and `draft-breakdown-q4`), the drafting step goes
from 3.74 to 3.29 ms at one request, 6.43 to 5.52 ms at four, 11.70 to
10.06 ms at sixteen requests with 300 tokens of context and 20.62 to
13.51 ms at sixteen requests with 1,300 tokens: the 4.3 ms of padding and
concatenation at the long context are gone and the backbone itself is
faster because every row reads only its own length. A third design, one
gather of the batch's rows into a padded buffer with a length mask, was
measured on the way and rejected: it lost what it saved because MLX
materializes a `keys[rows, :, :width]` index as a full-slot gather, while
`scaled_dot_product_attention` over a strided prefix view costs the same as
over a contiguous copy (0.36 versus 0.39 ms for sixteen rows at 307 keys),
so per-row attention needs no copy at all.

### Batched feature ingest

Every decode span of a step (at most `K + 1` accepted rows per request) is
now written by one pass per layer: the spans' feature rows are gathered from
the packed hidden states into `[rows, width, features]` (each span padded by
repeating its last row), fused and projected once, roped with a per-row start
position (`mx.fast.rope` takes an offset array), and scattered into the arena
at each row's positions. The padding columns land past the row's committed
length, where the next append or the block write overwrites them and no
reader looks before then; the arena rejects a span that would leave its slot.
Prefill chunks keep the per-request path (they are few per step and long).
In process at sixteen requests with 79 accepted rows the ingest drops from
8.6 to 2.2 ms per step (4.5 to 1.9 ms at eight, 2.3 to 1.2 ms at four); one
request keeps the single-row path. The batched projection is the multi-row
arithmetic path of the M4a numerics contract: against the sequential writes
the context differs by one to three bfloat16 ULPs on the real drafter, and
the FP32 unit test (`MLX_ENABLE_TF32=0`) asserts equality to 2e-5.

Through the serving path, with the M6 cost protocol at the same cells as
the M6b cost model (`results/m5max-m9-ab-02/cost-window-off` against
`results/m5max-m6-cost-03`; K=7, memory fraction 0.4, 64-token outputs),
the two changes take 14% to 21% off the drafting step at 160 tokens of
decode context (four requests 35.2 to 28.9 ms, eight 54.4 to 43.2 ms,
sixteen 68.8 to 57.4 ms) and 9% to 16% at 1,312 tokens (four 50.2 to
43.2 ms, eight 82.8 to 74.9 ms, sixteen 137.6 to 116.1 ms), with the p95
step now within 2% to 5% of the median where it had been 5% to 15% above
it; the bypass step, which ingests but never drafts, is 2% to 6% cheaper.
The in-process drafter work behind those cells went from 14.3 to 9.0 ms
at sixteen requests (9.5 to 7.7 ms at eight, 5.7 to 5.3 ms at four) and
its host bookkeeping from 3.8 to 1.8 ms.

### The verification kernel's window mode

The target's paged-attention kernel has a window mode
(`VLLM_METAL_SPEC_VERIFY_WINDOW=1`, opt-in and off by default because its
win is chip- and shape-dependent) in which the `K+1` verification rows of a
request share each KV block load instead of re-reading the context per row;
its outputs are bitwise identical to the expanded per-row layout. Measured
through the serving path with the M6 cost protocol at K=7 (`results/m5max-m9-ab-02`,
`cost-window-off3` and `cost-window-on3`; four, eight and sixteen requests
at 160 and 1,312 tokens of decode context, memory fraction 0.4, 64-token
outputs, the bypass server of each run as the control that the flag cannot
affect):

| Decode context | Requests | Bypass step ms, off / on (control) | K=7 step ms, window off (p95) | K=7 step ms, window on (p95) | Window effect |
| --- | --- | --- | --- | --- | --- |
| 160 | 4 | 9.3 / 9.1 | 29.6 (30.1) | 30.8 (31.2) | +4% |
| 160 | 8 | 13.6 / 13.5 | 44.4 (45.0) | 46.2 (47.2) | +4% |
| 160 | 16 | 22.6 / 22.6 | 59.4 (62.5) | 61.5 (66.8) | +3% |
| 1312 | 4 | 12.2 / 12.3 | 44.5 (45.3) | 46.4 (48.0) | +4% |
| 1312 | 8 | 19.9 / 19.7 | 75.4 (78.3) | 78.2 (80.3) | +4% |
| 1312 | 16 | 31.5 / 32.7 | 116.8 (122.4) | 116.4 (121.9) | -0% |

The window mode costs 3% to 4% at four and eight requests on both contexts
and nothing at sixteen requests with 1,312 tokens, with the control within
1% to 4% between the two runs, so it stays off for DSpark on this machine:
the documented wins are at concurrency sixteen to thirty-two with 8k
contexts on Ultra chips, and these shapes are not that. Two earlier pairs
in the same directory (`cost-window-off`/`-on` and `-on2`/`-off2`) are kept
but not read: the second run of the first pair was uniformly slower,
control included, and the second pair ran while another session's test
suite loaded the machine (load average above thirty), which is what made
the quiet-machine gate (`bin/wait-quiet.sh`) a precondition of every timing
phase after it.

### The decode pipeline on non-drafting steps

The M6b bypass floor had two parts: with speculative decoding configured the
runner's one-step-ahead decode pipeline (`decode_pipeline.py`) was off for
the server's life, and every step captured and ingested target features.
The second part is the drafter's contract (a context that falls behind
cannot be rebuilt without replaying the prompt), and after the batched
ingest it costs about 2 ms at sixteen requests. The first part is now a
per-step decision. A proposer that can consume a step whose sampled tokens
stay on the device implements the `DeferredStepProposer` seam
(`vllm_metal/v1/proposer.py`): `deferred_step_allowed` says whether the
proposer will draft at the end of a pure-decode step, and
`ingest_deferred_step` receives such a step's features and segments without
token values. The DSpark proposer answers from its mode: the bypass mode and
a step without speculative tokens never draft; the fixed mode always drafts;
the adaptive mode runs the same draft-or-not decision `propose` takes, with
every draftable request at its full cap and the step's request count and
context, so a batch the planner declines at the gate it declines after
sampling too. The runner's gate (`_evaluate_pipeline_gate`) treats such a
proposer per step ("drafter may draft" blocks the step) instead of for the
server's life, and the deferred sample path hands the step to the proposer
after the placeholders are appended, so the feature rows end at each
request's pending anchor exactly as on the synchronous path; the proposer
spends the scheduled requests' proposal records, ingests the features and
queues the arenas for evaluation behind the step's own work. A verification
step, a new or resumed request and a drafting step keep the synchronous
path, and the synchronous path always resolves the pending step before it
mutates request state, so a drafting step never sees a placeholder anchor.
The seam is complete and unit-tested at every layer (the proposer's
decision per mode, the ingest without token values, the runner's gate and
its deferred submit); whether a served DSpark server can reach it is a
question the evaluation below answers, and the answer is not yet.

The decomposition of the floor, through the M4d serving protocol at four and
eight clients (`results/m5max-m9-ab-02/bench-pipeline-off`: the adaptive
server, which bypasses at these concurrencies, against a target-only server
started with `VLLM_METAL_DECODE_PIPELINE=0`, five paired repetitions):

| Server | Bucket | C | Target-only tok/s | Candidate tok/s | Tokens/s benefit (median, 95% CI) | TPOT ms | Gap p95 ms | Goodput req/s | Accepted / drafted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| k7-adaptive | 128x128 | 4 | 316.6 | 292.7 | -7.6% [-9.7, -3.3] | 11.3 → 12.1 | 12 → 13 | 2.47 → 2.29 | 0 / 12 |
| k7-adaptive | 128x128 | 8 | 393.9 | 373.6 | -4.8% [-5.8, -0.3] | 18.6 → 19.3 | 19 → 20 | 3.08 → 2.92 | 0 / 12 |
| k7-adaptive | 1024x128 | 4 | 174.8 | 124.3 | -29.0% [-30.3, -26.1] | 15.5 → 12.6 | 14 → 13 | 0.68 → 0.49 | 24 / 30 |
| k7-adaptive | 1024x128 | 8 | 200.8 | 126.7 | -36.4% [-39.7, -34.0] | 27.6 → 12.6 | 60 → 13 | 0.00 → 0.25 | 74 / 122 |

On the 128-token bucket the adaptive server's bypass steps run 7.6% (four
clients) and 4.8% (eight) below a target-only server that also has the
pipeline off, against 11.0% and 6.4% below one that has it on (M6b): the
pipeline is three and two points of the floor, the rest is the capture and
ingest of target features and the proposer's per-step bookkeeping. The
1,024-token bucket told a different story: 29% and 36% below target-only,
with a *better* time per output token (12.6 versus 15.5 and 27.6 ms) and a
time to first token of 1.6 and 3.5 s against 0.9 and 1.5 s. The speculative
server's engine log had the cause: "Running: 2 reqs, Waiting: 6 reqs". Its
target KV cache had 153 blocks (2,448 tokens, 1.2 sequences of the model
length) where the target-only server had 2,154, because the paged-attention
planner subtracted the drafter's whole reservation (context, capture,
workspace: 3.26 GB at these limits) from a budget measured after the context
arena had already been allocated, so the arena's 0.67 GB was counted twice,
and because the workspace itself reserved three full copies of the context
for the padded batch that the arena no longer builds (2.0 of its 2.5 GB).
The M6b evaluation had the same arithmetic without the arena (438 blocks,
7,008 tokens) and its 20% loss at eight clients on the 1,024-token bucket
was this starvation, not verification cost: eight requests of 1,150 tokens
need more cache than that. The planner now subtracts only the capture and
workspace once the arena exists (`planning_reserve_bytes`), and the
workspace reserves one transient copy of the arena instead of three; at
the bench's limits the speculative server's cache goes from 153 to
1,006 blocks.

The same protocol on the tree with every change of this milestone, against
the production target-only server, at one, four and eight clients, five
paired repetitions per cell (`results/m5max-m9-pipeline-01/bench-adaptive`):

| Server | Bucket | C | Target-only tok/s | Candidate tok/s | Tokens/s benefit (median, 95% CI) | TPOT ms | Gap p95 ms | Goodput req/s | Accepted / drafted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| k7-adaptive | 128x128 | 1 | 141.4 | 226.2 | +60.1% [+59.6, +61.4] ✓ | 6.8 → 4.1 | 7 → 15 | 1.11 → 1.77 | 510 / 690 |
| k7-adaptive | 128x128 | 4 | 379.4 | 360.9 | -6.7% [-9.4, -1.7] | 9.5 → 10.1 | 10 → 10 | 2.96 → 2.82 | 0 / 12 |
| k7-adaptive | 128x128 | 8 | 455.2 | 437.7 | -4.5% [-6.2, -2.1] | 16.0 → 16.6 | 16 → 17 | 3.56 → 3.42 | 0 / 12 |
| k7-adaptive | 1024x128 | 1 | 107.0 | 110.6 | +3.3% [-0.9, +8.8] | 7.7 → 7.1 | 8 → 18 | 0.84 → 0.86 | 360 / 576 |
| k7-adaptive | 1024x128 | 4 | 207.2 | 193.2 | -7.7% [-8.4, -4.7] | 13.6 → 14.6 | 12 → 13 | 1.61 → 1.13 | 6 / 12 |
| k7-adaptive | 1024x128 | 8 | 234.1 | 215.5 | -7.5% [-8.6, -5.5] | 24.5 → 26.7 | 56 → 126 | 0.00 → 0.00 | 8 / 26 |
| k7-adaptive | 128x512 | 1 | 130.2 | 173.9 | +33.5% [+31.9, +42.4] ✓ | 7.5 → 5.6 | 8 → 22 | 0.25 → 0.34 | 2040 / 2964 |
| k7-adaptive | 128x512 | 4 | 330.3 | 316.7 | -3.6% [-4.5, -3.5] | 11.8 → 12.3 | 13 → 13 | 0.65 → 0.62 | 0 / 12 |
| k7-adaptive | 128x512 | 8 | 408.8 | 391.6 | -4.1% [-4.2, -3.9] | 19.1 → 19.9 | 21 → 22 | 0.80 → 0.76 | 0 / 12 |

Against M6b (`results/m5max-m6b-adaptive-03`) the one-request gain on the
128-token bucket goes from +40.5% to +60.1% and the long-prompt bucket at
eight clients from -20.1% to -7.5%; the four- and eight-client cells on the
short buckets move from -11.0%/-6.4% to -6.7%/-4.5% (128 outputs) and from
-7.1%/-6.4% to -3.6%/-4.1% (512 outputs). The serving gate passes in the
bypass mode (every scenario equal or a tie, no draft token reported) and in
the adaptive mode (22 equal and 7 ties; 13 and 18 with prefix caching), so
the seam and the budget change alter no output.

What the seam did not deliver, and why, is part of this record. The
speculative server's log never showed the deferred-step line: the Metal
platform forces synchronous scheduling whenever speculative decoding is
configured (`vllm_metal/platform.py`, every Metal proposer hands its drafts
back through `take_draft_token_ids`, which vLLM's asynchronous engine loop
never calls), and the decode pipeline's own gate requires asynchronous
scheduling, so no DSpark server can reach the deferred path today. The
bypass floor that remains (4% to 8% at four and eight clients) is therefore
the feature capture and ingest, the proposer's bookkeeping, and the absent
pipeline; and the true production gap is larger than that floor, because a
target-only server with asynchronous scheduling (`--async-scheduling`, the
M4d `k0-async` reference) is itself 5% to 12% faster than the synchronous one
these tables compare against. vLLM's own answer, which the upstream GPU
runner uses for every drafter under asynchronous scheduling, is to let the
scheduler reserve `num_speculative_tokens` placeholder slots per running
request and to substitute the worker's drafts at execution time; the Metal
runner's speculative-decode contract is scheduler-driven and rejects
placeholder sentinels. Bringing DSpark under asynchronous scheduling on
Metal is the next piece of this work (M9b): it removes the platform's
synchronous downgrade for DSpark, admits the placeholder contract in the
runner's segment building and verification, and only then can the
deferred-step seam recorded above engage.

### Drafter precision

The released drafters are bfloat16 checkpoints; the port converts them to
the target's affine 4-bit recipe at load, and `VLLM_METAL_DSPARK_DRAFT_PRECISION=source`
now keeps the source precision instead. Measured back to back in process on
an idle GPU (`results/m5max-m9-ab-01`, `draft-breakdown-q4` and
`draft-breakdown-bf16`), the quantized drafter is faster at every batch
size: the drafting step takes 3.29 versus 5.10 ms at one request, 5.52
versus 7.82 ms at four, 10.06 versus 11.14 ms at sixteen requests with 300
tokens of context and 13.51 versus 14.43 ms at 1,300 tokens; at one request
the drafter is bound by its weight reads (1.2 GB in bfloat16 against 0.35 GB
quantized) and at sixteen the sequential head and the language-model head
still read their weights once per position. The accepted length does not pay for the
quantization either: on the official prompt sets at the paper's temperature
1.0 (below) the quantized drafter reaches 6.12, 5.69, 5.11, 4.81, 3.54 and
3.40 on gsm8k, math500, humaneval, mbpp, mt-bench and alpaca, the source
drafter 6.03, 5.71, 5.30, 4.93, 3.57 and 3.46 (`results/m5max-m9-accept-01`,
`t1-quantized` and `t1-source`, 64 prompts per set), differences within the
sampling noise of 64 prompts and in neither direction consistently; under
greedy decoding the source drafter is 0.02 to 0.12 higher on every set
(6.21 against 6.15 on gsm8k, 5.97 against 5.87 on math500), a real but
small cost of the 4-bit conversion. The quantized drafter stays the
default: it is faster at every batch size and the operator can trade the
memory for that fraction of a draft token with the knob.

### Accepted length against the paper

`tools/dspark_acceptance_eval.py` runs the DeepSpec evaluation prompt
sets (first turn, the target's chat template without thinking; 64 prompts
per set, 512-token outputs, eight concurrent requests in process, fixed
mode at K=7) and reports the accepted length per drafting round, the
drafted tokens accepted plus the bonus or correction token, the paper's
Table 1 quantity. The paper samples at temperature 1.0 with standard
rejection-sampling verification over the full sets with 2,048-token
outputs and bfloat16 weights; here the target is the qualified 4-bit
conversion and the drafter either quantized or in source precision
(`results/m5max-m9-accept-01`):

| Data set | Paper (bf16, full set, T=1.0) | T=1.0, quantized | T=1.0, source | greedy, quantized | greedy, source |
| --- | --- | --- | --- | --- | --- |
| gsm8k | 6.11 | 6.12 | 6.03 | 6.15 | 6.21 |
| math500 | 5.70 | 5.69 | 5.71 | 5.87 | 5.97 |
| humaneval | 5.38 | 5.11 | 5.30 | 5.25 | 5.37 |
| mbpp | 5.13 | 4.81 | 4.93 | 4.98 | 5.09 |
| mt-bench | 3.64 | 3.54 | 3.57 | 3.65 | 3.67 |
| alpaca | 3.54 | 3.40 | 3.46 | 3.44 | 3.47 |

At the paper's temperature the port lands within 0.1 of the paper on
gsm8k, math500, mt-bench and alpaca and within 0.3 on humaneval and mbpp
(the two code sets, where the 4-bit target's own answers differ most from
the bfloat16 target the drafter was trained against); greedy decoding is
0.1 to 0.2 higher on every set. The port's drafter therefore keeps the
official drafter's strength: the accepted lengths that give DSpark its
advantage over Eagle3 and DFlash in Table 1 (30.9% and 16.3% higher on the
4B target) are reproduced here with quantized weights and exact
verification; the source drafter is consistently 0.02 to 0.12 higher
under greedy decoding and indistinguishable at temperature 1.0. Throughput in this offline
setting (eight requests, 512-token outputs, one engine in process) was
458 to 909 output tokens per second depending on the set, the code and
math sets fastest because their longer accepted prefixes make each
verification step worth more tokens.

### Cost model and serving results after the changes

The serving-path cost model was re-profiled on the optimized tree with the
M6 protocol (`results/m5max-m9-final-01/cost-04`: one to sixteen requests,
widths 0 to 7, decode contexts 160, 672 and 1,312 tokens, memory fraction
0.4) and the adaptive mode was evaluated with it through the M4d serving
protocol against the production target-only server at one, four, eight and
sixteen clients, five paired repetitions per cell
(`results/m5max-m9-final-01/bench-adaptive`), with the fixed mode at K=2
and K=7 at one and eight clients for comparison (`bench-fixed`):

Served step cost in milliseconds (median) per cell, the M6b profile
(`m5max-m6-cost-03`) against this one (`cost-04`), the drafted widths shown
for K=2, 4 and 7 of the seven profiled, and the in-process drafter work
the profile attributes at K=7:

| Decode context | Requests | Bypass (M6b -> M9) | K=2 | K=4 | K=7 | Draft ms (in process) |
| --- | --- | --- | --- | --- | --- | --- |
| 160 | 1 | 7.4 -> 7.2 | 11.2 -> 11.0 | 14.3 -> 12.5 | 19.4 -> 19.4 | 3.4 |
| 160 | 2 | 8.2 -> 7.9 | 20.3 -> 15.9 | 25.9 -> 25.8 | 31.5 -> 30.9 | 4.6 |
| 160 | 4 | 9.5 -> 9.2 | 30.3 -> 24.0 | 40.8 -> 35.0 | 35.2 -> 34.1 | 5.7 |
| 160 | 8 | 14.5 -> 13.6 | 41.4 -> 35.7 | 56.1 -> 52.2 | 54.4 -> 50.7 | 8.2 |
| 160 | 16 | 24.5 -> 22.4 | 60.6 -> 51.9 | 87.2 -> 77.2 | 68.8 -> 62.5 | 10.7 |
| 672 | 1 | 8.1 -> 8.0 | 13.8 -> 13.0 | 17.2 -> 16.7 | 24.3 -> 22.6 | 3.4 |
| 672 | 2 | 8.9 -> 8.8 | 22.8 -> 20.4 | 28.8 -> 27.5 | 34.9 -> 33.6 | 4.6 |
| 672 | 4 | 12.0 -> 11.5 | 33.7 -> 31.4 | 41.1 -> 38.5 | 43.0 -> 39.5 | 5.7 |
| 672 | 8 | 18.8 -> 18.1 | 48.2 -> 43.5 | 64.2 -> 59.1 | 66.8 -> 60.3 | 8.2 |
| 672 | 16 | 28.5 -> 27.6 | 74.9 -> 63.9 | 102.5 -> 89.4 | 94.7 -> 83.4 | 10.7 |
| 1312 | 1 | 8.3 -> 8.0 | 16.7 -> 15.0 | 19.4 -> 18.4 | 26.0 -> 25.0 | 3.4 |
| 1312 | 2 | 10.4 -> 9.6 | 25.7 -> 22.9 | 32.2 -> 29.7 | 39.3 -> 37.0 | 4.6 |
| 1312 | 4 | 12.7 -> 12.1 | 42.0 -> 34.1 | 47.1 -> 43.4 | 50.2 -> 46.2 | 5.7 |
| 1312 | 8 | 20.7 -> 19.7 | 64.1 -> 50.7 | 76.5 -> 68.1 | 82.8 -> 73.9 | 8.2 |
| 1312 | 16 | 34.4 -> 32.0 | 98.3 -> 80.7 | 132.4 -> 112.8 | 137.6 -> 115.8 | 10.7 |

Every drafted step is cheaper on the optimized tree, by 5% to 20%, most at
sixteen requests and the long context (K=7 137.6 to 115.8 ms, K=2 98.3 to
80.7 ms); the bypass step by 2% to 9%. One shape of the target is worth
naming because the planner's table now contains it: at eight and sixteen
requests K=4 (five rows per request, 80 rows at sixteen) costs more than
K=7 (128 rows), in both profiles, the affine-4 kernels' tiling favouring
the wider batch; the cost model interpolates over the measured rows and the
planner reads the measured surface, so it never picks the width by
assumption.

Adaptive mode (K<=7, the calibrated planner with the cost-04 table) against
the production target-only server, output tokens per second, median of five
paired repetitions with the bootstrap 95% interval of the relative benefit.
Three runs are shown because the machine was shared: `final-01` (17:29,
another session's bursts hit the cells marked contended), `pipeline-01`
(16:44, the same tree before the KV-budget fix landed in the final tree;
its 128x512 cells are the clean ones) and `final-02` (22:04, the
contamination-aware protocol: a pair taken while another process ran above
40% CPU or the load average exceeded 4 is discarded and repeated, up to five
extra pairs per cell, then kept flagged; the last column records how many
pairs each cell discarded and kept flagged and the highest load average
among kept pairs). Under outside load both servers slow down together, so
the relative benefit holds where the absolute rates do not:

| Bucket | C | First run (final-01), sync | Pipeline-01, sync | Final-02 under outside load (protocol: discarded / flagged pairs) |
| --- | --- | --- | --- | --- |
| 128x128 | 1 | 141.2 → 213.7, +51.5% [+50.3, +52.1] | 141.4 → 226.2, +60.1% [+59.6, +61.4] | 131.5 → 194.5, +50.1% [+41.5, +57.0] (0 discarded, 0 flagged, load1 max 3.7) |
| 128x128 | 4 | 380.9 → 356.6, -7.8% [-15.2, +3.9] (contended) | 379.4 → 360.9, -6.7% [-9.4, -1.7] | 344.9 → 332.0, -5.8% [-18.2, +13.0] (0 discarded, 0 flagged, load1 max 3.9) |
| 128x128 | 8 | 440.0 → 434.9, -0.3% [-10.2, +0.9] | 455.2 → 437.7, -4.5% [-6.2, -2.1] | 383.9 → 361.2, -5.9% [-28.0, +38.0] (5 discarded, 2 flagged, load1 max 4.5) |
| 128x128 | 16 | 586.4 → 562.5, -4.8% [-8.9, -1.6] | – | 490.6 → 455.2, -7.2% [-22.4, +14.1] (5 discarded, 1 flagged, load1 max 4.1) |
| 1024x128 | 1 | 107.0 → 113.9, +9.3% [-7.0, +20.6] | 107.0 → 110.6, +3.3% [-0.9, +8.8] | 93.3 → 99.5, +6.2% [-27.3, +33.4] (5 discarded, 1 flagged, load1 max 6.1) |
| 1024x128 | 4 | 202.3 → 187.8, -6.4% [-11.2, -2.3] | 207.2 → 193.2, -7.7% [-8.4, -4.7] | 166.9 → 160.9, -3.6% [-25.5, +27.1] (5 discarded, 5 flagged, load1 max 5.8) |
| 1024x128 | 8 | 232.5 → 214.9, -7.6% [-9.9, -4.3] | 234.1 → 215.5, -7.5% [-8.6, -5.5] | 200.9 → 188.2, -6.3% [-20.6, +3.3] (5 discarded, 4 flagged, load1 max 4.7) |
| 1024x128 | 16 | 198.6 → 183.2, -6.4% [-30.6, +23.1] (contended) | – | 221.2 → 199.8, -9.5% [-17.5, +1.8] (0 discarded, 0 flagged, load1 max 3.7) |
| 128x512 | 1 | 56.6 → 67.8, +23.3% [-17.8, +97.7] (contended) | 130.2 → 173.9, +33.5% [+31.9, +42.4] | 123.2 → 160.7, +28.3% [+24.9, +47.6] (0 discarded, 0 flagged, load1 max 3.6) |
| 128x512 | 4 | 182.2 → 170.6, -6.6% [-32.9, +27.2] (contended) | 330.3 → 316.7, -3.6% [-4.5, -3.5] | 299.9 → 318.5, +6.2% [-22.1, +12.4] (5 discarded, 5 flagged, load1 max 4.6) |
| 128x512 | 8 | 268.2 → 258.0, -4.3% [-24.3, +22.0] (contended) | 408.8 → 391.6, -4.1% [-4.2, -3.9] | 421.4 → 419.0, -1.3% [-14.1, +7.3] (5 discarded, 3 flagged, load1 max 5.9) |
| 128x512 | 16 | 477.3 → 482.0, +4.4% [-11.9, +10.0] (contended) | – | 532.5 → 517.5, -0.6% [-15.5, +4.9] (5 discarded, 5 flagged, load1 max 5.1) |

Fixed mode, K=2 and K=7 at one and eight clients (`final-01/bench-fixed`,
five paired repetitions, no contamination record in that run):

| Server | Bucket | C | Target-only tok/s | Candidate tok/s | Tokens/s benefit (median, 95% CI) | TPOT ms | Gap p95 ms | Goodput req/s | Accepted / drafted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| k2 | 128x128 | 1 | 138.4 | 203.3 | +49.7% [+45.6, +56.6] ✓ | 7.0 → 4.6 | 7 → 12 | 1.08 → 1.59 | 450 / 618 |
| k2 | 128x128 | 8 | 459.8 | 436.8 | -8.8% [-13.1, +18.1] | 15.0 → 15.8 | 15 → 40 | 3.59 → 3.41 | 3560 / 5042 |
| k2 | 1024x128 | 1 | 104.7 | 122.6 | +15.8% [-3.7, +19.9] | 7.9 → 6.2 | 8 → 16 | 0.82 → 0.96 | 438 / 630 |
| k2 | 1024x128 | 8 | 227.2 | 194.2 | -14.5% [-27.1, +7.1] | 24.1 → 26.8 | 53 → 159 | 0.00 → 0.00 | 3296 / 5544 |
| k7 | 128x128 | 1 | 137.5 | 194.2 | +41.8% [+35.5, +43.8] ✓ | 7.0 → 4.8 | 7 → 18 | 1.07 → 1.52 | 540 / 1524 |
| k7 | 128x128 | 8 | 454.8 | 389.0 | -12.8% [-27.3, -6.0] | 15.1 → 16.5 | 16 → 50 | 3.55 → 3.04 | 4102 / 13651 |
| k7 | 1024x128 | 1 | 106.1 | 90.5 | -14.4% [-26.3, -1.7] | 7.7 → 9.0 | 8 → 24 | 0.83 → 0.71 | 456 / 2070 |
| k7 | 1024x128 | 8 | 238.6 | 173.2 | -27.6% [-28.9, -23.7] | 23.5 → 31.0 | 54 → 200 | 0.00 → 0.00 | 3866 / 15191 |

The `final-02` fixed bench (23:15) ran entirely under outside load: every
cell discarded five pairs and kept three to five flagged ones with load
averages up to 5.7, and the outside work fell on the candidate more than
on the reference (K=7 at one client on the short prompts read +5.7%
against +41.8% in `final-01`). Its numbers are kept with their load record
in `results/m5max-m9-final-02/bench-fixed` as the protocol's own evidence
of contamination and are not used above.

Read across the three runs, the picture is stable. At one client on the
short prompts the adaptive server is 50% to 60% faster than target-only
(M6b: +40.5%), on the 512-token outputs 28% to 34% (M6b: +33.3%), and on
the 1,024-token prompts 3% to 9% with intervals that include zero. From four
clients on the planner bypasses on nearly every step (the accepted/drafted
column shows a few dozen draft tokens per cell) and the server runs 1% to 9%
behind target-only, where M6b lost 6% to 20%; at eight clients on the long
prompts the loss went from 20.1% to 6.3-7.6% because the KV starvation is
gone, and the remaining floor is the per-step feature capture and ingest
plus the synchronous scheduler the M9b record removes. In fixed mode K=2
gains 49.7% at one client on the short prompts and 15.8% on the long ones
and K=7 gains 41.8% on the short prompts, while forced K=7 loses 14.4% on
the long prompts at one client and 12.8% to 27.6% at eight clients, the
cost-04 rows above (a K=7 step at eight requests costs 3.7x a decode) made
visible; the M4d fixed numbers were +13.7% and -15.3% at one client on the
same two buckets, so the arena and ingest work is worth 28 points on the
short prompts at K=7 and the long-prompt loss is unchanged because it is the
target's verification cost, not the drafter's.

Gates on the optimized tree: serving check in bypass mode
(`results/m5max-m9-pipeline-01/serving-bypass`: every scenario equal or a
tie against target-only, zero draft tokens reported, as `--expect-no-drafts`
requires), in adaptive mode (`serving-adaptive` there and in
`results/m5max-m9-final-01`: 22 equal and 7 ties on the direct server, 13
equal and 18 ties on the prefix-cached one, 934 drafts / 2,760 draft tokens /
1,554 accepted in the final run, zero failures) and in fixed mode
(`serving-fixed`: 20 equal and 9 ties, 14 and 17 on the prefix server, 823
drafts / 5,645 draft tokens / 1,723 accepted, zero failures); the stochastic
gate in fixed mode (`stochastic-fixed`: all six distribution cells pass at
4,000 samples, 3,943 drafted / 2,018 accepted at temperature 1.0 and 3,994 /
2,870 at temperature 0.7 with top-p 0.9, greedy parity 3 equal and 4 ties,
seeded runs reproducible). The adaptive stochastic check at one sequence
(`stochastic-adaptive-1seq`) reproduces the M6b run histogram for histogram
(4,000 drafted / 2,153 accepted at temperature 1.0, 4,000 / 3,029 at 0.7 with
top-p 0.9), including the one cell that run already recorded as the target's
own nucleus flip between its single-row and multi-row forward (temperature
0.7, top-p 0.9, position 2, chi-square 129.2; the target-only control without
drafting fails the same cell at every position), so the M9 changes moved no
distribution.

### Where this leaves the port against the official runners

The official runners' advantage rests on three things: a drafter whose
context costs nothing per step beyond its own rows, a verification budget
chosen per step from calibrated confidences against a measured cost curve,
and a target whose eight verification rows per request are nearly free
because a datacenter GPU is memory-bound far past that batch size. The port
now has the first two on Metal (the arena and the deferred-step seam close
the drafter's per-step overheads and the bypass floor; M6 already chose the
budget the way the vLLM runner does, by expected accepted tokens per unit of
step cost over a profiled table), and its accepted length on the official
prompt sets is within 0.1 of the paper's Table 1 on gsm8k, math500, mt-bench and alpaca and within 0.3 on the two code sets at the paper's temperature, with a 4-bit target and drafter. The third is where the remaining gap sits,
and cost-04 locates it precisely: at one request the target's step costs
1.23x a single-row decode with five rows (K=4) but 1.76x with six and 2.18x
with eight (K=7). The M7 matmul probe shows the same knee in every affine-4
projection (the MLP 1.14x at four rows, 1.8x at eight, 2.3x at sixteen):
`mx.quantized_matmul` streams the weights once on its GEMV path up to five
rows and from six switches to a GEMM tiling that is mostly idle below about
sixteen rows, so eight verification rows cost the target about twice a
decode. That is why the planner's gain on this machine is concentrated at
low concurrency (+50% to +60% at one client on short prompts, +28% to +34% on 512-token outputs, +3% to +9% on 1,024-token prompts, and -1% to -9% at four to sixteen clients where the planner bypasses (M6b: -6% to -20%)), why fixed K=7 loses on the long prompts
at one client (a 19.4 ms round needs 2.7 accepted tokens to break even),
and why at high concurrency the right production behaviour is the one the
adaptive mode now delivers: bypass at no measurable cost against target-only
serving, drafting the moment the measured curve says a request would gain.
Part of that knee is the kernel rather than the hardware: mlx-dspark closed
it on M4-class machines with a small-M kernel that dequantizes each weight
group once and applies it to every row, and M9c brings that path to the
port, measured per weight shape at load. A larger target on the same machine
also moves the crossover up, because its decode is more weight-bound per
row; the M8 pairs measure that.

### Against the other Apple-silicon runtime

mlx-dspark (ARahim3, 0.19.0, MIT) is a standalone MLX server with its own
DSpark runtime, cost-curve cap and small-M verify kernel. Its `benchmark`
command and this repository's `tools/dspark_single_stream_bench.py` ran
the same protocol on this machine (`results/m5max-m9-compare-mlx-dspark-01`:
its three prompts, one stream, greedy, 200-token outputs, the median of
three timed passes after a warm-up), each on its own runtime:

| Runtime, target | Target-only tok/s | DSpark tok/s (accepted length) | Speedup |
| --- | --- | --- | --- |
| mlx-dspark, Qwen3-4B-8bit, auto cap | 106.5 | 168.4 (2.65) | 1.58x |
| mlx-dspark, Qwen3-4B-8bit, cap 7 | 106.5 | 138.1 (3.24) | 1.30x |
| mlx-dspark, Qwen3-4B-4bit, auto cap | 182.1 | 209.2 (2.59) | 1.15x |
| mlx-dspark, Qwen3-4B-4bit, cap 7 | 182.1 | 153.6 (2.97) | 0.84x |
| this port, 4-bit pair, K=4: chat / code / math | 142.8 / 142.3 / 138.5 | 217.6 (2.80) / 311.2 (4.42) / 317.0 (4.63) | 1.52x / 2.19x / 2.29x |
| this port, K=7: chat / code / math | same | 175.5 (3.06) / 313.6 (6.42) / 335.9 (6.86) | 1.23x / 2.20x / 2.43x |
| this port, adaptive: chat / code / math | same | 210.3 (2.81) / 295.6 (4.42) / 306.7 (4.63) | 1.47x / 2.08x / 2.21x |

Two things follow. With the same 4-bit target on the same GPU the other
runtime's DSpark is worth 15% at its measured cap and loses 16% at cap 7,
the verification-row cost this record measures, while the port's is worth
1.5x on the chat prompt and 2.1x to 2.4x on the code and math prompts
(accepted lengths of 6.4 and 6.9 at K=7, the paper's regime); its 1.58x on
the 8-bit target rests on a target-only floor of 106 tokens per second,
where verification rows are cheap relative to the weight stream. And the
other runtime decodes the target alone at 182 tokens per second where
vLLM's engine with this plugin decodes it at 140: the per-step overhead of
the serving stack at one stream is a target-only matter outside DSpark and
worth its own measurement.

### What the asynchronous scheduler changes (M9b, below)

vLLM's asynchronous scheduler overlaps the scheduling of step `k+1` with
the execution of step `k`, and the Metal decode pipeline builds on it; a
target-only server gains 5% to 12% from it on this machine (M4d,
`k0-async`). With speculative decoding configured the Metal platform forces
synchronous scheduling because every Metal proposer hands its drafts to the
engine through `take_draft_token_ids`, which the asynchronous engine loop
never calls. Upstream's contract for drafters under asynchronous scheduling
(the GPU runner uses it for every method, DSpark included) is different:
the scheduler pads every running request with `num_speculative_tokens`
placeholder slots (`-1`) and books their KV blocks, the worker substitutes
the drafts it produced at the end of the previous step when it executes
the batch, pads the unused slots as invalid (skipped by verification and
reported as `num_invalid_spec_tokens`), and keeps its own optimistic
`num_computed_tokens` corrected by the previous step's accepted counts.
Bringing DSpark under that contract on Metal means: lifting the platform's
synchronous downgrade for `dspark`; letting the spec-decode controller
build verification segments from the runner's own retained drafts instead
of the scheduler's (placeholder) tokens, with trailing invalid slots
allowed; reconciling the runner's request bookkeeping with the scheduler's
optimistic counts; and the adaptive planner's per-request prefix becoming
the number of valid slots. The deferred-step seam recorded above then
engages on its own (bypass steps become pipeline steps), and both the
asynchronous scheduler's overlap and the pipeline's deferred sync apply to
every non-drafting step. The M9b section below records that work and every gate,
bench and soak repeated in that configuration.

## M9b: DSpark under asynchronous scheduling (M5 Max, `b9f50ef`)

The M9 record closed on a structural limit: the Metal platform forced
synchronous scheduling whenever speculative decoding was configured, so a
DSpark server ran 5% to 12% behind an asynchronous target-only server
before any drafter cost, and the deferred-step seam that lets the decode
pipeline run on non-drafting steps could not engage. The reason was the
draft handoff. Every Metal proposer handed its drafts to the engine through
`take_draft_token_ids`, which vLLM's asynchronous engine loop never calls:
that loop schedules step `k+1` while step `k` executes, so the scheduler
cannot wait for step `k`'s drafts, and upstream's answer (the GPU runner
uses it for every drafter, DSpark included) is a different contract. The
asynchronous scheduler books `num_speculative_tokens` placeholder slots
(`-1`) for every running request and allocates their KV blocks; the worker
keeps the drafts it produced at the end of the previous step and
substitutes them into those slots when it executes the batch; the slots it
does not fill are reported as invalid so the speculative-decode statistics
count real drafts; and the scheduler rolls `num_computed_tokens` back by
the rejected count exactly as it does under synchronous scheduling, so a
request that fills no slot simply decodes one token.

This milestone brings DSpark under that contract. The platform keeps
vLLM's scheduling decision for `dspark` (asynchronous by default, as for
target-only serving) and still downgrades the draft-model, MTP and n-gram
proposers, which keep the synchronous handoff. The spec-decode controller
gains `substitute_retained_drafts`, which resolves each placeholder list
into the request's retained drafts cut to the slot count, passes a
synchronous handoff through unchanged, and records the unused slots; its
`validate_supported` accepts the runner's own resolution, in which a request
may verify fewer rows than the slots it was scheduled with, while a
scheduler-reported invalid count still fails closed. The runner retains
each request's drafts after `propose` (`_retain_drafts`), resolves the
step's drafts once per scheduler output (`_resolve_spec_tokens`, shared by
the pipeline gate, the validation and the segment builder), spends a
request's retained drafts at the first step that schedules it again
(its anchor token changes there) and drops them on release, preemption and
resume. A retained draft is anchored at the token the request's previous
step sampled, and under asynchronous scheduling the runner's own request
state, not the scheduler's `new_token_ids`, is the source of that token, so
the anchor is exact. Deferred steps do not propose, so a step after a
pipelined step verifies nothing; a drafting step keeps the synchronous
sample the seam already required. The harness tools take
`--async-scheduling` so every server of a comparison runs the production
scheduler.

### Evidence

Every server of `results/m5max-m9b-async-01` ran with `--async-scheduling`.
The proposer's one-time log line "first deferred step ingested; the decode
pipeline is running on non-drafting steps" appears in every adaptive and
bypass server of the run, so the seam the M9 record left inert is engaged.

- Serving gates: fixed K=7 (`serving-fixed-async`, 19:41), adaptive
  (`serving-adaptive-async`, 19:45) and bypass with `--expect-no-drafts`
  (`serving-bypass-async`, 19:49) all pass, every scenario equal or a tie
  against the asynchronous target-only server, zero failures.
- Stochastic gate (`stochastic-fixed-async`, 4,000 samples per cell): five
  of six distribution cells pass and the greedy parity scenarios are equal or
  ties, but temperature 0.7 with top-p 0.9 at position 2 fails (chi-square
  67.6 on 29 buckets, p = 4e-5; 3,993 drafted / 2,867 accepted) where the
  synchronous run of the same check passed that cell an hour earlier
  (chi-square 28.8). The cell is the one the M6b and M9 records already
  identified as the target's own nucleus flip between its single-row and
  multi-row forward, and the asynchronous scheduler changes the batch
  shapes the requests meet; the same check re-run with a third,
  target-only engine at one sequence as a control
  (`results/m5max-m9c-smallm-01/window/stochastic-async-control`, every
  engine asynchronous) reproduces the speculative comparison histogram for
  histogram (seeded) and shows the target's own spread between its batch
  shapes at that temperature and top-p: target-only at one sequence against
  target-only at the default sequence count disagrees at every position of
  the 0.7/0.9 cell with chi-square 258, 211 and 174 (p from 1e-54 to 1e-22),
  and at temperature 1.0 position 1 with p = 0.015 where the speculative
  comparison read p = 0.0019. The speculative server's 67.6 at position 2
  sits well inside the target's own 174 there, so the cell measures the
  target's nucleus boundary moving with the batch shape, as the M6b control
  found under synchronous scheduling and as the same diagnostic repeated
  under synchronous scheduling shows again (`window/stochastic-sync-control`:
  the speculative comparison passes every cell, the target-only control
  disagrees at the 0.7/0.9 cell with chi-square 257, 217 and 179), not the
  verification. The gate is
  reported as it stands; the M4a contract's tie class covers the mechanism.
- Adaptive bench against the asynchronous target-only server, five paired
  repetitions, load average 2.1 throughout (`bench-adaptive-async`):

| Bucket | C | Target-only tok/s | Adaptive tok/s | Benefit (95% CI) | Synchronous run (final-01) |
| --- | --- | --- | --- | --- | --- |
| 128x128 | 1 | 159.5 | 208.9 | +31.4% [+29.5, +34.7] | +51.5% (141.2 -> 213.7) |
| 128x128 | 4 | 420.1 | 412.0 | -2.5% [-4.9, +3.0] | -7.8% |
| 128x128 | 8 | 474.3 | 463.3 | -3.6% [-4.2, -0.0] | -0.3% |
| 1024x128 | 1 | 99.9 | 98.9 | -7.3% [-10.3, +15.7] | +9.3% [-7.0, +20.6] |
| 1024x128 | 4 | 215.0 | 204.1 | -4.3% [-6.7, -3.0] | -6.4% |
| 1024x128 | 8 | 245.8 | 235.7 | -4.3% [-6.1, -3.7] | -7.6% |
| 128x512 | 1 | 152.0 | 175.0 | +14.9% [+13.4, +24.0] | contaminated |
| 128x512 | 4 | 376.1 | 372.0 | -1.8% [-2.3, -0.1] | contaminated |
| 128x512 | 8 | 448.8 | 460.4 | +2.7% [+2.1, +3.1] | contaminated |

  The asynchronous target-only server is 13% faster than the synchronous one
  at one client (159.5 against 141.2 tokens per second) and the DSpark
  server is where it was (208.9 against 213.7), because a drafting step
  still synchronizes: the pipeline covers the non-drafting steps only, and
  at one client the adaptive planner drafts on nearly every step. The bypass
  floor at four and eight clients moved from -7.8/-0.3 to -2.5/-3.6 points on
  the short prompts and from -6.4/-7.6 to -4.3/-4.3 on the long ones, and the
  long-output bucket at eight clients turned positive.
- Fixed bench (`bench-fixed-async`, K=2 and K=7 at one and eight clients
  against the asynchronous target-only server): K=2 128x128 C1 +31.6%
  (138.2 -> 182.7), C8 -20.1%; 1024x128 C1 +4.4%, C8 -16.9%; K=7 128x128 C1
  +19.5% (147.0 -> 176.8), C8 -17.3%; 1024x128 C1 -22.3%, C8 -27.8%. The
  phase ran at a load average of 3.4 to 4.4 and its candidates read 9-10%
  below the synchronous run's at one client (K=7 194.2, K=2 203.3 tokens per
  second) while the adaptive phase at load 2.1 read its candidate unchanged;
  a repeat on a quiet machine (`window/bench-fixed-async-2`): was itself contaminated (another session's
  `bun` scan at 98% CPU during the phase, load average 3.1 to 5.0; K=7 128x128
  C1 read 137.1 -> 130.1 with a 27 ms p95 gap), so the fixed-mode numbers
  under asynchronous scheduling remain the first phase's, with their load
  noted; K=2 128x128 C1 +31.5% (153.3 -> 202.0),
  C8 -12.9%; 1024x128 C1 +12.3%, C8 -16.3%; K=7 128x128 C1 -1.9% (145.8 ->
  142.6), C8 -20.2%; 1024x128 C1 -30.1% (109.6 -> 76.6), C8 -27.6% (no pair
  discarded, load average at most 3.8). The K=2 server matches its
  synchronous throughput (202.0 against 203.3) and the adaptive one nearly
  (208.9 against 213.7), but the fixed K=7 server is 27% below its
  synchronous self at one client (142.6 against 194.2) with the same 540
  accepted of 1,524 drafted, so each K=7 round costs about a third more
  under the asynchronous scheduler; two diagnostics settle what that is. The served step cost
  under the asynchronous scheduler is not higher (`diag-cost-async`, one
  request, 128-token context: 6.55 ms target-only, 11.25 ms at K=2, 16.76 ms
  at K=7, against the synchronous profile's 7.25, 11.0 and 19.4 ms), and a
  back-to-back bench of K=4 and K=7 at one client under both schedulers
  (`k47-sync`, `k47-async`) shows K=4 unchanged (234.9 sync, 232.0 async)
  while the K=7 server's five repetitions spread from 159 to 192 tokens per
  second under the asynchronous scheduler and from 182 to 196 under the
  synchronous one, with the same 540 of 1,524 accepted: a repetition-level
  variance of the widest fixed width, not a per-step cost of the contract.
  K=4 is the better fixed width at one client on this machine (+72% sync,
  +52% async against the respective target-only servers), and the adaptive
  server (208.9) sits between it and K=7.
- Soak (`soak-fixed-async`, fixed K=7, eight clients, 20 minutes, load
  average 4.3): PASS, 3,790 requests (3,510 completed, 280 cancelled at the
  deadline as the protocol allows), zero errors, 238.5 output tokens per
  second, end-to-end p50/p95/p99 1.72/8.17/10.07 s, first token p95 0.58 s,
  183,339 accepted draft tokens, resident set flat (4.39 GB peak, 4.28 GB at
  the end).

### What the asynchronous scheduler does not give DSpark yet

A drafting step ends with a host-side verification (`verify` walks the
accepted prefix per request and the stochastic path samples the residual
on the host) whose result decides the next step's input tokens and the
drafter's ingest spans, so the runner synchronizes on every drafting step
and the pipeline only covers the steps the proposer will not draft. That
is why the asynchronous target-only server gained 13% at one client and
the DSpark server did not: the gap is now the synchronization, not the
scheduler. Closing it means keeping verification and the draft handoff
GPU-resident (accepted counts and tokens as arrays, the arena's committed
lengths advanced on the device, the host reading the outcome one step
later the way the pipeline reads a deferred sample), which touches the
proposer's bookkeeping end to end; it is recorded as the next lever after
the small-M kernel (M9c) rather than done here.

## M9e: the load regime (M5 Max, `e0de683`)

The M9b benches left a floor of 2.5% to 4.3% at four to sixteen clients
under the asynchronous scheduler: the planner declined to draft on nearly
every step (a K=7 step costs 2.8x to 3.8x a decode at eight to sixteen
requests on this machine, cost-04), yet the proposer still captured the
target's features on every step and ingested them into every draft
context, so the server paid for drafting it never did. The proposer now
follows the load. When the planner would decline a batch of the step's
size on 32 consecutive steps it lapses: it releases every context and its
arena slot, answers the runner's capture query with no (the target forward
runs exactly as it does for target-only serving), and spends each step on
bookkeeping only; it primes new requests again after the planner would
draft on 4 consecutive steps. The verdict is the planner's own
(`AdaptivePlanner.decide` with every draftable decode request at its full
cap and prior expectations), so it follows the request count and context
rather than the contexts a lapse released. A request that ran through a
lapse keeps target-only generation for its lifetime, as one that arrived
while the slots were full always did; the bypass mode lapses from the
start and the fixed mode never lapses. `VLLM_METAL_DSPARK_LAPSE=0` keeps
every context current at all loads, and the counters record entries,
exits and lapsed steps.

### Evidence

Every server of `results/m5max-m9e-lapse-01` ran the adaptive mode with the
cost-04 table under the asynchronous scheduler.

- Serving gates with the lapse on: adaptive (`serving-adaptive`, 01:22) and
  bypass with `--expect-no-drafts` (`serving-bypass`, 01:28) pass, every
  scenario equal or a tie against target-only, zero failures; the bypass
  server captures no feature and keeps no context from its first step.
- Adaptive bench at four, eight and sixteen clients, the lapse on against
  off (`VLLM_METAL_DSPARK_LAPSE=0`), each five paired repetitions against
  the asynchronous target-only server, back to back (00:30 and 00:45, no
  pair discarded except two and four in two 128x512 cells):

| Bucket | C | Lapse on | Lapse off |
| --- | --- | --- | --- |
| 128x128 | 4 | +5.5% [-15.2, +9.8] | -5.5% [-34, +5] |
| 128x128 | 8 | -0.8% [-3.8, +7.8] | -4.2% [-7, +5] |
| 128x128 | 16 | -1.8% [-10.7, +10.4] | -7.7% [-24, +16] |
| 1024x128 | 4 | -3.4% [-17.0, +2.6] | -4.0% [-18, +2] |
| 1024x128 | 8 | -14.5% [-20.2, +14.0] | -2.4% [-13, +0] |
| 1024x128 | 16 | -12.4% [-15.8, +1.7] | -11.6% [-17, -1] |
| 128x512 | 4 | +1.2% [-13.4, +9.6] | -9.5% [-18, +14] |
| 128x512 | 8 | -1.8% [-13.5, +18.2] | +4.7% [-7, +7] |
| 128x512 | 16 | +0.7% [-8.9, +8.2] | -4.0% [-8, +4] |

  The lapse is better or equal in seven of nine cells and the floor on the
  short prompts moves from -4.2/-7.7 to -0.8/-1.8 at eight and sixteen
  clients. The 1024x128 cell at eight clients was the first exit rule's
  fault: the server log shows eleven lapse entries and ten exits in fifteen
  minutes at eight requests, because the planner's verdict for a batch of
  that size flips between arrivals near its threshold and every exit primed
  every new 1,024-token prompt through the drafter for nothing. The regime
  now leaves only on a real drop in load (fewer requests than the lapse began
  with) with the verdict to draft on eight consecutive steps; the bench repeated with that rule
  (`bench-adaptive-lapse2`, 01:19, one lapse entry at four requests and no
  exit for the run): 128x128 +0.9% / +0.2% / +1.1% at four, eight and
  sixteen clients, 128x512 +2.9% / +3.1% / -0.5%, 1024x128 -4.7% / -13.4%
  [-17, +10] / -12.3% [-19, +7]. Against the run without the lapse that is
  better or equal in eight of nine cells, with the bypass floor on the
  short and the long-output prompts now within 3 points of target-only in
  either direction. The long-prompt cell at eight clients keeps its median
  11 points below the no-lapse run in both lapse runs, from repetitions
  that alternate with the pair order (candidate first 212-219, reference
  first 177-193 tokens per second, first-token time 1.5 against 2.2-2.4 s)
  where the no-lapse run's did not; the lapse does no per-step work there,
  so the mechanism is the two servers' shared memory and cache state under
  eight concurrent 1,024-token prefills rather than the regime, and it is
  recorded as open with the stall below.
- Both the 1024x128 candidates at eight and sixteen clients show a p95
  inter-token gap near 125 ms with and without the lapse (the M9b run too),
  against about 20 ms for target-only: a stall of the asynchronous adaptive
  server on long prompts at high load that is not the lapse's and is
  recorded as open.
- Soak, adaptive mode, eight clients, ten minutes, the lapse on: PASS (`soak-adaptive-2`, `--expect-no-drafts`): 1,507 requests (1,394 completed, 113 cancelled at the deadline), zero errors, 184.7 output tokens per second, end-to-end p50/p95 1.69/11.30 s, first token p95 0.31 s, resident set flat (4.45 GB peak, 4.43 GB at the end), one lapse entry at eight clients and one exit as the load drained. (A first run of the same soak read the same numbers, 1,507 requests and zero errors, and failed only the tool's draft-work criteria, which are the fixed mode's; `--expect-no-drafts` is the tool's new switch for a load the planner declines.)
