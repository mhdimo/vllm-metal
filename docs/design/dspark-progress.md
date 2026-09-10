# DSpark implementation progress

The [specification](dspark.md) defines milestone gates; the
[audit and experiment handoff](dspark-validation.md) preserves baseline evidence.
All runtime support remains experimental until the corresponding model, memory
and serving qualification gates pass.

| Milestone | Status | Change and validation |
| --- | --- | --- |
| M0: Baseline and contract | [Merged: #1](https://github.com/mhdimo/vllm-metal/pull/1) | Startup guards, resolved draft identity/revision, exact source provenance and normal lint coverage. Executable F1/F3 regressions track the remaining M1/M2 failures. |
| M1: Target capture | [Merged: #2](https://github.com/mhdimo/vllm-metal/pull/2) | Native Qwen3 capture, selected logits and complete prefill feature spans. |
| M2: Context lifecycle | Implemented and locally validated | Exact per-request ingest, physical rollback, lifecycle invalidation and safe prefix-hit behavior. |
| M3-M8 | Planned | Loading/resource qualification, serving, stochastic verification, confidence scheduling and additional model pairs. |
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
F5 revision/support guards and F9 provenance/lint are addressed; full F5 loading
and precision qualification remains M3 work. No speedup is claimed.

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
of 24 tokens, and `VLLM_METAL_MEMORY_FRACTION=0.12`. The current loader's default
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
exceeds the target planner's 2.75 GB allowance because full DSpark loading and
memory accounting are still M3 work. The probes fit the available machine;
the current fraction must not be interpreted as a complete DSpark memory cap.

These short, instrumented runs are correctness probes, not controlled throughput
benchmarks. Several speculative runs were slower than baseline. No speedup or
production serving envelope is claimed. M3/M4 must qualify allocation/reuse,
batch admission and fixed-K performance before confidence scheduling is tuned.

```bash
python -m tools.dspark_lifecycle_check \
  --target /path/to/pinned/Qwen3-4B-4bit \
  --draft /path/to/pinned/dspark_qwen3_4b_block7 \
  --concurrency 4 --width 2 --prefix-cache \
  --output-dir /path/to/results
```

### Remaining experiments and handoff

M0-M2 complete the contract, capture and context foundations. M3 remains the
next gate: account for draft weights and load-time conversion before allocating
target KV, bound per-request context/workspace, qualify precision, and test
pressure/recovery and long-running memory reuse. M4 must measure fixed K=0/1/2/7
with warmup, repeated trials, context/output-length buckets, TTFT, inter-token
latency and throughput, including prefix-hit and mixed-prefill workloads. The
fixed cap of 32 draft requests still needs fair, measured admission in M4.

Repeat the supplied capture and lifecycle checkers on the M5 Max 48 GB / 2 TB
machine before extending its workload envelope; that machine was not accessed
for this milestone. Use the [existing experiment matrix](dspark-validation.md#required-experiments)
for larger standalone pairs and model-specific gates. The current results do
not qualify Qwen3-8B/14B, Gemma4, stochastic decoding, calibrated confidence or
integrated V4. V4 remains outside both available machines' memory scale.
