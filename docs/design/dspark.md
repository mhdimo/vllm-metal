# DSpark implementation specification and roadmap

The branch contains a useful standalone DSpark prototype, but it is not ready
for a production support claim. The immediate priority is preserving the target
model's behavior through mixed batches and request lifecycle changes. Confidence
scheduling and kernel optimization come after those contracts are established.

This document defines the implementation contract and release gates. The
[validation and experiment handoff](dspark-validation.md) records the audited
revision, observed results, reproductions, and experiments still required.
Requirements below are proposed engineering decisions unless identified as
existing behavior or attributed to an external source.

Completed changes and their validation are tracked in the
[implementation progress record](dspark-progress.md). The audit findings below
describe the original baseline, not the resolution status of later commits.

## Scope and completion criteria

There are two distinct checkpoint interfaces:

| Interface | Initial models | Completion boundary |
| --- | --- | --- |
| Standalone DeepSpec DSpark | Qwen3-4B, 8B, 14B; then Gemma4-12B | A production standalone release requires qualified model pairs, greedy and stochastic verification, calibrated confidence scheduling, bounded memory, continuous batching, lifecycle correctness, and demonstrated serving benefit. |
| Integrated DeepSeek-V4 DSpark | Flash and Pro | A separate workstream requires a qualified V4 target backend and its integrated DSpark layers. Passing standalone tests does not establish V4 support. |

An earlier, explicitly limited **greedy Qwen3 experimental release** is a useful
milestone. It must not be called complete DSpark support. Conversely, implementing
training infrastructure, every possible research head, or third-party checkpoint
formats is not required to serve the published standalone checkpoints. Unsupported
variants must be rejected explicitly. Completion claims must name the supported
models, precisions, request features, hardware, and context/concurrency envelope.

The available experiment machines are an M4 with 32 GB RAM and an M5 Max with
48 GB RAM and 2 TB storage. Start with Qwen3-4B on the M4, repeat it on the M5 Max,
then use the M5 Max for bounded 8B/14B and Gemma4-12B qualification. Each model
must fit weights, target KV, draft context and temporary tensors together.
The executable roadmap is capped at these machines. Cases exceeding their
capacity remain explicitly deferred in the handoff; they are not waived or
scheduled on assumed larger hardware. Storage capacity does not enlarge the
resident memory budget.

## Reference contract

DeepSeek's [DeepSpec repository][deepspec] is the primary reference for standalone
model computation and verification. Its Qwen3 and Gemma4 checkpoints use a
seven-position block and five draft layers. The released checkpoints use the
vanilla Markov head; the reference also implements gated and recurrent variants.
The release's training data uses non-thinking target outputs, so thinking-mode
acceptance and calibration must be measured separately. [Sources: checkpoint
table][deepspec], [head implementations][heads].

For a standalone block, the input is the current anchor token followed by six
mask tokens. All seven hidden positions can predict draft tokens; the anchor
position is not discarded. The target verifies the anchor followed by the
selected draft prefix, requiring one more target input than the draft count.
The draft transformer sees valid target context plus the entire bidirectional
draft block. This differs from causal target attention. [Reference proposal
construction][proposal] and [attention-mask definition][common].

For vanilla Markov drafting, the base logits at position k receive a correction
from the embedding of the preceding sampled token. Sampling is sequential across
positions even though the backbone is parallel. Confidence uses the block hidden
state and, when configured, that same preceding-token Markov embedding; it is
not the probability or maximum logit of the selected token. [Qwen3 model][qwen]
and [Markov heads][heads].

The [DSpark paper, Algorithm 1 and Appendix A][paper] specifies cumulative
survival a[r,j] = product(c[r,1:j]), verification work B = sum(1 + length[r]),
and expected emitted tokens tau = sum(1 + sum(a[r,1:length[r]])). It maximizes
tau * SPS(B), where SPS is steps per second. The synchronous search stops at
the first non-improvement. Retrospectively choosing a length using future draft
tokens can bias stochastic output. Its asynchronous production design separates
historical budget estimation from current prefix ranking. These results establish
algorithmic constraints, not expected Metal speedups.

The available implementations have different roles:

| Source | Reuse as a reference for | Do not assume |
| --- | --- | --- |
| [DeepSpec evaluator][proposal] and [verifier][verifier] | Block positions, exact draft distributions, acceptance/residual sampling, accepted context rows | A production Metal scheduler; its proposal builder is batch-size one and supports a confidence threshold. |
| [vLLM 0.28 DSpark speculator][vllm-speculator] and [adaptive verification][vllm-adaptive] | Sequential head, separate draft/verify cost curves, historical confidence buffers, packed metadata | CUDA streams, compiled Torch/Triton kernels, and GPU runner integration are portable to Metal. |
| [SGLang planner][sglang-planner] and [STS artifacts][sglang-sts] | Explicit planner state, calibration artifacts, historical budgets and cache/verification separation | Identical scheduler APIs or numerical behavior. Pin source before adapting it. |
| [DeepSeek-V4 inference model][v4-model] | V4 feature reduction, mHC, integrated `mtp.*` draft stages, sliding attention and shared head | Compatibility with the standalone `DSparkConfig` loader. |

DeepSpec pins Transformers 5.10.2 in its [requirements][deepspec-requirements].
Reference tests must distinguish that environment from the plugin's supported
runtime. Updating one dependency without checking both model families is not a
valid parity procedure.

## Current branch architecture and gaps

The audit baseline is `eea6f8194828dda88186277312527a36c131367a` on `Dspark`.
It contains the collaborator's proposer and hidden-state capture work on Metal
base `b332259644a7564bd2518a66bfc3b18110f15eb5`. It already includes lifecycle
release wiring; the older `Dspark-implement` branch is not the integration base.

| Existing owner | Current responsibility | Required change |
| --- | --- | --- |
| `platform.py`, upstream `SpeculativeConfig` | Configuration and execution-mode selection | Establish the supported DSpark option contract before model loading. Reject unsupported options instead of accepting inert settings. |
| `v1/model_lifecycle.py`, `v1/model_runner.py::install_drafter` | Target loading; draft factory | Resolve and validate the model pair and reserve draft resources before target cache allocation. |
| `v1/dspark/{config,loader,model}.py` | Checkpoint parsing, quantization, MLX draft backbone, context K/V, heads | Strict architecture/precision validation, revision-aware loading, provenance, reference parity; make the confidence path usable. |
| `v1/hidden_state_tap.py`, `v1/model_adapter.py` | Selected residual capture and target logits | Preserve the model's forward semantics and selective logits layout. Avoid a generic hand-written layer loop for unqualified architectures. |
| `v1/dspark_proposer.py` | Context ingestion, prompt replay, padded batched drafting, fixed cap | Exact position coverage, bounded storage/replay, per-row caps, fair admission, explicit proposal records. |
| `v1/spec_decode.py` | Eligibility, packed verify segments and greedy verification | Retain one target verification owner; add probability-aware verification here or a helper it owns. |
| `v1/cache_policy.py`, `v1/worker.py` | Physical/scheduler cache planning | Account for DSpark weights, context, workspace and lookahead; coordinate admission and eviction. |
| `v1/model_runner.py::_reconcile_request_lifecycle` | Finish, preemption and resume invalidation | Preserve existing `release_requests` calls; extend cleanup to future probability, calibration-history and workspace state. |

Source links for these observations are in the [audit findings](dspark-validation.md#branch-findings).
The following are release blockers, not cosmetic cleanup:

1. **Target logits layout:** capture ignores `logits_indices`, while the runner
   can retain compact logits boundaries. Mixed decode/prefill batches can sample
   the wrong prefill row. A direct adapter/runner-layout probe reproduces this.
2. **Context completeness:** a final prefill chunk can be stored as a complete
   prompt at position zero; gap handling appends available rows at an unrelated
   offset; rollback changes a counter without trimming physical K/V. The latter
   two failures are reproduced at the contract boundary, not claimed as observed
   production request sequences.
3. **Memory ownership:** DSpark has no draft cache/resource plan. Draft weights
   load after the target cache is allocated, and each batch copies padded full
   context buffers. A fixed 32-request cap does not bound memory in bytes.
4. **Incomplete configuration and inference contracts:** draft revision is not
   forwarded, family detection is permissive, non-vanilla heads are not selected,
   and an unused probability-sampling method imports an absent module. Confidence
   is loaded but not consumed by the proposer.
5. **Evidence and maintenance:** baseline tests mainly stub the drafter; the
   losslessness checker can pass with zero speculative work and omits prefix hits,
   cancellation, output stopping and mixed sampling. The baseline documentation's
   stale behavior and unqualified performance/quantization claims are corrected
   with this roadmap; measured qualification remains open. Vendored-code
   attribution needs reconciliation with its source license and revision.

## Required interfaces and invariants

### Configuration, loading and compatibility

Resolve one immutable model-pair manifest before allocating either cache:

- Target repository/local identity, resolved revision, tokenizer identity and
  chat-template hash, architecture, layer count, hidden width, vocabulary, RoPE
  settings, dtype and quantization recipe.
- Draft identity and resolved revision, complete tensor schema, architecture,
  target feature IDs in their trained order, block size, mask ID, head type/rank,
  confidence inputs, dtype and quantization recipe.
- Runtime/backend versions and supported request modes. Calibration and timing
  artifacts must identify this manifest and reject mismatches.

Use the requested revision for both config and weights, including local/offline
paths. Honor relevant download/cache options through the repository's existing
loading conventions. Validate shard indexes, tensor shapes, missing/duplicate
keys, finiteness, mask/vocabulary bounds, and unsupported quantization formats.
Do not silently fall back to a different revision or permissive weight loading.

Start with exact Qwen3 standalone architectures and vanilla heads. Add Gemma4
only after its own target and draft adapters pass parity. Reject integrated V4,
Qwen3.5, GIDD, gated/RNN heads and speculators packaging until each has an explicit
tested adapter. A similarly named model or matching vocabulary size is not proof
of a trained target/draft pairing.

Use `method="dspark"` as the canonical intent, while testing upstream's existing
`draft_model` auto-detection. vLLM 0.28 can select its GPU V2 runner for DSpark;
the Metal path must select/validate its own supported runner before startup.
Current experiments explicitly set `VLLM_USE_V2_MODEL_RUNNER=0` and synchronous
scheduling. Do not advertise upstream adaptive, draft-sampling, synthetic
acceptance or top-k draft flags unless the Metal path implements their semantics.
See [vLLM runner selection][vllm-config] and [speculative configuration][vllm-options].

Keep FP32/unquantized draft computation available as the numerical oracle. Expose
quantization as an explicit recipe rather than an unconditional speed claim.
Qualify BF16/FP16 and 8-/4-bit drafters separately. Preserve sensitive confidence
and normalization computations at validated precision. Quantization can change
acceptance and calibration even when correct verification preserves target
sampling. Sharing a target embedding/head is optional: it requires identical
weights and transforms, not just matching tensor dimensions.

### Target features and logits

DeepSpec feature ID i denotes the residual after decoder layer i; its helper
indexes Transformers hidden states at i+1. ID -1 denotes the embedding output.
IDs must be strictly increasing. The reference evaluator rejects the final
decoder layer because its returned hidden state is normalized differently from
training cache capture. [Feature extraction][common] and [evaluation guard][verifier].

The initial adapter must reject unsupported feature selections at startup. A
future embedding/final-layer feature mode needs a named, tested convention.
Never sort or deduplicate feature IDs during loading, because that changes the
fusion matrix's inputs. Preserve native embedding scaling, layer masks, RoPE,
normalization, adapter selection and head postprocessing.

`target_forward` must provide two independently indexed outputs:

- Fused features for **all packed input rows**, accompanied by request identity,
  absolute start position and row count.
- Logits for exactly `logits_indices`, when provided, or all rows when absent.
  The runner's logits boundaries must describe the returned tensor.

Capture must be available on every prefill chunk that contributes needed
features, including steps that do not sample. Ingest and release chunks promptly;
do not retain every layer's full prompt activations until generation ends. Tests
must cover no capture, capture alone, selective logits alone, and both together,
with multiple prefills and decode requests in the same packed batch.

M1 implements this contract for native MLX Qwen3 through local observers on a
shallow body copy. The original model and attention wrappers are shared without
replacing the live layer list. Existing runner segment/prefill DTOs carry the
absolute spans; the adapter need not infer request identity from tensor shapes.
See the [progress record](dspark-progress.md) for real checkpoint parity evidence.

### Request state and context K/V

For an active request with committed token list length n, the final token is the
pending anchor at position n-1. Immediately before drafting:

```text
context positions = [0, n-1)
context length at every draft layer = n-1
anchor position = n-1
draft backbone positions = [n-1, n-1+gamma)
target verify inputs = [anchor, draft_1, ..., draft_K]
```

The standalone full-attention checkpoint requires all earlier context; silently
turning it into a sliding cache changes its model. A bounded implementation must
limit admission/context length, reconstruct exact state, or stop drafting that
request. It cannot drop old positions and renumber the remainder.

If verification accepts a draft prefix of length a, it emits that prefix plus
one correction/bonus token, unless a stopping condition terminates the request.
For a continuing request, ingest exactly a+1 target feature rows: the old anchor
and the accepted draft inputs. The correction/bonus is the next pending anchor
and has no target hidden state yet. Rejected or unverified suffix rows must never
enter persistent draft context. The official [evaluator update][evaluator]
implements this row relationship.

Maintain one request state record with generation/epoch, physical context length,
covered absolute position range, pending feature chunks, and proposal provenance.
At each ingest, the incoming start position must equal the coverage end. On a
gap, duplicate span, rollback or changed identity, discard/rebuild or skip
speculation; do not repair just the length counter. If retaining a prefix, trim
all physical layer buffers and associated metadata together.

| Lifecycle transition | Required action |
| --- | --- |
| Admission and prefill chunks | Append exact feature spans, subject to the resource budget. Intermediate chunks do not need an LM-head projection just to collect features. |
| Prefix-cache hit | Reuse compatible draft features/KV if present. Target KV alone cannot reconstruct intermediate residuals. Otherwise perform bounded, accounted replay, or explicitly use target-only generation. |
| Verification | Commit only the accepted input span; keep draft probabilities attached to the exact scheduled proposal. |
| Scheduler clips or omits drafts | Use the actual scheduled prefix. Preserve/reconcile context for requests that advance without drafting. |
| Preemption or resume | Release DSpark state through the existing lifecycle owner; recompute against the resumed epoch before reuse. |
| EOS, stop string, max tokens, cancellation, error | Honor normal target stopping and streaming semantics; release all proposal/context/history allocations. No extra visible bonus token after termination. |
| Finished ID reused in the same step | Invalidate the old generation before admitting the new one; test the runner's existing release path end to end. |

Prefix replay must run in a context isolated from scheduler-owned target KV
writes and any active LoRA/multimodal state. Bound its tokens and workspace;
measure its TTFT and admission impact. A hidden second full padded prompt forward
for every new request is not an acceptable steady-state solution.

### Draft execution and memory planning

Keep the seven-position backbone intact when reducing K: later masked positions
participate in bidirectional attention. Selecting fewer head/output positions is
different from changing the trained backbone's block length. Draft K may range
from zero through gamma, further limited by the scheduler, remaining output
budget, context limit, and available memory. Reserve a correction/bonus slot for
continuing verification. Use each row's own cap; do not read only `plans[0].cap`.

The current padded mask correctly allows each row's valid context and every
block position when physical lengths and offsets agree. Preserve that behavior.
Use per-request absolute RoPE offsets, mask all padding, and retain GQA/MQA K/V
head counts without tiling them to query-head counts. Prove batched execution
against independent rows, including unequal context lengths and request order.

Budget the following before deciding target cache capacity:

```text
peak = target weights + draft weights + target KV + persistent draft context
     + feature staging + draft/verify workspace + logits/probabilities
     + allocator/compilation reserve
```

For full-attention standalone context, per-token K/V bytes are
sum_over_layers(Hkv * (Dk + Dv) * bytes_per_element). For the official Qwen3
drafts with five layers, eight KV heads, 128-dimensional K/V and two-byte values,
this is 20 KiB/token: an 8,192-token context is 160 MiB/request. Thirty-two such
requests require 5 GiB just for persistent context. The prototype can create
another similarly sized padded batch copy, before other temporaries. These are
analytical storage estimates, not measured peak-memory results.

Load/materialize the drafter or conservatively reserve its measured peak before
target KV allocation. Prefer reusing the existing cache planner, but do not
register synthetic autoregressive draft KV groups with the wrong lifecycle or
byte count. Initially a bounded proposer arena can be valid if the planner
subtracts its entire capacity and enforces admission. Longer term, paged draft
context should share exact prefix blocks with reference counts and copy-on-write.
This requires feature-compatible prefix keys and reclamation tied to request
epochs, rather than relying solely on target token hashes.

Optimize in measured order: remove duplicate prefill work; avoid full-context
copies and concatenation each step; reuse allocated buffers; group ragged work
by length or implement paged cross-attention; then optimize sequential head and
host synchronization. CPU scheduling is acceptable initially for small batches
if its full cost is measured. Add Metal kernels only after profiling identifies
the bottleneck and a reference implementation defines the result.

### Verification and request features

The initial greedy path retains the shared verifier and its eligibility rules.
Check every request setting affecting target logits, including min-token EOS
masking, penalties, allowed/bad tokens, custom processors, structured output and
logprobs. Unsupported requests must use the ordinary target sampler without
speculative tokens, with an observable reason; an explicitly unsupported global
configuration should fail before model work.

For stochastic support, a proposal record must retain its request epoch, anchor,
absolute positions, selected token IDs, and exact normalized draft distributions
q from the sampling operation. Transformations and precision used to construct q
are part of that record. Store the needed distributions or enough immutable data
to reconstruct them exactly; token IDs or selected-token probabilities alone
are insufficient for residual sampling.

At each position use target distribution p after the request's target sampling
transforms. Accept x sampled from q with probability min(1, p(x)/q(x)). At the
first rejection sample from normalized max(p-q, 0); after full acceptance sample
the target bonus distribution. Handle finite precision, zero support and
underflow explicitly. Validate tiny-vocabulary cases by enumerating outcomes,
including q with truncated support, rather than substituting a heuristic
residual. [DeepSpec verifier][verifier] and [sampling helpers][sampling].

Target penalties and grammar state must evolve along the verified prefix, not
one shared pre-step state. Add such features only with distribution tests, or
route them to target-only sampling. Use per-request random streams with separate
proposal/acceptance/target draws; cancellation or reordering another request must
not consume this request's stream. Distributional equivalence is the contract;
identical output at the same seed is not evidence of stochastic equivalence.

Keep only one authoritative commit/rollback/stop decision. Target KV slot
validity, prefix publication, returned token counts and DSpark state must all use
it. Verify mixed greedy and stochastic requests, K=0, partial scheduler admission,
EOS within an accepted prefix, and output-budget truncation before enabling the
mode in serving.

### Confidence calibration and adaptive verification

Implement these as separate components with small pure reference functions:

1. **Recorder:** collect raw confidence logits, observed accepted-prefix labels,
   valid lengths, and model/sampling identity. Record untruncated proposals on a
   calibration split separate from training and final evaluation. Positions
   beyond the scheduled verification length are censored, not negative labels.
   Within a fully scheduled block, the first rejection makes all longer prefix
   survival labels zero; it does not establish their conditional acceptance
   labels. Exclude/censor positions beyond EOS or output limits.
2. **STS fit/apply:** fit positive finite per-position temperatures sequentially
   on cumulative survival predictions, holding earlier fitted temperatures fixed.
   Apply `c[r,k] = sigmoid(z[r,k] / T[k])` to raw confidence logits z before
   computing prefix survival; calibration temperature is distinct from token
   sampling temperature. [Inference application][sglang-model].
   Store the grid, bin definition, objective, sample counts, dataset revision and
   split, and per-position ECE/Brier/reliability before and after fitting. Reject
   wrong gamma, non-finite values and manifest mismatch. Validate on a separate
   holdout; calibration from an unquantized or different sampling mode is not
   automatically reusable. [Recording semantics][confidence], [STS artifact reference][sglang-sts].
3. **Cost profiler:** measure completed draft and verify steps on the actual
   backend, accounting for synchronization, sampling and data movement. Model
   context length, active request count, padded shapes and mixed prefill work
   when material. A function of total verification tokens alone is a hypothesis
   to test on Metal. Store measured samples, uncertainty and validity bounds.
4. **Planner:** allocate only feasible prefix extensions with deterministic ties
   (earlier position before its descendants). Guarantee one target input for
   each active decode request. K=0 is a normal decision, not a failed request.

The performance objective for the Metal implementation is expected useful
emitted tokens divided by total step time. For a fixed set of drafted requests,
include draft, context movement, verification and host costs in the denominator.
Choosing whether to draft at all is a separate decision: K=0 after executing the
backbone does not recover its cost. Admission/bypass decisions use information
available before the affected candidate is sampled.

First implement a causally ordered synchronous prefix planner with a pure CPU
oracle and strict early stopping. Expose only the next eligible extension per
request and freeze each admission decision before revealing its descendant.
Hard caps may truncate this prefix, not fill unused capacity from future scores.
Never claim global optimality on a non-unimodal measured cost curve.

An optional later planner may estimate a fixed current budget from versioned
historical confidences and timing information, then rank current prefixes within
that budget. Reset history on request reuse, preemption, mode changes and idle
steps. It must pass causality tests before relaxing early stopping. The existing
Metal scheduler is synchronous; upstream CUDA/ZOS mechanisms cannot be enabled
by simply removing its guard. [vLLM historical budget implementation][vllm-adaptive].

Scheduling must operate at a point where the actual verification batch and its
capacity are known. Initially return a valid prefix through `DraftTokenIds` and
let the scheduler clip it; test upstream placeholder behavior. If global budget
decisions are moved into verification, introduce an explicit metadata handshake:
effective lengths, slot mappings, position IDs, logits boundaries and accepted
token accounting must change together. Do not shorten just a tensor while the
host scheduler still counts the original width.

Missing/stale cost or calibration artifacts must disable adaptive mode with a
clear reason, or select an explicitly configured fixed mode. Do not invent a
flat SPS curve or silently treat uncalibrated confidence as production-ready.
Add counters for bypass reasons, proposed/scheduled/verified/accepted tokens,
correction/bonus tokens, per-position opportunities and acceptances, planner
lengths, replayed tokens, context bytes, and draft/verify/host times. Keep request
IDs and per-step informational logs out of the default hot path.

## Implementation sequence

### Alignment with earlier Metal integrations

Gemma4 MTP landed as separate [target capture](https://github.com/vllm-project/vllm-metal/pull/369),
[assistant loading](https://github.com/vllm-project/vllm-metal/pull/374),
[KV sharing](https://github.com/vllm-project/vllm-metal/pull/384),
[scheduler handoff](https://github.com/vllm-project/vllm-metal/pull/387), and
[documentation/benchmark](https://github.com/vllm-project/vllm-metal/pull/410)
changes. DSpark follows that staged integration, retaining the existing proposer,
adapter, lifecycle and verification owners.

The subsequent [proposer-policy cleanup](https://github.com/vllm-project/vllm-metal/pull/483)
removed an unnecessary generic registry. The [resolved-config cleanup](https://github.com/vllm-project/vllm-metal/pull/597)
made upstream's draft `ModelConfig` authoritative. DSpark should use these
contracts directly, without speculative abstractions or fallback config sources.

[Draft cache integration](https://github.com/vllm-project/vllm-metal/pull/630)
replaced a rejected private prefix cache with scheduler-owned cache groups so
cache salts, prefix-read policy and admission stayed consistent. DSpark must not
introduce cross-request prefix reuse outside that ownership. A temporary bounded
context allocation must be accounted for by the cache planner and remain
request-local until a proper shared-cache contract exists. [Lifecycle release](https://github.com/vllm-project/vllm-metal/pull/551)
and the [measured proposer overhead investigation](https://github.com/vllm-project/vllm-metal/issues/482)
also motivate exact invalidation, incremental ingest and before/after measurements.

Each milestone is committed on a focused branch, reviewed as a PR into `Dspark`,
validated and merged before the next milestone is based on it. The fork's current
CI filters target `main`; PRs into `Dspark` therefore need recorded local checks.
This branch workflow does not establish upstream approval or full-feature release
readiness.

Each row is an independently reviewable change with a measurable exit gate.
Dependencies are semantic; estimated calendar dates are intentionally omitted.

| Milestone | Depends on | Deliverable and principal owners | Exit gate |
| --- | --- | --- | --- |
| M0: Baseline and contract | None | Preserve source manifest; convert audit witnesses into regressions; explicit unsupported-mode guards in platform/factory; provenance and docs cleanup | Canonical and alias config reach the Metal proposer; inert/unsupported options fail; baseline experiment is reproducible; branch findings have tracked tests. |
| M1: Target capture | M0 | Adapter-native feature capture; selective-logit parity; prefill chunk span metadata in adapter/runner | Every logits/hidden layout combination passes; real Qwen3 target output with capture equals no capture at the same execution shape; mixed-prefill reproduction fixed. |
| M2: Context lifecycle | M1 | Exact ingest/rollback/replay state in proposer; runner lifecycle regressions | Every accepted length 0..K, holes, rollback, mixed chunks, prefix hit/miss, finish/reuse, cancel and preemption satisfy the physical cache invariant. |
| M3: Loading and memory | M0, M2 | Revision-aware loader; qualified quantization; complete resource plan; bounded context/workspace | Deterministic loading; no unbudgeted allocations; memory-pressure admission and recovery tests; no retained state after lifecycle soak. |
| M4: Fixed greedy serving | M1-M3 | Ragged/per-row caps, fair draft admission, real checkpoint e2e harness | Qwen3-4B qualification in a named envelope; actual speculative tokens verified; exact greedy comparisons and streaming/stop tests; fixed-K performance baseline. |
| M5: Stochastic verification | M2-M4 | Proposal distribution ownership and rejection sampler | Analytic finite-vocabulary tests plus statistical/e2e tests pass; supported sampling features enumerated; unimplemented transforms fall back correctly. |
| M6: Confidence and planner | M3-M5 | Calibration recorder/fitter, measured cost artifacts, causal planner, metrics | Held-out calibration and scheduler causality gates; variable K including zero; benefit over target-only and tuned fixed K across the declared load range. |
| M7: Initial production qualification | M4-M6 | Profile-driven memory/kernel optimizations, operational docs, CI/nightly coverage | Qwen3-4B serving/performance/soak matrix passes; all release-critical evidence attached; no unresolved correctness findings in its declared envelope. |
| M8: Additional standalone pairs | M1-M3; M4-M7 gates reused | Qualify Qwen3-8B/14B; add family-specific Gemma4 target adapter and draft parity, text-only first | E10/E11 pass for every claimed pair; Gemma4 coverage includes sliding/full target attention and KV-sharing boundaries; same correctness, memory, sampling and performance gates. |
| V4: Integrated DSpark | Separate target/backend prerequisite | Integrated checkpoint conversion/loading, target-feature reduction and draft stages | Target-only V4 qualified first, then the equivalent full DSpark gates on suitable hardware; remains open independently of M7/M8. |

M4 is not permission to skip M5-M8 while calling standalone DSpark complete.
M5-M7 can proceed on 4B while larger-model work is pending; those experiments
remain required for the broader standalone release rather than blocking local
algorithm development. Family-specific M8 implementation can proceed in parallel
once its prerequisites are met, but qualification repeats the complete gates.
Prototype defects should be fixed in focused commits before performance changes,
so the numerical and lifecycle effects remain reviewable. A feature is not
qualified merely because a fallback produces correct text.

## Integrated DeepSeek-V4 workstream

The official Flash/Pro configs use `dspark_*` fields and integrated draft weights,
not standalone DeepSpec packaging. Flash declares block size 5 and Markov rank
256; Pro declares block size 5 and rank 512. Do not hard-code seven positions or
rank 256 across interfaces. [Flash config][v4-flash], [Pro config][v4-pro].

The official inference code stores DSpark stages under `mtp.*`, reduces selected
target mHC residuals by a mean across the hyperconnection dimension, projects and
normalizes those features, shares target embedding/head, and uses its own draft
sliding-attention cache. Its sample generation script runs ordinary generation;
the existence of `forward_spec` is not a production verifier or scheduler.
[Model][v4-model] and [generation loop][v4-generate].

Proceed only after target-only support establishes checkpoint quantization,
mHC/MoE, sparse/compressed attention, position semantics, and cache behavior on
Metal. Derive per-layer context retention from the V4 checkpoint; do not copy
the standalone full-context policy. Add verification-window and rollback support
for compressed/indexed target caches, then integrate draft layers and calibrated
scheduling. Reuse shared contracts, not Qwen3 tensor layouts.

The released V4 checkpoints cannot be qualified on either available Mac
(32/48 GB RAM). Record required weight conversion format, RAM and device topology
in the deferred handoff; do not schedule this work within the current experiments.
Do not substitute a small Qwen3 benchmark or extrapolated CUDA throughput for
V4 evidence. A distributed Metal target deployment also needs the repository's
speculation/parallelism restrictions addressed and tested first.

## Release decision

Use the [experiment matrix and evidence requirements](dspark-validation.md#required-experiments)
as the merge checklist. A release can proceed only for an explicitly named
support envelope with no open correctness findings, passing resource/lifecycle
tests, reproducible model-pair parity and useful serving measurements. Performance
claims need before/after artifacts under the project's
[contribution requirements](../CONTRIBUTING.md).

The next implementation should start with M0/M1 and the reproduced mixed-batch
logits defect. Passing small draft-model parity checks supports continuing this
work; it does not justify skipping the serving integration fixes.

## Primary sources

Code references are pinned to the versions inspected. Reference links identify
the original publisher; the independent MLX port is provenance evidence, not the
official DeepSeek specification.

- DeepSeek-AI, [DeepSpec][deepspec], commit `005e03b81cec38b7da6399833d609ee89a2587f2`.
- DeepSeek-AI, [DSpark paper][paper], arXiv:2607.05147v1, July 2026.
- vLLM project, [v0.28.0][vllm-options], commit `2cf0a6915ce544dc493a0990f2ea38d81601128a`.
- SGLang project, [DSpark planner][sglang-planner], commit `2092f6df05960c52d9df1994b0af812c2fc6544a`.
- DeepSeek-AI, standalone and integrated checkpoint configs, pinned individually in the [validation manifest](dspark-validation.md#checkpoint-manifest).

[deepspec]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/README.md
[deepspec-requirements]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/requirements.txt
[paper]: https://arxiv.org/html/2607.05147v1
[qwen]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/qwen3/modeling.py
[heads]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/markov_head.py
[common]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/common.py
[proposal]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/dspark/draft_ops.py
[evaluator]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/dspark/evaluator.py
[verifier]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/base_evaluator.py
[sampling]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/utils/sampling.py
[confidence]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/dspark/confidence_head.py
[vllm-options]: https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/config/speculative.py
[vllm-config]: https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/config/vllm.py
[vllm-speculator]: https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py
[vllm-adaptive]: https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py
[sglang-planner]: https://github.com/sgl-project/sglang/blob/2092f6df05960c52d9df1994b0af812c2fc6544a/python/sglang/srt/speculative/dspark_components/dspark_planner.py
[sglang-sts]: https://github.com/sgl-project/sglang/blob/2092f6df05960c52d9df1994b0af812c2fc6544a/python/sglang/srt/speculative/dspark_components/dspark_sts.py
[sglang-model]: https://github.com/sgl-project/sglang/blob/2092f6df05960c52d9df1994b0af812c2fc6544a/python/sglang/srt/models/dspark.py
[v4-flash]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/62af8fffb2f7030cac4de2f0169f5b8d1101b646/config.json
[v4-pro]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark/blob/7c09739fd136abfb70a49ec334157f65f45b52cd/config.json
[v4-model]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/62af8fffb2f7030cac4de2f0169f5b8d1101b646/inference/model.py
[v4-generate]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/62af8fffb2f7030cac4de2f0169f5b8d1101b646/inference/generate.py
