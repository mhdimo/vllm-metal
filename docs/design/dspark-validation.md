# DSpark validation and experiment handoff

This is the evidence record and execution checklist for the
[DSpark specification](dspark.md). The implementation baseline is
[`eea6f8194828dda88186277312527a36c131367a`][baseline]. Findings and results below
refer to that revision; a later documentation commit does not imply the runtime
findings are fixed.

See the [implementation progress record](dspark-progress.md) for resolutions
after this audit and the validation attached to each milestone.
For current status and migration to the M5 Max, use the
[development handoff](dspark-handoff.md); this document retains the original
audit findings and full experiment matrix.

## Checkpoint manifest

These are inspected official checkpoint configurations, not a list of models
already qualified on Metal. Every inference experiment must also pin its target,
tokenizer, quantization/conversion recipe, and runtime.

| Draft checkpoint | Revision | Trained target | Feature layers | Block / Markov rank |
| --- | --- | --- | --- | --- |
| [Qwen3-4B][q4] | `3457dff1417cb84927f6098a5fcb7cee85c934b7` | `Qwen/Qwen3-4B` | 1, 9, 17, 25, 33 | 7 / 256 |
| [Qwen3-8B][q8] | `03326e5043815da1f81b109078b2889737c26017` | `Qwen/Qwen3-8B` | 1, 9, 17, 25, 33 | 7 / 256 |
| [Qwen3-14B][q14] | `83207b416acf99f41c2184648923632fccea6dd0` | `Qwen/Qwen3-14B` | 1, 10, 19, 28, 37 | 7 / 256 |
| [Gemma4-12B][g12] | `2fa72e765eec2965fc4d86a8663ce6769eba6218` | `google/gemma-4-12B-it` | 5, 17, 29, 41, 46 | 7 / 256 |
| [V4-Flash integrated][vf] | `62af8fffb2f7030cac4de2f0169f5b8d1101b646` | Same V4 checkpoint | 40, 41, 42 | 5 / 256 |
| [V4-Pro integrated][vp] | `7c09739fd136abfb70a49ec334157f65f45b52cd` | Same V4 checkpoint | 58, 59, 60 | 5 / 512 |

The standalone target associations come from the [official release table][release].
Third-party quantized targets require separate qualification against these trained
model identities. The V4 model card contains a seven-token launch example while
its config declares `dspark_block_size=5`; the minimal model uses that config
value. Resolve checkpoint/engine length semantics in the V4 implementation
milestone rather than copying the launch example into Metal documentation.

## Branch findings

Priorities describe the order of work before a production release. “Reproduced”
means a concrete local witness; “inspection” means the relevant path is present
in code but its full serving consequence has not been measured.

| ID | Priority / evidence | Finding | Required resolution |
| --- | --- | --- | --- |
| F1 | Correctness blocker; reproduced | [Capture][adapter] returns all logits even with `logits_indices`. [Runner][runner] retains compact logits boundaries. For packed boundaries `[0,3,7,12]`, capture returns 12 rows although selected indices are `[0,1,2,6,11]`; prefill sampling reads source rows 3/4 instead of 6/11. | Preserve selected logits while retaining full hidden rows; integration regression with mixed decode and multiple prefills. |
| F2 | Context blocker; inspection | [Proposer][proposer] skips pure-prefill capture; when capture occurs in a mixed batch it stashes only the final chunk and ingests it at offset zero. Prefix-hit/final-chunk length is not validated against full committed context. | Explicit absolute spans, capture/ingest all required chunks, coverage checks and bounded prefix replay/fallback. |
| F3 | Context blocker; reproduced boundary cases | `_ensure_context` caps a missing span to currently available rows and proceeds with incomplete coverage. Its rollback branch changes `_n_cached` without trimming layer K/V. | Fail closed or rebuild/trim all physical state; tests for mismatch, zero rows and missing hidden states. |
| F4 | Resource blocker; inspection | [Cache setup][cache] allocates target KV before `install_drafter`. DSpark has no `_draft_dims` resource plan; [batch drafting][proposer] allocates padded context copies, and [CtxCache][model] repeatedly concatenates. | Budget loaded weights, persistent context and peak workspace; byte-based admission; eliminate full-context copies where profiling warrants. |
| F5 | Reproducibility/support blocker; inspection | [Loader][loader] downloads without draft revision and quantizes by default. [Config][config] accepts families/variants beyond the qualified target adapter. | Immutable pair manifest; explicit loader options; strict supported architecture/head/precision validation. |
| F6 | Feature-completion blocker; inspection | [Model][model] always constructs vanilla Markov for positive rank; confidence is unused in [proposer][proposer]. `sample_block_probs` imports absent `dspark/sampling.py`. | Keep unsupported modes unreachable with clear guards; implement exact distribution ownership and the calibration/planner milestones before advertising these features. |
| F7 | Coverage blocker; inspection | [Tests][tests] largely replace the draft model/backbone. The test helper ignores its `finished_req_ids` argument. [Lossless checker][checker] can report success without requiring positive proposed/verified counts and forces prefix caching off and EOS ignored. | Add reference fixtures, real-checkpoint tests and runner-level lifecycle coverage; make zero-work speculation an explicit failure of an acceleration test. |
| F8 | Release/documentation blocker; inspection | [User guide][guide] describes per-request drafting despite batched code, mentions vLLM 0.25.1 despite the 0.28 pin, and claims K=2 is an Apple Silicon optimum and arbitrary target quantizations work without linked evidence. | Replace universal claims with qualified results and current configuration; document exact request eligibility/fallback behavior. |
| F9 | Provenance/maintenance blocker; inspection | [Loader][loader] identifies a vendored MIT source, but the baseline has no corresponding source copyright/permission notice and source revision alongside these files. The package is excluded from Ruff. | Record the adapted source revision and preserve required source notices; remove dead standalone-library messages and bring maintained code under normal lint/type checks. |

The existing runner already invalidates proposer state on finish, preemption and
resume via `release_requests`. Do not replace that ownership with pruning based
only on absence from `request_states`: IDs can be reused within a step. New
probability/workspace/history state must join the same cleanup path.

The initial documentation change corrected stale F8 descriptions and added part
of the F7 reference evidence. M0-M2 subsequently resolved F1-F3 and the guarded
support/provenance portions of F5/F8/F9. See the [progress record](dspark-progress.md)
for passing regressions and real 4B results. M3 subsequently addressed F4 resource
planning and F5 loading/precision for its named 4B recipe and memory envelope;
its [qualification record](dspark-m3-validation.md) preserves both passing results
and extended parity failures. F6 completion, broader model-pair qualification and
the F7/M4-M7 serving matrix remain open. The baseline witnesses below intentionally
preserve the original failure evidence.

The MLX port is attributed to [ARahim3/mlx-dspark][port]; its inspected
[MIT notice][port-license] names copyright holder `erahim3`. That independent
implementation is distinct from DeepSeek's official model/evaluation code.
M0 subsequently established that baseline `config.py` and `model.py` match
upstream commit `9e39ea2fdc6d99d855af2cb7ef9933391c4391db` byte-for-byte. Their
source notice and license are now included alongside the maintained files.

### Minimal F1 reproduction

Run from a checkout of the audited branch with the project's runtime installed.
This exercises the real capture path and real logits-layout computation with a
deterministic toy backbone; it does not require model weights.

```python
from types import SimpleNamespace
import mlx.core as mx
from vllm_metal.v1.model_adapter import DefaultModelAdapter
from vllm_metal.v1.model_runner import MetalModelRunner

body = SimpleNamespace(
    embed_tokens=lambda ids: ids[..., None],
    layers=[lambda h, mask, cache: h],
    norm=lambda h: h,
)
adapter = DefaultModelAdapter()
adapter._target_backbone = lambda model: body
adapter._compute_target_logits = lambda model, hidden: hidden
layout = MetalModelRunner._paged_logits_layout(
    SimpleNamespace(_selective_logits_supported=True),
    [0, 3, 7, 12],
    num_decode_segments=1,
)
result = adapter.target_forward(
    object(),
    mx.arange(12, dtype=mx.float32)[None],
    cache=[None],
    capture_layer_ids=[0],
    logits_indices=layout.indices,
)
print(layout.indices.tolist(), result.logits.shape)
print(result.logits[0, 3:, 0].tolist())
# Audited: [0, 1, 2, 6, 11], (1, 12, 1); trailing values start at 3.
# Required: logits shape (1, 5, 1), final logits values [6, 11],
# while result.hidden_states still contains all 12 packed rows.
```

### Minimal F3 reproductions

This uses the branch's existing test fixtures to exercise the real context
reconciliation method. The first case checks physical rollback; the second
checks a missing feature span with only one new row available.

```python
import mlx.core as mx
from tests.test_dspark_proposer import _context, _proposer, _segment, _state
from vllm_metal.v1.dspark.model import CtxCache

for cached, committed in ((5, 4), (2, 8)):
    proposer = _proposer()
    cache = CtxCache()
    cache.append(mx.zeros((1, 1, cached, 1)), mx.zeros((1, 1, cached, 1)))
    proposer._ctx_caches["r"] = [cache]
    proposer._n_cached["r"] = cached
    state = _state(list(range(committed)))
    ctx = _context(
        decode_reqs=[("r", state)],
        decode_segments=[_segment("r", num_query_tokens=1)],
        target_hidden_states=mx.zeros((1, 4)),
    )
    plan = proposer._ensure_context(ctx, state, ctx.decode_segments[0], 2)
    print(plan.n_cached, cache.length, committed - 1)
# Audited: counter / physical / required = 3 / 5 / 3, then 3 / 2 / 7.
# The fixture stubs update_context; the second witness is the incomplete
# logical coverage and returned plan, not the stub's physical length.
```

These are defensive-contract witnesses; they do not establish how often real
traffic enters those states. Production regressions must also exercise real
context updates and request transitions.

## Observed validation

Audit host: Apple M4, 32 GiB unified memory, macOS 15.6, Python 3.12.13,
MLX 0.32.1, mlx-lm `9e6acca691e64d6d8bb808c328fcdea459099cca`, vLLM 0.28.0+cpu,
mlx-vlm 0.6.8. Tests ran against the checkout above. Reference inference used
DeepSpec `005e03b81cec38b7da6399833d609ee89a2587f2` on Torch CPU, with MLX
computation on the Mac. The reproducible checker uses Transformers 5.10.2 and
tokenizers 0.22.2 in an isolated reference dependency directory. Torch was 2.13.0,
whereas DeepSpec's requirements pin 2.9.1; this is an API-level numerical probe,
not a reproduction of its complete training/evaluation environment. These
results do not certify production performance.

| Experiment | Result | What it establishes |
| --- | --- | --- |
| Existing proposer, hidden-tap and runner-generate tests | 140 passed | Current targeted unit coverage passes; it does not catch F1. |
| F1 packed-row witness | Reproduced | A deterministic integration-contract defect exists. |
| F3 physical rollback and missing-span witnesses | Reproduced | Counters/coverage can disagree with physical context. |
| Tiny FP32 Qwen3, copied official state dict, context lengths 1/5/9, incremental 5+3 | Hidden max absolute difference <= 5.97e-7; logits <= 8.95e-8; confidence <= 5.97e-8; identical greedy draft tokens | The tested small unquantized model computations agree. This is not real-checkpoint, quantized or target-capture parity. |
| Tiny Qwen3 ragged batch, context lengths 2/6 | Same draft token rows as separate execution | Basic padded batch/offset behavior agrees for the tested case. |
| Tiny FP32 Gemma4, same copied-weight and incremental protocol | Hidden max absolute difference <= 4.42e-6; logits <= 3.58e-7; confidence <= 1.72e-7; identical greedy draft tokens | The tested draft model agrees with the pinned reference. Target Gemma4 capture and real checkpoint serving remain unqualified. |
| Tiny Gemma4 ragged batch, context lengths 2/6 | Same draft token rows as separate execution | Basic padded batch behavior for this family. |
| Real Qwen3-4B target/draft smoke | Not run: checkpoint transfer remained incomplete | No real-checkpoint correctness, acceptance or timing claim. This is a transfer prerequisite, not a 32 GB hardware limitation. |

Re-run the baseline tests with:

```bash
VLLM_METAL_BUILD_FROM_SOURCE=1 python -m pytest \
  tests/test_dspark_proposer.py tests/test_hidden_state_tap.py \
  tests/test_v1_model_runner_generate.py -q
```

### Reproduce the reference-model checks

The [manual checker](https://github.com/mhdimo/vllm-metal/blob/Dspark/tools/dspark_reference_check.py) imports the official
model definitions, verifies the DeepSpec checkout revision, copies their state
dicts into the MLX model, and checks hidden states, logits, confidence, incremental
context and batched token proposals. It uses tiny random weights and FP32 with
`atol=rtol=1e-5`; the raw configuration, seeds, versions and observed errors are
in [the result artifact](dspark-reference-results.json).

Run from the Metal checkout with its development runtime active. The reference
packages below are installed in a separate directory, preserving server runtime
dependencies. If the reference checkout already exists, verify its revision
instead of cloning over it.

```bash
mkdir -p .validation-dspark
git clone https://github.com/deepseek-ai/DeepSpec.git .validation-dspark/DeepSpec
git -C .validation-dspark/DeepSpec checkout --detach 005e03b81cec38b7da6399833d609ee89a2587f2
python -m pip install --target .validation-dspark/reference-deps --no-deps \
  transformers==5.10.2 tokenizers==0.22.2
PYTHONPATH="$PWD/.validation-dspark/reference-deps" \
  python -m tools.dspark_reference_check \
    --deepspec-checkout .validation-dspark/DeepSpec \
    --output .validation-dspark/reference-result.json
```

With the audit runtime's newer Transformers 5.16.1, the official Gemma4 model
failed to initialize because its expected configuration attributes had changed.
Using 5.10.2 resolved that reference-side failure. Do not patch the oracle's model
equations to make a port pass. The helper is a manual audit tool and adds no
DeepSpec dependency to the serving package.

### Repeat the real-checkpoint checks

The selected target is `mlx-community/Qwen3-4B-4bit` at
`4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`; the drafter is the official Qwen3-4B
revision above. Their weight files are 2,263,022,529 and 2,786,273,970 bytes.
Weight downloads did not finish during the original audit. They completed during
M0-M2 implementation, enabling the [capture and lifecycle checks](dspark-progress.md).
No model weights or partial downloads are committed to the repository.

The following uses local immutable snapshots for reproducibility. M0 also
forwards the resolved draft revision when using a repository ID. Use the updated
checkers on the implementation branch; running the audited baseline remains
useful only to capture failures and limited diagnostics.

```bash
hf download mlx-community/Qwen3-4B-4bit \
  --revision 4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25 \
  --include '*.json' '*.safetensors' \
  --local-dir .validation-dspark/target
hf download deepseek-ai/dspark_qwen3_4b_block7 \
  --revision 3457dff1417cb84927f6098a5fcb7cee85c934b7 \
  --include '*.json' '*.safetensors' \
  --local-dir .validation-dspark/draft

python -m tools.dspark_lifecycle_check \
    --target "$PWD/.validation-dspark/target" \
    --draft "$PWD/.validation-dspark/draft" \
    --concurrency 4 --width 2 --prefix-cache \
    --output-dir "$PWD/.validation-dspark/results"
```

The new checker preserves engine logs/results, imposes a worker timeout, requires
positive proposal/verification/acceptance counts, and exercises prefix hits and
cancellation/cleanup. The older `check_sd_lossless` retains its F7 limitations.
Streaming and natural stopping still require the broader E4 qualification;
these bounded probes and their wall times are not production benchmarks.

## Required experiments

Statuses below are requirements, not claims that unimplemented modes were tested.
The implementation milestone IDs refer to the [roadmap](dspark.md#implementation-sequence).
Each experiment needs a committed harness and an attached machine-readable result
before its status can change to passed.

### Hardware tiers

The experiment inventory is the audit M4 with 32 GB RAM and the collaborator's
M5 Max with 48 GB RAM and 2 TB storage. The latter configuration is reported
availability; no experiment has yet run on it. Confirm actual free RAM/disk,
OS and runtime versions on arrival. The estimates below are not measured capacity
guarantees. Follow the aggregate memory equation in the specification, including
OS and allocator headroom. Begin at concurrency one and increase only after
measuring peak memory. Do not plan runs requiring a larger machine.

| Tier | Intended experiments | Qualification boundary |
| --- | --- | --- |
| Local M4, 32 GB RAM | Tiny FP32 oracles; Qwen3-4B quantized target/draft smoke, serving and calibration. Add bounded unquantized 4B comparisons after the loader/resource plan exists. | E1-E9 for the explicitly tested 4B envelope. Checkpoint transfers must finish; this host is not blocked by model weight capacity for the selected quantized smoke. |
| M5 Max, 48 GB RAM / 2 TB storage | Repeat the matched 4B workload for device comparison; qualify quantized 8B/14B and Gemma4-12B, then short-context unquantized comparisons where the measured peak fits. | E8-E11 within a declared 48 GB envelope. Start larger models at context 2,048/concurrency 1, then sweep 2/4/8 requests and longer contexts only after each capacity check. No implied 8,192-token/concurrency-32 capacity. |
| Outside available hardware; deferred | Integrated V4, or any standalone precision/context/concurrency case that cannot fit either machine. | Record the failing resource estimate or measured admission limit, model identity, missing backend and experiment ID. Keep it open without attempting an oversized run or substituting an extrapolated result. |

Use the M5 Max's storage for immutable snapshots, reusable tokenized datasets,
feature fixtures and traces. For expensive numerical comparisons, capture bounded
target features first, release the target and its caches, then compare Torch and
MLX draft stages against those fixtures. Sequential reference runs can avoid
holding two large model copies simultaneously. Such staged numerical tests do
not establish serving capacity; E8/E9 must keep the actual target and drafter
resident together. Swap/offload timing is not evidence for an in-memory serving
claim. Do not count checkpoint transfer or fixture I/O as decode computation.

Preserve only required checkpoint variants, record free storage before each
download/conversion, and avoid parallel benchmark runs on the same device.
Collect and fit a separate cost artifact for each Mac; do not reuse M4 timing
curves on M5 Max. Repeat calibration validation if device/precision/kernel changes
alter numerical behavior.

### Experiment matrix

| ID / purpose | Setup and assertions | Dependency / execution tier |
| --- | --- | --- |
| E1: Numerical oracle | Deterministic tiny Qwen3/Gemma4 fixtures; compare embedding, fused context, per-layer K/V, hidden, base/corrected logits, confidence and incremental updates against pinned official code. Require max error tolerances justified by dtype; test batch permutation and ragged rows. | Extend current probes at M1; 32 GB Mac + CPU reference. |
| E2: Target capture | Real target with capture on/off at identical input shape; compare logits and feature layers. Cross product of selected/all logits and no/single/multiple captured layers. Include mixed decode + unequal/chunked prefills, cached prefixes, and prompt logprobs. | M1; Qwen3-4B on 32 GB first. Must reproduce F1 before fixing it. |
| E3: Lifecycle and cache | Enumerate accepted lengths 0..K; assert exact absolute spans and every layer's physical length. Add cancellation, same-step ID reuse, preemption/resume, skipped drafting, empty context, interrupted prefill, repeated prefix hits, and out-of-memory recovery. Assert rejected suffix never becomes shared prefix state. | M2/M3; model-free CI plus real 4B serving on 32 GB. |
| E4: Greedy e2e | Compare target-only and DSpark token IDs on identical tokenized prompts, precision and execution settings. Test K=1/2/4/7, concurrency 1/4 and mixed arrivals, output budgets 1/2/31/128, EOS/min tokens/stops, streaming and non-streaming. Require actual proposed and verified tokens >0. | M4; 4B locally, repeat every claimed model/quantization pair. |
| E5: Stochastic correctness | Exact tiny-vocabulary outcome enumeration for p=q, disjoint/partial support, tiny mass, first/middle/full acceptance, variable K and K=0. Distribution tests for short sequences with independent seeds, top-p/top-k/temperature and each implemented target transform. Check target logprob and stopping semantics separately. | M5; CPU/MLX first, real 4B after integration. Statistical power and family-wise error limits must be fixed before inspecting results. |
| E6: Causal scheduling | Perturb token k and every suffix while keeping its visible prefix/history fixed: admission through k must not change. Include equal/zero confidence, all K=0, stale history, changing batch membership, jagged timing curves, and budget clipping. Compare planner to a simple oracle. | M6; no large model required. Include a negative retrospective-search fixture. |
| E7: Calibration | Disjoint fit/evaluation splits with pinned chat, code and math data. Record uncensored full proposals and actual accepted-prefix labels. Fit per-position STS; publish ECE, Brier, reliability bins, effective sample counts and uncertainty, per sampling/quantization mode. | M5/M6; real 4B locally. Larger pairs require their own calibration, not copied temperatures. |
| E8: Cost model and benefit | Profile completed draft/verify/host costs over active requests, context lengths, verify tokens and padding. Validate predictions on shapes not used for fitting. Compare adaptive DSpark with target-only and the best fixed-K selected on a separate tuning split. | M6/M7; 32 GiB initial tier; repeat for each broader hardware/workload claim. |
| E9: Production serving and soak | OpenAI-compatible HTTP requests with arrivals, streaming, client disconnects, timeouts and mixed request settings. Force memory pressure/preemption; verify cleanup after drain. Sustained mixed traffic for >=1 hour and >=10,000 completed/cancelled requests. | M7; increase concurrency only within the declared resource envelope. |
| E10: Larger Qwen3 pairs | Same numerical/capture/lifecycle/sampling/performance gates for 8B and 14B at their qualified quantizations and context limits. | M8, reusing M4-M7 gates; M5 Max 48 GB, quantized first and bounded unquantized cases if measured capacity permits. Any non-fitting case remains deferred. |
| E11: Gemma4 pair | Pin licensed target access and runtime; validate scaled embedding, global head dimensions, proportional/partial RoPE, sandwich norms/scalar, softcap, target sliding/full attention and KV-sharing boundaries; text-only initially. Repeat E2-E9. | M8; tiny oracle locally, real 12B on M5 Max 48 GB after target adapter and resource qualification. Start quantized at bounded concurrency. |
| E12: Integrated V4 | Establish target-only parity, weights/quantization, mHC/MoE, sparse/compressed attention, target feature reduction and all cache transitions. Then qualify integrated draft stages and repeat stochastic/scheduler/serving gates. | Deferred beyond the available 32/48 GB hardware and missing V4 backend. Valuable for integrated DSpark completion, but not executable in the current matrix. |

Use a CPU oracle for exact probability identities and an independently expressed
test model for state transitions. Tests that merely repeat implementation logic
are insufficient. Random generation is useful for counterexamples, but every
discovered failure needs a small deterministic regression.

### Performance protocol and proposed acceptance criteria

Freeze the workload and comparison protocol before tuning. These are proposed
release criteria, not DeepSeek's reported results or measured branch guarantees.

- Start with Qwen3-4B on the 32 GB tier: input lengths 128/1,024/4,096, output
  lengths 32/128/512 where the context limit permits, concurrency 1/4/16, plus
  a heterogeneous arrival stream. Test 8,192-token contexts and concurrency 32
  as capacity allows; a declared capacity limit is recorded, not a failed speed
  measurement silently removed from the report.
- Use identical target weights, tokenizer/chat mode, dtype, sampling, prompt
  token IDs, output stopping and runtime configuration for the causal comparison.
  Separately compare against the best supported target-only production setup
  (including async scheduling if it helps); synchronous-baseline isolation alone
  can overstate the deployable benefit.
- Include cold startup, cold prefixes and warm prefixes separately. Use fixed
  decode lengths for timing isolation and natural stopping for user-visible
  goodput. Do not compare faster runs that produced fewer useful tokens.
- Run warmups until compilation is excluded and timings stabilize; run at least
  five independent measured repetitions per selected scenario, alternating run
  order. Record thermal/power state, other load, memory pressure and process
  lifetime. Extend measurement when confidence intervals cannot decide a gate.
- Report TTFT, end-to-end latency, per-request TPOT and p50/p95/p99 streaming
  inter-token gaps, completed requests/sec, useful output tokens/sec and goodput
  at a declared latency SLO. Token bursts from speculation must not hide long
  pauses behind an average TPOT. A saturated throughput point alone is insufficient.
- Require a >=10% median improvement in TPOT or latency-constrained goodput on
  at least one predeclared target workload, with the paired 95% interval excluding
  no improvement. Report all other scenarios. Adaptive mode should remain within
  5% of target-only goodput and p95 latency in its declared load envelope, or
  demonstrate pre-draft bypass that meets those limits. If it cannot, narrow the
  enabled envelope explicitly; do not call it a universally beneficial default.
- Require no unexplained greedy divergence or stochastic bias, no request loss,
  duplicate streamed tokens or unexplained errors, and no monotonic active-memory
  growth after repeated drain/reuse cycles. Allocator-reserved memory may remain
  cached; report it separately from live request-owned allocations.
- Verify aggregate measured peak memory fits the declared budget throughout
  load/recovery. Publish acceptance by position, proposal/verify counts, context
  replay volume, byte traffic/allocation profiles and step-time breakdown so an
  apparent gain can be traced to less work rather than a fallback or mismatch.

The project's existing [profiling guide](../profiling.md) and
[performance contribution requirements](../CONTRIBUTING.md) provide the native
measurement workflow. Save raw benchmark JSON and command lines with the change;
do not publish an unsupported universal K value.

### Required result bundle

Every real-checkpoint or performance result must contain:

```text
implementation_commit; reference_commit; test_harness_commit
target/draft repo + immutable revision (or local content hashes)
tokenizer/template hashes; precision and quantization/conversion settings
OS/device/RAM; dependency versions; kernel/build/feature configuration
seed(s); dataset revision + split; prompt token IDs; request settings
requested/actual output token IDs, lengths, finish reasons and errors
proposed/scheduled/verified/accepted/correction/bonus counts
latency distributions, throughput/goodput, timings, peak/live/reserved memory
warmup/repetition policy; paired observations and confidence intervals
pass/fail gate; remaining limitations; reproduction command
```

Public benchmark fixtures should use shareable prompts. Keep raw private request
text out of production telemetry; bounded diagnostic capture should be explicit.

## Handoff order and open experiments

After M0-M2: E1 remains **partially covered** by the committed tiny-model checker;
E2/F1 and E3/F2-F3 pass the Qwen3 capture/context milestone gates, including real
4B capture parity and bounded lifecycle probes. E4 has **partial 4B evidence**:
40 paired request outputs at concurrency 1/4 and K=2/7 match baseline, with
positive verification and acceptance. Full serving/resource/performance
qualification and E5-E12 remain **open**. E5-E9 require implementation, not different hardware.
E10/E11 need model-specific implementation and capacity/access checks. E12
requires a V4 backend and hardware beyond both available Macs.

1. Preserve the passing F1-F3 regressions and existing runner lifecycle owner.
   Use the new local-checkpoint checkers as the starting smoke tests.
2. Implement the complete immutable pair/resource plan before increasing context length,
   model size or concurrency. Re-run local 4B tests with positive verification
   counters, prefix caching and mixed arrivals.
3. Complete E1-E4 locally for the supported greedy subset. Keep reference tests
   isolated from server dependency changes. Fix divergence before measuring
   optimization; do not label every mismatch a floating-point tie.
4. Implement stochastic proposal ownership and verifier before collecting
   stochastic calibration data. Run E5/E6 before enabling adaptive sampling.
5. Fit calibration/cost artifacts on tuning data, freeze them, then execute
   E7-E9 on held-out traffic. Only now select performance defaults.
6. Repeat 4B qualification on the M5 Max, then execute E10/E11 within its 48 GB
   resource envelope. Track missing access/backend prerequisites and any
   non-fitting cases alongside the test ID and command/artifacts required.
   E12 stays a separate deferred coverage requirement beyond available hardware.

Unavailable hardware is a reason to defer a particular experiment, not to infer
its result. Missing implementation is a separate reason: neither a 32 GB nor a
larger Mac can qualify a sampler or target adapter that has not been implemented.
At each implementation milestone, update this record with the new commit,
results and remaining experiments. Keep narrower release status explicit while
integrated V4 or another claimed support tier remains unqualified.

[baseline]: https://github.com/mhdimo/vllm-metal/tree/eea6f8194828dda88186277312527a36c131367a
[adapter]: https://github.com/mhdimo/vllm-metal/blob/eea6f8194828dda88186277312527a36c131367a/vllm_metal/v1/model_adapter.py#L427-L494
[runner]: https://github.com/mhdimo/vllm-metal/blob/eea6f8194828dda88186277312527a36c131367a/vllm_metal/v1/model_runner.py
[proposer]: https://github.com/mhdimo/vllm-metal/blob/eea6f8194828dda88186277312527a36c131367a/vllm_metal/v1/dspark_proposer.py
[model]: https://github.com/mhdimo/vllm-metal/blob/eea6f8194828dda88186277312527a36c131367a/vllm_metal/v1/dspark/model.py
[config]: https://github.com/mhdimo/vllm-metal/blob/eea6f8194828dda88186277312527a36c131367a/vllm_metal/v1/dspark/config.py
[loader]: https://github.com/mhdimo/vllm-metal/blob/eea6f8194828dda88186277312527a36c131367a/vllm_metal/v1/dspark/loader.py
[cache]: https://github.com/mhdimo/vllm-metal/blob/eea6f8194828dda88186277312527a36c131367a/vllm_metal/v1/cache_policy.py#L1157-L1184
[tests]: https://github.com/mhdimo/vllm-metal/blob/eea6f8194828dda88186277312527a36c131367a/tests/test_dspark_proposer.py
[checker]: https://github.com/mhdimo/vllm-metal/blob/eea6f8194828dda88186277312527a36c131367a/tools/check_sd_lossless.py
[guide]: https://github.com/mhdimo/vllm-metal/blob/eea6f8194828dda88186277312527a36c131367a/docs/speculative_decoding.md
[port]: https://github.com/ARahim3/mlx-dspark/tree/eb2c1a70c185c984980ea4864b3059abcb932fd5
[port-license]: https://github.com/ARahim3/mlx-dspark/blob/eb2c1a70c185c984980ea4864b3059abcb932fd5/LICENSE
[release]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/README.md
[q4]: https://huggingface.co/deepseek-ai/dspark_qwen3_4b_block7/blob/3457dff1417cb84927f6098a5fcb7cee85c934b7/config.json
[q8]: https://huggingface.co/deepseek-ai/dspark_qwen3_8b_block7/blob/03326e5043815da1f81b109078b2889737c26017/config.json
[q14]: https://huggingface.co/deepseek-ai/dspark_qwen3_14b_block7/blob/83207b416acf99f41c2184648923632fccea6dd0/config.json
[g12]: https://huggingface.co/deepseek-ai/dspark_gemma4_12b_block7/blob/2fa72e765eec2965fc4d86a8663ce6769eba6218/config.json
[vf]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/62af8fffb2f7030cac4de2f0169f5b8d1101b646/config.json
[vp]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark/blob/7c09739fd136abfb70a49ec334157f65f45b52cd/config.json
