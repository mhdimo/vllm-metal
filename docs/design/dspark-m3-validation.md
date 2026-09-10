# DSpark M3 resource and precision validation

M3 implements deterministic loading, bounded context storage and complete draft
resource accounting. Qualification here is for the pinned Qwen3-4B pair on an
M4 with 32 GB unified memory. It does **not** establish a production DSpark
release: extended generation exposed an exact-token parity failure that remains
an explicit M4 gate. The failed experiments are preserved below.

The [development handoff](dspark-handoff.md) provides the verified integration
revision, fresh M5 Max environment, evidence archive and ordered reproduction
commands. No destination-machine qualification is implied by this M3 record.

## Model pair and environment

| Component | Identity |
| --- | --- |
| Target | `mlx-community/Qwen3-4B-4bit`, revision `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`; supplied affine 4-bit/group-64 weights, BF16 compute |
| Draft | `deepseek-ai/dspark_qwen3_4b_block7`, revision `3457dff1417cb84927f6098a5fcb7cee85c934b7`; official BF16 source, serving conversion to MLX affine 4-bit/group-64 including embedding and prediction heads |
| Official oracle | DeepSpec `005e03b81cec38b7da6399833d609ee89a2587f2`, unmodified Qwen3 model on Torch CPU |
| Hardware | Apple M4, 32 GiB RAM, macOS 15.6; Metal recommended working set 22,906,503,168 bytes |
| Serving allowance | Fraction 0.22 of the recommended working set: 5,039,430,696 bytes |
| Runtime | Python 3.12.13, MLX 0.32.1, mlx-lm 0.32.0 (`9e6acca691e64d6d8bb808c328fcdea459099cca`), vLLM 0.28.0+cpu, Torch 2.13.0 |
| Reference dependencies | Isolated Transformers 5.10.2/tokenizers 0.22.2; serving uses Transformers 5.16.1 |

DeepSpec pins Torch 2.9.1. This comparison validates model equations and the
loaded checkpoint through compatible APIs, rather than reproducing its complete
training/evaluation environment. The [memory artifact](dspark-memory-results.json)
records file hashes, settings and observed resource counters. Model weights and
large intermediate arrays are not committed.

## Resource checks

The target KV planner now sees both resident models and subtracts the full
DSpark context/capture/workspace reservation. Startup estimates conversion
overlap before reading weight data, and rejects an impossible context/workspace
reservation before target profiling. Each source tensor is checked for finite
values before conversion. MLX's allocation limit is a guideline; the explicit
planner/admission checks and observed peaks provide the evidence here.

The sustained engine run used four concurrent requests, K=7, model length 1,024,
64 scheduled tokens per step, greedy 64-token output and no target prefix cache.
Prompt lengths were 5/106/219/481. Four warmup rounds preceded 128 measured
rounds. It completed 528 batch requests, 16 cancellations and 16 public-ID reuses;
every completed stream matched the corresponding target-only stream. It verified
35,927 draft tokens and accepted 28,274. Active memory after drain was exactly
4,336,877,694 bytes on every measured round, with zero observed drift. Peak active
MLX memory was 4,483,220,458 bytes; maximum observed active plus allocator cache
was 4,655,569,162 bytes. The run lasted about 14.6 minutes with instrumentation;
its wall time is not a throughput benchmark or the M7 production soak.

The run separately exercised exhausted byte admission, exhausted context slots,
and one injected `MemoryError` after a real context write. Subsequent requests
resumed drafting. Fault injection tests the exception/recovery path without
exhausting the host's RAM. Fallback counters count observed spans, not unique
requests. Normal rounds require positive drafting, so fallback-only execution
cannot satisfy the qualification.

A direct allocation probe additionally filled all four 1,024-token contexts,
then executed the real seven-position batched drafter with the target model and
its KV pool resident. Synthetic sinusoidal features provide a reproducible
allocation fixture. Persistent context occupied exactly 83,886,080 bytes, matching
the plan. Peak active MLX memory was 4,730,879,166 bytes, below the allowance;
all temporary contexts were released. This is a full-capacity allocation test,
not a claim that an arrival workload kept every context full simultaneously.

After adding the final source-payload and startup guards, a fresh 16-round run
plus four warmups repeated all three controlled faults and the full-capacity
probe. All 80 batch completions and two ID reuses matched baseline; 4,903 drafts
were verified and 3,858 accepted. Active memory after drain again had zero range.
Peak active allocation, including loading and full-capacity drafting, was
4,730,879,170 bytes. This final run is also included in the memory artifact.

The short preemption case used four identical five-token prompts, 64-token
output, K=2 through the `draft_model` alias, model limit 80, and six
scheduler-visible KV blocks. The scheduler performed seven actual preemptions.
All four outputs matched baseline, with 206 verified/142 accepted draft tokens,
complete physical context invariants and cleanup after normal drain. Maximum
observed active plus allocator cache was 4,838,459,904 bytes. The override reduces
the scheduler's logical capacity; the physical Metal pool remains allocated
from the complete budget. No direct call to a private preemption method is used.

The old fraction 0.12 is an intentional negative startup control: draft
conversion is rejected before installing the drafter or allocating target KV.
Tiny real-buffer tests also check exact FP16/BF16/FP32 byte accounting, rollback,
growth boundaries, partial-write failures, restored allocator limits and recovery.

## Full-checkpoint precision

The [precision artifact](dspark-precision-results.json) contains all per-fixture
errors and acceptance counts. Sixteen fixtures use real target features from
four deterministic prompts at context lengths 8/64/256/512. Target, official
Torch BF16 draft, MLX BF16 draft and MLX affine-4 draft run in separate processes.
MLX ingestion splits at token 17 to exercise incremental storage and offsets.
The native mlx-lm target also reproduces all four 64-token baseline streams.

Gates were fixed before observing the comparison: normalized L2 error
`||actual-reference|| / max(1, ||reference||)` at most 0.05 for each BF16 array,
and quantized accepted-prefix totals at least 80% of MLX BF16's nonzero total.
The existing tiny FP32 Qwen3/Gemma4 oracle retains its stricter `atol=rtol=1e-5`.

Embedding, fused features, every context K/V layer, backbone hidden states,
base/corrected logits and confidence all pass the BF16 gate. Maximum observed
normalized L2 error was 0.032615. Official BF16, MLX BF16 and affine-4 each
accepted 86 of the possible 112 draft positions across these fixtures. Complete
draft rows matched official BF16 in 15/16 BF16 and 11/16 affine-4 cases; identical
draft outputs across different precisions are not assumed.

This qualifies a bounded greedy conversion trace, not general acceptance or
calibration quality. Affine-4's maximum confidence error was approximately 0.4535
under the same metric. Confidence is unused in the current serving proposer;
M6 must fit and validate calibration for the actual quantized serving recipe.
These repeated prompts are numerical fixtures, not a held-out workload corpus.

## Failed extended parity checks: M4 remains open

A 900-output-token run at model limit 1,024 failed strict greedy comparison in
three of four requests, first at output positions 172, 239 and 94 (zero-based).
Its baseline prompt lengths were 5/106/116/116 after the fixed input-limit
truncation. The first recorded verifier divergence selected token 60650 with
logit 20.0 over baseline token 279 with logit 19.75. The failure remains a failure;
the checker exits nonzero and saves the actual and expected token streams.

Replaying the identical committed prefix through native mlx-lm, with **no draft
model loaded**, selects different tokens under different teacher-forced chunk
sizes. In the first case, chunk size 1 selects 60650, while size 4 selects 279.
The [replay artifact](dspark-target-replay-results.json) includes the prefixes,
candidate logits and native results for every divergent request.

A separate 256-output-token preemption stress case, using four identical
106-token prompts, model limit 384 and 25 scheduler-visible blocks, also failed
exact parity. Even its target-only baseline produced three distinct streams
from those identical prompts. Its speculative run performed six preemptions.
The [additional replay](dspark-preemption-replay-results.json) preserves its
divergent prefixes for further diagnosis. This demonstrates target sensitivity
to execution history; it does not prove every divergence is benign or waive
the failed serving gate.

The larger preemption run did pass physical context, full drain and resource
checks, with zero retained active-memory drift. The checker records that evidence
and still exits nonzero for the failed exact-token comparison.

M4 must reconcile the target's numerical behavior across prefill, recomputation,
decode and verification, check cache/logits correctness at each first divergence,
and rerun these cases before claiming broad exact-token parity. Changing the
comparison tolerance or silently shortening the failing workload is not a
resolution. M3's passing short preemption test and allocation tests establish
their stated resource/lifecycle contracts only.

## Reproduction and next-machine handoff

Use the immutable snapshots above and the isolated reference dependencies from
the [reference setup](dspark-validation.md#reproduce-the-reference-model-checks).
Run GPU experiments sequentially and keep worker logs and result JSON files.

```bash
python -m tools.dspark_memory_check \
  --target /path/to/pinned/target --draft /path/to/pinned/draft \
  --rounds 128 --faults --capacity-probe --output-dir results/memory

python -m tools.dspark_memory_check \
  --target /path/to/pinned/target --draft /path/to/pinned/draft \
  --rounds 1 --warmup-rounds 0 --width 2 --method draft_model \
  --prompt-set shared-short --max-model-len 80 --output-length 64 \
  --num-gpu-blocks 6 --require-preemption --output-dir results/preemption

python -m tools.dspark_precision_check \
  --target /path/to/pinned/target --draft /path/to/pinned/draft \
  --deepspec-checkout .validation-dspark/DeepSpec \
  --reference-python-path .validation-dspark/reference-deps \
  --baseline-result results/memory/k0.result.json --output-dir results/precision
```

Reproduce the failing extended cases by using `--output-length 900` with
`--max-model-len 1024`, or the shared-prompt case with `--prompt-set shared`,
`--output-length 256`, `--max-model-len 384`, `--num-gpu-blocks 25`, K=2 and the
alias. Keep `--rounds 1 --warmup-rounds 0`. `--diagnose-mismatch` records the first
divergent verifier logits and stops early. Without that flag, resource checks
finish before the checker returns failure for a token mismatch.

The native target replay can consume a checker failure or its own committed
result artifact, so it can run without regenerating the long output:

```bash
VLLM_METAL_BUILD_FROM_SOURCE=1 python -m tools.dspark_target_replay \
  --target /path/to/pinned/target \
  --failure docs/design/dspark-target-replay-results.json --output replay.json
```

Repeat these checks on the M5 Max with 48 GB RAM / 2 TB storage before expanding
the envelope. It was not accessed for M3. Record its actual recommended working
set, budget and measured peaks; 2 TB storage does not increase serving RAM.
Larger Qwen3/Gemma4 pairs remain M8 work, within the existing 48 GB capacity
matrix. Integrated V4 still requires an unavailable target backend and memory
beyond both machines. M4 performance/HTTP behavior, M5 stochastic verification,
M6 calibrated planning and M7's at-least-one-hour/10,000-request HTTP soak remain
required; the experiments above do not substitute for them.
