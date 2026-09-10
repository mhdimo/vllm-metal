# DSpark implementation progress

The [specification](dspark.md) defines milestone gates; the
[audit and experiment handoff](dspark-validation.md) preserves baseline evidence.
All runtime support remains experimental until the corresponding model, memory
and serving qualification gates pass.

| Milestone | Status | Change and validation |
| --- | --- | --- |
| M0: Baseline and contract | Implemented and locally validated | Startup guards, resolved draft identity/revision, exact source provenance and normal lint coverage. Executable F1/F3 regressions track the remaining M1/M2 failures. |
| M1: Target capture | Planned | Native Qwen3 capture, selected logits and complete prefill feature spans. |
| M2: Context lifecycle | Planned | Exact per-request ingest, physical rollback, lifecycle invalidation and safe prefix-hit behavior. |
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

Three strict expected failures preserve the reproduced F1/F3 contracts. They
are release blockers, not successful correctness tests; M1 and M2 must remove
them by fixing the behavior. Existing runner lifecycle ownership is retained.
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
