# DSpark M4a: target parity contract and evidence

M3 left two exact-token parity failures open: a 900-token K=7 run and an
extended preemption run through the `draft_model` alias. This record states
what they are, proves it with both engines' logits, and defines the parity
contract M4 uses from here on. All measurements are from the M5 Max (48 GB,
macOS 26.6, MLX 0.32.1, `VLLM_METAL_BUILD_FROM_SOURCE=1`, NAX kernels loaded)
at `02f3b2d`, with the pinned Qwen3-4B pair from the M3 manifest. Raw artifacts
are under the run directories named below.

## Result

Both failures reproduce on the M5 Max and every first divergence has one of
two causes, neither of which is a DSpark defect:

| Class | Definition | Long-output K=7 | Preemption K=2 alias |
| --- | --- | --- | --- |
| Tie | Both engines rank the two tokens first and second within 2 bfloat16 ULPs | 3 of 4 requests | 0 of 3 |
| Target-unstable | The target-only engine, with no drafter, returns different greedy tokens for the same committed prefix under different execution shapes | 1 of 4 | 3 of 3 |
| Defect (inadmissible) | Anything else | 0 | 0 |

The divergence sets are deterministic across a repeat run and unchanged by
the verify-window kernel (`VLLM_METAL_SPEC_VERIFY_WINDOW=1`); with
`VLLM_METAL_DISABLE_NAX=1` the tie positions move and the unstable one stays.

## Mechanism

The qualification prompts repeat one sentence seven times and generate 256 to
900 greedy tokens with `ignore_eos`. The target enters a loop with the argmax
logit near 41 and a bistable state: at some positions one execution lands in
that "content" basin (token 9664, logit 41) and another lands in a "junk"
basin (tokens 198 / 320 / 582, logits near 12). Evidence, all target-only:

- Four identical 106-token prompts, no preemption, no drafter
  (`m5max-m4a-isolate-01/nopreempt`): all six pairs diverge from each other at
  output position 146 (absolute 252); three requests show the junk basin
  there and one at absolute 207. Their prefill chunking differed only because
  the 64-token budget was split across four arrivals.
- `tools/dspark_target_stability.py` (`m5max-m4a-stability-02`): every one of
  the seven divergent prefixes yields more than one greedy token when only the
  prefill chunk budget changes (64, 16, 4 tokens, NAX off, MLX cache path):

  | Prefix (request / absolute position) | chunk 64 | chunk 16 | chunk 4 | chunk 64 no NAX | MLX cache path |
  | --- | --- | --- | --- | --- | --- |
  | long 0 / 177 | 279 | 60650 | 279 | 279 | 279 |
  | long 1 / 207 | 198 | 9664 | 198 | 198 | 9664 |
  | long 2 / 324 | 279 | 264 | 279 | 279 | 264 |
  | long 3 / 272 | 369 | 369 | 279 | 279 | 369 |
  | preempt 0 and 1 / 252 | 9664 | 582 | 198 | 9664 | 9664 |
  | preempt 3 / 207 | 198 | 9664 | 198 | 198 | 9664 |

- Native mlx-lm with no vllm-metal kernels (`dspark_target_replay` and
  `native_chunking`): at absolute 252 five of nine teacher-forcing plans land
  in the junk basin (including 64-token chunks from position 0); at absolute
  207 all nine land in the content basin while the paged engine lands in junk
  for three of five shapes. The paged and MLX-cache paths therefore round
  differently near this knife edge; both are valid bfloat16 executions of the
  same model and neither is a wrong computation (the kernel parity suites
  pass), so the asymmetry is recorded as an observation, not a defect.
- The one M3 preemption witness (baseline chose 320 at absolute 207 where
  native chose 9664) is this mechanism in the target-only baseline; it is not
  preemption-specific and not a recompute bug: the runner's resume path
  re-prefills from position 0 with `cached_final` sampling and the traces show
  `cache_start_pos == state_len - 1` for every row.

DSpark's own behaviour in the traces is correct: at absolute 207 the drafter
proposed 9664 and the target's verification row rejected it with junk logits;
the correction token then continued the junk basin exactly as a target-only
engine in that basin does. Verification uses the target's own multi-row
logits, which are one more execution shape.

## Contract

Greedy speculative decoding must reproduce the target-only greedy stream
except at positions where the target's own greedy decision is not numerically
stable under bfloat16 execution. A first divergence is admissible only if:

1. it is a **tie**: both engines rank the two tokens first and second, each
   within `MAX_TIE_ULPS = 2` bfloat16 ULPs (upstream `#524`'s criterion), or
2. it is **target-unstable**: `tools/dspark_target_stability.py` returns more
   than one greedy token for that committed prefix across its fixed shape set
   (paged prefill budgets 64/16/4, NAX disabled, MLX cache path).

Every other divergence, and any untraced or trace-inconsistent one, fails the
gate. Ties and unstable prefixes are reported with counts; a run with either
is not called lossless. The checkers keep strict exact-token comparison as
their primary result and exit nonzero on any mismatch; the gate is a separate,
recorded step over the traced logits:

```bash
python -m tools.dspark_memory_check ... --trace-logits --output-dir "$RUN"
python -m tools.dspark_target_stability --target "$DSPARK_TARGET" \
  --artifact "$RUN/k7.result.failure.json" --output "$RUN/stability.json"
python -m tools.dspark_divergence_classify --run-dir "$RUN" --width 7 \
  --stability "$RUN/stability.json" --gate --output "$RUN/gate.json"
```

A defect that corrupts draft or target state shows as an engine-disagreement
at a stable prefix and still fails. A defect that only manifests at an
unstable prefix cannot be separated from the instability; that limitation is
accepted and stated.

## Non-degenerate workload

The repeated-sentence fixtures are kept because they exercise preemption and
long contexts, but a parity claim also needs prompts where the target's greedy
decisions are stable. `--prompt-set natural` supplies eight varied prompts
(explanations, a story, code, comparisons) that ask for long answers. Results
of the natural workload on the M5 Max are recorded in the section below.

## M5 Max natural workload results

`m5max-m4a-natural-01`, eight natural prompts of 23 to 35 tokens, 256 greedy
output tokens, four concurrent requests, model length 1,024 (512 for the
preemption case), memory fraction 0.22:

| Case | Strict exact parity | First divergences | Gate | Drafts verified / accepted | Preemptions |
| --- | --- | --- | --- | --- | --- |
| K=7 | fails, 8 of 8 requests | output positions 4, 5, 34, 38, 52, 62, 81, 146 | PASS: 8 ties, 0 target-unstable, 0 defects | 4,732 / 1,352 | 0 |
| K=2 `draft_model` alias, 60 scheduler blocks | fails, 5 of 8 requests | output positions 6, 54, 62, 205, 227 | PASS: 5 ties, 0 target-unstable, 0 defects | 1,861 / 1,100 | 4 |

Every natural divergence is a tie of 0 or 1 bfloat16 ULP in both engines,
most of them exact ties that `argmax` breaks by index in one engine and by a
one-ULP difference of the multi-row forward in the other. With bfloat16
logits, whose spacing is 0.125 between 16 and 32, exact ties between the two
best tokens are common, so strict exact-token parity with a one-row baseline
fails within a few hundred tokens for most prompts on this 4-bit target under
any multi-row verification, speculative or not. The target-stability probe
also marks five of the eight K=7 tie prefixes and four of the five K=2 tie
prefixes unstable, which is consistent: a tie is where a rounding-order
change decides.

Acceptance on natural text is far below the repeated-sentence fixtures: 28.6%
of drafted tokens at K=7 (about 1.4 accepted per window) and 59.1% at K=2,
against 78.7% at K=7 on the repeated set. This is the first measurement that
matters for M4d: on natural prompts most of a K=7 window is rejected, so the
fixed-K performance study must weigh verification rows against acceptance
rather than assume the repeated-set rate.
