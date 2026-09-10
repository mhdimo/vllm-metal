# DSpark development and experiment handoff

This is the migration entry point for continuing DSpark on an **M5 Max with
48 GB unified memory and 2 TB storage**. Status was checked against the remote
`mhdimo/vllm-metal` repository on **2026-09-10**. The completed work was tested
on an **M4 with 32 GiB RAM and macOS 15.6**; the M5 Max has not been accessed or
qualified. In this document, **M0–M8 name implementation milestones**, while
**M4 / M5 Max hardware** is identified explicitly as a machine or chip.

## Current status and exact starting point

Continue on the collaborator fork's **`Dspark`** branch. Its verified runtime
integration commit is **`65f67179c966a2cf90a6b973161b107645929bb2`**. Its tree
matches the final tested M3 feature head
**`bae84f915d4d30c9bb30382bd9445c7f12f12fc7`**. The runtime hashes recorded in
[the memory artifact](dspark-memory-results.json) were rechecked against that
checkout and all matched. This handoff adds documentation and migration
constraints; it makes no runtime change or new model-performance claim.

At the status check, local `Dspark` and `origin/Dspark` were identical and clean,
all six implementation PRs below were merged, and no DSpark experiment workers
were running on the source machine. There was no open implementation PR or
uncommitted runtime work to transfer. Later documentation commits are expected;
record the actual checkout SHA when starting new experiments.

| Milestone / change | Status and integration record |
| --- | --- |
| M0: supported contract and provenance | Complete, [PR #1](https://github.com/mhdimo/vllm-metal/pull/1) |
| M1: native target capture | Complete for the named Qwen3 target, [PR #2](https://github.com/mhdimo/vllm-metal/pull/2) |
| M2: exact request context lifecycle | Complete within the recorded lifecycle probes, [PR #3](https://github.com/mhdimo/vllm-metal/pull/3) |
| M3a: deterministic streamed loading | Complete, [PR #4](https://github.com/mhdimo/vllm-metal/pull/4), feature `19c010ff10f61aafc8abf4d45eb499e08ee45728` |
| M3b: full resource planning and bounded storage | Complete, [PR #5](https://github.com/mhdimo/vllm-metal/pull/5), feature `b4aa6958bcf5208901b755552d98f7e2d25383fc` |
| M3c: resource, recovery and precision qualification | Complete for its named 4B envelope, [PR #6](https://github.com/mhdimo/vllm-metal/pull/6), feature `bae84f915d4d30c9bb30382bd9445c7f12f12fc7` |
| M4: complete fixed-greedy serving and performance | Complete on M5 Max for the pinned 4B pair: M4a, the extended exact-token failures are ties or target-unstable prefixes under the [parity contract](dspark-m4-parity.md); M4b, configurable admission caps measured on a real engine; M4c, the HTTP serving matrix passes with every divergence a tie; M4d, fixed K=2-4 meets the 10% gate at one request (+36-49% on short prompts, +10-14% on 1,024-token prompts) and every width loses 4-28% at four concurrent requests because multi-row verification is expensive on this Metal path (see the [progress record](dspark-progress.md)). |
| M5: exact stochastic verification | Complete on M5 Max for the pinned 4B pair: plain temperature/top-k/top-p requests draft from exact float32 proposal distributions kept with the scheduled proposal and are verified by rejection sampling with per-request random streams; enumerated tiny-vocabulary oracle, powered distribution tests and the real-engine `tools/dspark_stochastic_check.py` gate (see the [progress record](dspark-progress.md)). Penalties, logprobs, constraints and structured output stay target-only. |
| M6: confidence calibration and adaptive planning | Open; confidence is not used by the current proposer. |
| M7: production performance and reliability | Open; no qualifying HTTP soak or production speedup result. |
| M8: additional standalone model pairs | Open; each pair needs its own adapter, precision, capacity and serving gates. |
| Integrated DeepSeek-V4 | Separate deferred workstream: missing target backend and beyond available memory. |

**DSpark is experimental, not production-complete.** Passing M0–M3 establishes
specific foundations, not broad lossless serving or satisfactory production
performance. In particular, the long-generation failures are still release gates.

`Dspark-implement` is an older divergent prototype, not the integration base.
At the audit it ended at `2b20dcc`, with two commits unique to that branch and
142 unique to `Dspark`. Do not switch to it or merge those commits by name alone.
The fork's default branch is `main`, so explicitly select `Dspark` when cloning.

## Reading order and evidence map

| Document | Purpose |
| --- | --- |
| [Specification and roadmap](dspark.md) | Canonical architecture, invariants, milestone gates, performance protocol and pinned primary references |
| [Implementation progress](dspark-progress.md) | What changed and passed in M0–M3 |
| [M3 qualification](dspark-m3-validation.md) | Actual resource/precision results, limitations and failed parity experiments |
| [Original audit and experiment matrix](dspark-validation.md) | Historical baseline F1–F9 and E1–E12; do not treat original findings as current unresolved code defects without checking progress |
| [Memory and identity manifest](dspark-memory-results.json) | Target/tokenizer/draft file hashes, final implementation hashes, measured capacities and recovery |
| [Precision results](dspark-precision-results.json) | Full-checkpoint BF16 and serving quantization comparisons |
| [Long-output replay](dspark-target-replay-results.json), [preemption replay](dspark-preemption-replay-results.json) | Self-contained first-divergence prefixes usable by the native target replay tool |
| [Tiny reference results](dspark-reference-results.json), [historical lifecycle results](dspark-lifecycle-results.json) | Earlier oracle and real-engine evidence; historical M2 memory accounting was superseded by M3 |
| [Migration constraints](dspark-runtime-constraints.txt) | Observed serving/development package versions; not a replacement for project dependency policy |

The official standalone oracle is
[DeepSeek DeepSpec](https://github.com/deepseek-ai/DeepSpec/tree/005e03b81cec38b7da6399833d609ee89a2587f2),
pinned at `005e03b81cec38b7da6399833d609ee89a2587f2`. The independent MLX port
used by the original branch is a different source: its provenance is
`9e39ea2fdc6d99d855af2cb7ef9933391c4391db`, with the MIT notice retained in
`vllm_metal/v1/dspark/NOTICE`. Keep those origins distinct.

## Implementation ownership and invariants to preserve

Paths here are relative to the new repository checkout.

| Owner | Responsibility / where the next changes belong |
| --- | --- |
| `vllm_metal/platform.py`, `vllm_metal/v1/dspark/contracts.py` | Fail unsupported global configurations before target weights load |
| `vllm_metal/v1/dspark/config.py`, `loader.py` | Exact family/head/RoPE schema, authoritative shards, finite payloads, streamed quantization and startup estimates |
| `vllm_metal/v1/dspark/memory.py`, `model.py` | Complete memory plan, bounded context K/V and draft backbone / Markov heads |
| `vllm_metal/v1/model_adapter.py` | Native Qwen3 target semantics, full selected-layer features, independently selected logits |
| `vllm_metal/v1/dspark_proposer.py` | Absolute span ingestion, request-owned context, batched drafting and pressure fallback |
| `vllm_metal/v1/model_runner.py` | Model installation order, scheduled spans, request-generation lifetime and cleanup |
| `vllm_metal/v1/cache_policy.py` | Subtract complete draft reservation before target KV allocation |
| `vllm_metal/v1/spec_decode.py`, `sampling_batch.py` | Authoritative target verification, sampling eligibility and transforms |
| `vllm_metal/attention/`, `vllm_metal/metal/paged_ops.cpp`, `vllm_metal/metal/kernels_v2/` | Target attention/layout/numerical diagnosis; profile and reproduce before changing kernels |
| `tests/test_dspark_*.py`, runner/speculative/cache tests | Regression contracts; extend existing owners rather than building a second scheduler |
| `tools/dspark_*` | Bounded offline reference, capture, lifecycle, memory, precision and replay experiments |

1. A context belongs to the actual runner `RequestState` object, not merely a
   public request ID. Every physical K/V layer must cover exactly the committed
   prefix **[0, anchor position)**. Holes or inconsistent lengths require safe
   fallback or a complete rebuild, never inferred coverage.
2. Ingest every scheduled target feature span at its absolute position, including
   intermediate prefill, K=0 and currently ineligible requests. Verification
   contributes the old anchor and accepted draft inputs only. The correction or
   bonus is the next anchor. Trim physical context before replacing a recomputed
   suffix; rejected tokens must remain invisible when storage is reused.
3. Runner lifecycle reconciliation owns finish, cancel, preempt, resume and
   same-step ID reuse. Future proposal probabilities and planner history must
   join that cleanup path. Missing prefix-hit features currently discard private
   draft context and use target-only fallback until full recomputation from zero.
   There is no proposer-private cross-request prefix cache.
4. Keep all seven trained backbone positions when reducing K. The draft block is
   bidirectional and uses per-row absolute offsets. Each row's cap reserves a
   correction/bonus slot. Prove ragged batching against independent rows.
5. Check slots and bytes before allocating. The context cap defaults to 32
   (`VLLM_METAL_DSPARK_MAX_CONTEXTS`, planner-budgeted); context storage grows
   in chunks of 256 within a hard maximum. A binding per-step draft cap
   rotates least-recently-drafted requests first. Allocation recovery
   releases private contexts and preserves already computed target output;
   unrelated execution errors propagate. Cast target features to draft precision.
6. Budget both models, full context/capture/workspace and loading overlap before
   target KV profiling/allocation. Do not invent an autoregressive DSpark KV group
   with the wrong lifetime. The current full-attention 4B drafter uses
   **20 KiB per context token per request** before temporary workspaces.

The current supported global path is matched Qwen3 standalone vanilla-Markov
DSpark, paged attention, TP=1, synchronous scheduling and the supported greedy
request subset. `dspark` and `draft_model` are aliases for the same factory.
Unsupported per-request sampling transforms use the existing target-only path;
global unsupported settings fail early. LoRA, asynchronous DSpark scheduling,
integrated V4, unqualified target families, stochastic/adaptive configuration and
unqualified draft loading/quantization/cache overrides remain guarded. Read
`contracts.py` and `sampling_batch.py` before changing eligibility. The missing
`dspark/sampling.py` referenced by `sample_block_probs` is not a finished feature.

## M5 Max setup: fresh environment, pinned inputs

Use a native arm64 shell and Python, a current working Rust toolchain, `uv`,
`shellcheck`, and Xcode with its Metal toolchain. This checkout's normal source
installer builds the NAX artifact and requires a macOS **SDK at least 26.2**;
NAX also has a macOS 26.2 deployment floor. Record the actual OS, SDK and
selected runtime path on the M5 Max. More recent hardware can dispatch different
kernels, so copying M4 results or compiled artifacts cannot qualify it.

The following is a fresh-checkout recipe. Install/select the required Xcode and
Rust tools first if absent. Keep model and result data outside the repository.
Run these blocks from the repository root with the serving environment active.

```bash
mkdir -p "$HOME/Development"
git clone --branch Dspark --recurse-submodules \
  https://github.com/mhdimo/vllm-metal.git "$HOME/Development/vllm-metal"
cd "$HOME/Development/vllm-metal"
git merge-base --is-ancestor 65f67179c966a2cf90a6b973161b107645929bb2 HEAD
git status --short --branch
git submodule status --recursive
uname -m
xcrun -sdk macosx --show-sdk-version
rustc --version
uv --version
shellcheck --version

uv venv .venv-vllm-metal --python 3.12.13 --seed
UV_CONSTRAINT="$PWD/docs/design/dspark-runtime-constraints.txt" ./install.sh
source .venv-vllm-metal/bin/activate
uv pip install -c docs/design/dspark-runtime-constraints.txt -e '.[dev]'
uv pip check
```

Use this checkout's **local `./install.sh`**. It reads the pinned vLLM release tag,
installs the official `vllm-0.28.0+cpu` macOS arm64 Python 3.12 wheel, installs this
local project editable and builds native artifacts. A remote installer targeting
the latest release would not select the DSpark development tree.

The constraints snapshot contains 188 observed package versions, excluding the
editable project. Constraints only affect requested packages/dependencies; they
do not install every listed optional package. Important versions are Python
3.12.13, MLX/`mlx-metal` 0.32.1, Torch 2.13.0, Transformers 5.16.1, tokenizers
0.23.2, safetensors 0.8.0, and mlx-lm 0.32.0 **from git
`9e6acca691e64d6d8bb808c328fcdea459099cca`**, as pinned in `pyproject.toml`.
Version `0.32.0` alone does not identify the mlx-lm source. The snapshot is not
a hermetic SDK/build/wheel lockfile. If resolution or native build fails, preserve
the error and resolve the specific incompatibility; do not silently upgrade the
serving stack while claiming to reproduce M3. Record any necessary change and
repeat the appropriate correctness gates. Recreate environments and build caches
on M5 Max; do not transfer the M4 virtual environment or native binaries.

### M5 Max numerics: TF32 is the FP32 GEMM default

Measured on the M5 Max (applegpu_g17s, MLX 0.32.1, checkout `4af9a92`) with a
2,560-square random matmul against a float64 reference:

| Operation | Rows | Relative error / difference |
| --- | --- | --- |
| FP32 matmul, MLX default | 1 | 4e-7 |
| FP32 matmul, MLX default | 8 | 8e-4 |
| FP32 matmul, `MLX_ENABLE_TF32=0` | 8 | 9e-7 |
| BF16 matmul, 8 rows vs 1 row | 8 | 9.5e-4 (unchanged by the switch) |
| Affine 4-bit/group-64 matmul, 8 rows vs 1 row | 8 | 7.8e-3 (unchanged by the switch) |

Multi-row FP32 matmuls run on the tensor units at TF32-class precision unless
`MLX_ENABLE_TF32=0` is set before MLX loads; `MLX_METAL_GPU_ARCH` overrides the
same dispatch. The BF16 and quantized row-count differences are intrinsic to
the M=1 and M>1 kernels and exist on every Apple GPU; they are the mechanism
behind near-tie token flips between single-row decode and multi-row verify.
Consequences:

- `tests/conftest.py` and `tools/dspark_reference_check.py` pin
  `MLX_ENABLE_TF32=0`, because their FP32 oracles gate at 1e-5 or 2e-5.
  `tests/test_metal_numerics.py` fails if the pin stops holding.
- Serving keeps MLX's default. The serving path is BF16 and affine 4-bit, so
  the switch is not a serving-parity tool; record which setting an experiment
  used whenever an FP32 array is compared.
- Do not loosen an FP32 oracle tolerance to absorb TF32; fix the environment.

The first M5 Max environment used CPython 3.12.12: uv 0.9.18, which
`scripts/lib.sh` installs, has no download for 3.12.13. Record the interpreter
in every `environment.json`; no numerical result here depends on the patch level.

```bash
export DSPARK_DATA="$HOME/DSpark-data"
export DSPARK_RUN="$DSPARK_DATA/results/m5max-baseline-01"
export DSPARK_REFERENCE="$DSPARK_DATA/DeepSpec"
export DSPARK_REFERENCE_DEPS="$DSPARK_DATA/reference-deps"
mkdir -p "$DSPARK_RUN"
export VLLM_METAL_BUILD_FROM_SOURCE=1
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_METAL_USE_PAGED_ATTENTION=1
python - <<'PY'
import importlib.metadata as md
import json, os, platform, subprocess
from pathlib import Path
import mlx.core as mx
import psutil
from vllm.platforms import current_platform
assert platform.machine() == "arm64"
assert type(current_platform).__name__ == "MetalPlatform"
assert mx.metal.is_available()
record = {
    "python": platform.python_version(), "macos": platform.mac_ver()[0],
    "platform": type(current_platform).__name__, "metal": mx.device_info(),
    "physical_memory_bytes": psutil.virtual_memory().total,
    "swap": psutil.swap_memory()._asdict(),
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "versions": {n: md.version(n) for n in (
        "vllm", "vllm-metal", "mlx", "mlx-metal", "mlx-lm", "torch",
        "transformers", "tokenizers", "safetensors", "nanobind")},
}
Path(os.environ["DSPARK_RUN"], "environment.json").write_text(
    json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
PY
uv pip freeze > "$DSPARK_RUN/packages.txt"
git submodule status --recursive > "$DSPARK_RUN/submodules.txt"
xcrun -sdk macosx --show-sdk-version > "$DSPARK_RUN/sdk.txt"
```

Record NAX versus baseline attention dispatch and source-build versus packaged
artifacts in later experiment manifests. The checker workers explicitly use
source builds. Packaging/deployment qualification must separately exercise the
built artifact path. `VLLM_ENABLE_V1_MULTIPROCESSING=0` is set internally by the
instrumented lifecycle/memory workers; it is not a general production requirement.

### Pin and verify the model pair

| Component | Repository and immutable revision | Weight-file size |
| --- | --- | --- |
| Target | `mlx-community/Qwen3-4B-4bit` at `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25` | 2,263,022,529 bytes |
| Drafter | `deepseek-ai/dspark_qwen3_4b_block7` at `3457dff1417cb84927f6098a5fcb7cee85c934b7` | 2,786,273,970 bytes |

The target uses supplied affine 4-bit/group-64 weights with BF16 compute. The
64-tensor official BF16 draft is converted during loading to affine
4-bit/group-64, including embedding and prediction heads. Its five draft layers
consume target taps `[1, 9, 17, 25, 33]`, with block length seven. Full config and
tokenizer hashes are part of the committed memory manifest, not just the weights.

```bash
export DSPARK_TARGET="$DSPARK_DATA/models/target/4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25"
export DSPARK_DRAFT="$DSPARK_DATA/models/draft/3457dff1417cb84927f6098a5fcb7cee85c934b7"
python - <<'PY'
import json, os
from pathlib import Path
from huggingface_hub import snapshot_download
manifest = json.loads(Path("docs/design/dspark-memory-results.json").read_text())
for role, variable in (("target", "DSPARK_TARGET"), ("draft", "DSPARK_DRAFT")):
    item = manifest["model_pair"][role]
    snapshot_download(repo_id=item["repo"], revision=item["revision"],
        local_dir=os.environ[variable],
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
PY
```

Alternatively transfer the two already downloaded snapshots from the source
machine, **dereferencing Hugging Face snapshot symlinks**, for example with
`rsync -aL` to an export directory. Copying snapshot symlinks without the cache
blobs produces broken models. Place the exported contents in `DSPARK_TARGET` and
`DSPARK_DRAFT` above, then run the same verification below. The two weight files
total about 5.05 GB plus tokenizer/config files; no additional model is required
to resume M4 work. Keep access tokens out of Git and experiment artifacts.

```bash
python - <<'PY'
import hashlib, json, os
from pathlib import Path
manifest = json.loads(Path("docs/design/dspark-memory-results.json").read_text())
for role, variable in (("target", "DSPARK_TARGET"), ("draft", "DSPARK_DRAFT")):
    for item in manifest["model_pair"][role]["files"]:
        path = Path(os.environ[variable]) / item["name"]
        assert path.stat().st_size == item["bytes"], path
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"], path
    print("Verified pinned files:", role)
PY
```

### Isolate the official reference and documentation dependencies

```bash
git clone https://github.com/deepseek-ai/DeepSpec.git "$DSPARK_REFERENCE"
git -C "$DSPARK_REFERENCE" checkout --detach 005e03b81cec38b7da6399833d609ee89a2587f2
uv pip install --target "$DSPARK_REFERENCE_DEPS" --no-deps \
  transformers==5.10.2 tokenizers==0.22.2
uv venv "$DSPARK_DATA/docs-venv" --python 3.12.13
uv pip install --python "$DSPARK_DATA/docs-venv/bin/python" -r docs/requirements-docs.txt
```

Use a fresh reference dependency directory. Apply its `PYTHONPATH` only to the
individual tiny-reference invocation, or pass `--reference-python-path` to the
precision checker. Never globally replace serving Transformers with the reference
version. DeepSpec's requirements pin Torch 2.9.1; the recorded M3 oracle used
Torch 2.13.0 CPU with the isolated Transformers/tokenizers pair. This is an
explicit API/numerical comparison, not exact reproduction of the official full
environment. Preserve this distinction when interpreting results.

## Ordered validation on the destination machine

The lifecycle and memory checkers use `ignore_eos=True` to obtain fixed-length
comparisons. They do not qualify EOS, stop strings, HTTP streaming or production
arrival behavior; those remain M4 serving-harness work. The memory checker runs
its own K=0 baseline before the chosen speculative width; its public `--width`
argument accepts 1 through 7, not 0.

Run GPU experiments **sequentially** with no unrelated GPU workload. Use a fresh
output directory for every invocation: tools write fixed `k0`/`kN` files and can
overwrite earlier results. A failed checker is a failed gate; retain its exit
status and logs. Do not suppress failures to make a batch appear successful.
The commands below are supplied for the destination; they have not been run on
that machine.

### 1. Local regression and source checks

```bash
python -m pytest -m 'not slow' tests/ -q
ruff check .
ruff format --check .
mypy vllm_metal
shellcheck -- *.sh scripts/*.sh
READTHEDOCS_CANONICAL_URL=https://vllm-metal.readthedocs.io/ \
  "$DSPARK_DATA/docs-venv/bin/mkdocs" build --strict --site-dir "$DSPARK_RUN/site"
```

The final M3 source passed **2,315 tests, 15 skipped, 53 deselected**, with no
expected failures; Ruff, mypy (145 source files), shellcheck and strict docs also
passed on the source machine. Investigate changed results, especially device
specific kernel tests; do not simply copy the old count into new evidence.
On the M5 Max at `4af9a92` the same suite gave 2,310 passed and 5 failed at
MLX defaults (four `test_dspark_proposer.py` context-parity cases and one
Whisper feature case, all FP32 oracles); with `MLX_ENABLE_TF32=0` every one
passes. Ruff, mypy, shellcheck and the strict docs build passed unchanged.
The repository's complete `scripts/test.sh` additionally builds/verifies the
wheel and runs two ordinary target-serving smokes, which download their own
pinned small targets. Preserve that flow for packaging/runtime changes:

```bash
UV_CONSTRAINT="$PWD/docs/design/dspark-runtime-constraints.txt" scripts/lint.sh
env -u VLLM_METAL_BUILD_FROM_SOURCE \
  UV_CONSTRAINT="$PWD/docs/design/dspark-runtime-constraints.txt" scripts/test.sh
```

These scripts are broader release checks, not DSpark HTTP qualification. The
second invocation removes the source-build override for ordinary serving smokes
to exercise the freshly built artifacts. The scripts may reinstall dependencies;
check versions afterwards. The existing ordinary
smoke test permits an alternate numerical golden on different attention kernels;
that does **not** waive the explicit DSpark parity gates.

### 2. Tiny official equations, then real target capture

```bash
PYTHONPATH="$DSPARK_REFERENCE_DEPS" python -m tools.dspark_reference_check \
  --deepspec-checkout "$DSPARK_REFERENCE" --output "$DSPARK_RUN/reference.json"
python -m tools.dspark_target_check \
  --target "$DSPARK_TARGET" --draft-config "$DSPARK_DRAFT/config.json" \
  --output "$DSPARK_RUN/capture.json"
```

The tiny FP32 Qwen3/Gemma4 oracle uses `atol=rtol=1e-5` and checks incremental and
ragged context. It does not qualify a real Gemma4 target adapter. The real Qwen3
capture checker previously passed 12 bit-exact capture-on/off cases, including
physical target KV, selected logits and packed layouts.

### 3. Lifecycle, prefix fallback and actual short preemption

```bash
python -m tools.dspark_lifecycle_check --target "$DSPARK_TARGET" --draft "$DSPARK_DRAFT" \
  --concurrency 1 --width 7 --memory-fraction 0.22 --output-dir "$DSPARK_RUN/lifecycle-c1-k7"
python -m tools.dspark_lifecycle_check --target "$DSPARK_TARGET" --draft "$DSPARK_DRAFT" \
  --concurrency 4 --width 2 --memory-fraction 0.22 --output-dir "$DSPARK_RUN/lifecycle-c4-k2"
python -m tools.dspark_lifecycle_check --target "$DSPARK_TARGET" --draft "$DSPARK_DRAFT" \
  --concurrency 4 --width 7 --prefix-cache --memory-fraction 0.22 \
  --output-dir "$DSPARK_RUN/lifecycle-c4-k7-prefix"
python -m tools.dspark_lifecycle_check --target "$DSPARK_TARGET" --draft "$DSPARK_DRAFT" \
  --concurrency 1 --width 2 --prefix-cache --memory-fraction 0.22 \
  --output-dir "$DSPARK_RUN/lifecycle-c1-k2-prefix"
python -m tools.dspark_memory_check --target "$DSPARK_TARGET" --draft "$DSPARK_DRAFT" \
  --rounds 1 --warmup-rounds 0 --width 2 --method draft_model \
  --prompt-set shared-short --max-model-len 80 --output-length 64 \
  --memory-fraction 0.22 --num-gpu-blocks 6 --require-preemption \
  --output-dir "$DSPARK_RUN/preemption-short"
```

The lifecycle tool fixes model length 256, scheduled-token budget 32 and output
length 24. It compares separate target-only and speculative workers, exercises
cancellation/ID reuse and requires real proposals, verification and acceptance.
The short preemption fixture uses four five-token prompts, 64 output tokens,
C=4 and T=64. On the source machine it caused seven actual scheduler preemptions
and all four streams matched. `--num-gpu-blocks` constrains scheduler-visible
capacity; it does not shrink the physically budgeted Metal pool. Require the
preemption counter, rather than assuming any small-looking block count preempts.

### 4. Memory recovery, full capacity and full-checkpoint precision

```bash
python -m tools.dspark_memory_check --target "$DSPARK_TARGET" --draft "$DSPARK_DRAFT" \
  --rounds 16 --warmup-rounds 4 --width 7 --max-model-len 1024 --output-length 64 \
  --memory-fraction 0.22 --faults --capacity-probe --output-dir "$DSPARK_RUN/memory-final"
python -m tools.dspark_precision_check --target "$DSPARK_TARGET" --draft "$DSPARK_DRAFT" \
  --deepspec-checkout "$DSPARK_REFERENCE" --reference-python-path "$DSPARK_REFERENCE_DEPS" \
  --baseline-result "$DSPARK_RUN/memory-final/k0.result.json" \
  --output-dir "$DSPARK_RUN/precision"
```

The memory tool fixes C=4, T=64 and target prefix caching off. `--faults` requires
at least 16 rounds and covers exhausted byte/slot admission and an injected
allocation failure after a real context write. The capacity probe uses synthetic
sinusoidal target features to fill all contexts while the real target KV pool is
resident. It proves allocation capacity, not full-context arrival throughput.

The precision checker needs the **memory checker's** `k0.result.json`, containing
prompt IDs and complete baseline tokens. A committed summary or lifecycle result
has a different schema. Capture, official Torch BF16, MLX BF16 and affine-4 stages
run in separate processes. Previously all three draft variants accepted 86/112
positions across 16 fixtures; the predefined BF16 normalized-L2 gate was 0.05
and the quantized/BF16 accepted-prefix-total ratio gate was 0.8. Do not tighten or
relax these after observing a new result. Affine-4 confidence error reached about
0.4535; this is not a calibrated confidence model or held-out performance corpus.

The M4 machine's recommended working set was 22,906,503,168 bytes; fraction 0.22
allowed 5,039,430,696 bytes. The final-source test peaked at 4,730,879,170 active
MLX bytes and retained zero active-memory drift after drain. Four complete
1,024-token contexts occupied 83,886,080 bytes. These are measured M4-machine
numbers, not destination predictions or a hard OS-enforced memory cap.

Fraction 0.22 is a conservative **starting recipe**, relative to the destination's
actual recommended Metal working set, not physical RAM. Record its resulting
byte budget, active plus allocator-cache usage, RSS, memory pressure and swap.
The M4 negative startup control used fraction 0.12; the same fraction may fit on
M5 Max and is not a portable expected failure. Construct a deliberately
insufficient byte allowance from the destination's actual startup estimate for
a new negative control. Historical M2 fraction-0.12 probes predated M3 accounting.

### 5. Reproduce both unresolved extended parity failures

First replay the committed prefixes without loading the drafter:

```bash
python -m tools.dspark_target_replay --target "$DSPARK_TARGET" \
  --failure docs/design/dspark-target-replay-results.json --output "$DSPARK_RUN/replay-long.json"
python -m tools.dspark_target_replay --target "$DSPARK_TARGET" \
  --failure docs/design/dspark-preemption-replay-results.json --output "$DSPARK_RUN/replay-preemption.json"
```

Then run each full failure reproduction separately, preserving the nonzero exit
and output if it recurs:

```bash
python -m tools.dspark_memory_check --target "$DSPARK_TARGET" --draft "$DSPARK_DRAFT" \
  --rounds 1 --warmup-rounds 0 --width 7 --prompt-set ragged \
  --max-model-len 1024 --output-length 900 --memory-fraction 0.22 \
  --output-dir "$DSPARK_RUN/long-output"
```

```bash
python -m tools.dspark_memory_check --target "$DSPARK_TARGET" --draft "$DSPARK_DRAFT" \
  --rounds 1 --warmup-rounds 0 --width 2 --method draft_model --prompt-set shared \
  --max-model-len 384 --output-length 256 --num-gpu-blocks 25 --require-preemption \
  --memory-fraction 0.22 --output-dir "$DSPARK_RUN/preemption-extended"
```

| Unresolved case | Existing witness and implication |
| --- | --- |
| 900-token output, K=7 | Three of four requests diverged at output positions 172, 239 and 94 (zero-based); truncated prompt lengths were 5/106/116/116. |
| First long-output witness | Verification chose token 60650 (logit 20.0) over baseline token 279 (19.75). With the identical prefix and no drafter, native target chunk size 1 chose 60650, while size 4 chose 279. |
| Extended preemption, K=2 alias | Six actual preemptions; all four speculative streams differed. The target-only baseline itself yielded three distinct streams from four identical 106-token prompts. Physical context/drain/resource checks still passed. |

Native target chunk sensitivity explains a concrete mechanism to investigate;
it does not prove every mismatch has that cause or rule out cache/layout errors.
Not every stored witness reproduced a changed native token across chunk sizes.
On M5 Max, different dispatch may change or hide a witness. A pass on the new
machine does not retroactively fix the M4 machine's failing envelope.

M5 Max outcome (2026-09-10): both cases reproduce with deterministic
divergence sets. Traced logits and the target-stability probe attribute every
divergence to a tie or to a prefix where the target-only engine itself returns
different greedy tokens under different execution shapes; see the
[M4a parity record](dspark-m4-parity.md) for the evidence, the contract and
the gate commands (`--trace-logits`, `dspark_target_stability`,
`dspark_divergence_classify --stability --gate`). The strict checkers still
exit nonzero on these cases; run the gate as the recorded second step.

Without `--diagnose-mismatch`, a token mismatch allows remaining resource/drain
checks to finish, writes `kN.result.failure.json` with expected and actual streams,
and still returns failure. The ordinary result's `tokens` / `token_sha256` refer
to the expected baseline; inspect **`actual_tokens` in the failure file** when
diagnosing output. With `--diagnose-mismatch`, the checker stops at the first
verifier disagreement and writes `kN.result.trace.json`; rerun into a new output
directory if that trace is needed. Early diagnostic exits are not drain proofs.

Record both engines' logits first. `--trace-logits` makes the memory checker
store the top-8 target logits of every sampled row (baseline and speculative)
with the runner's own request identity, absolute position, row count of the
forward and the drafted token; `tools.dspark_divergence_classify` then labels
each first divergence from those records:

```bash
python -m tools.dspark_memory_check --target "$DSPARK_TARGET" --draft "$DSPARK_DRAFT" \
  --rounds 1 --warmup-rounds 0 --width 7 --prompt-set ragged --trace-logits \
  --max-model-len 1024 --output-length 900 --memory-fraction 0.22 --output-dir "$DSPARK_RUN/long-output-trace"
python -m tools.dspark_target_replay --target "$DSPARK_TARGET" \
  --failure "$DSPARK_RUN/long-output-trace/k7.result.failure.json" --output "$DSPARK_RUN/long-output-trace/replay.json"
python -m tools.dspark_divergence_classify --run-dir "$DSPARK_RUN/long-output-trace" --width 7 \
  --replay "$DSPARK_RUN/long-output-trace/replay.json" --output "$DSPARK_RUN/long-output-trace/classified.json"
```

A `tie` means both engines rank the two tokens first and second within two
bfloat16 ULPs of each other (the criterion upstream's draft-model e2e adopted
in #524): summation order between one-row and multi-row forwards split it and
no state is wrong. An `engine-disagreement` means the engines assigned
materially different logits to the same committed prefix, which is invalid
state in at least one of them; the native replay names the engine that agrees
with mlx-lm. Identical prompts are also compared within one engine.

Next investigate the first differing target logits at an identical committed
prefix: token/position/slot maps, selected-logit row ownership, target KV contents
and valid lengths, masks, prefill/decode/verify/recompute chunk shape, quantized
matmul and paged attention dispatch. Compare capture enabled/disabled and draft
loaded/unloaded. Use reduced deterministic fixtures to separate arithmetic order
from invalid state, then add a regression at the owning layer. Do not waive
exact-token comparison, choose alternate expected tokens, or shorten the failing
workload to close M4. A proposed numerical policy change needs a separately
reviewed contract and evidence before any qualification claim.

### 6. Sustained bounded memory run

After the shorter checks, repeat the original sustained allocation/recovery run:

```bash
python -m tools.dspark_memory_check --target "$DSPARK_TARGET" --draft "$DSPARK_DRAFT" \
  --rounds 128 --warmup-rounds 4 --width 7 --max-model-len 1024 --output-length 64 \
  --memory-fraction 0.22 --faults --capacity-probe --output-dir "$DSPARK_RUN/memory-sustained"
```

The earlier run completed 528 batch requests including warmups, plus 16
cancellations and 16 ID reuses, verified 35,927 draft tokens and accepted 28,274.
All compared completed streams matched; retained active memory had zero range.
It lasted about 14.6 minutes with instrumentation. This is neither a controlled
speed benchmark nor the required M7 production soak. Preserve positive draft
work counters: target-only fallback throughout cannot qualify acceleration.

### 7. M4 and M5 gates on the real pair (M5 Max)

The M4b, M4c, M4d and M5 gates run against the same pinned pair after steps 1-6.
The admission checker records, per request, the steps it held a complete
context and the steps it was drafted; the serving harness launches real
`vllm serve` processes and judges every divergence by the K=0 server's own
logprobs; the step profiler attributes per-step costs in process and the
performance benchmark runs the paired streamed protocol (run it alone on an
idle machine; it takes about twenty minutes for the default matrix):

```bash
python -m tools.dspark_admission_check --target "$DSPARK_TARGET" --draft "$DSPARK_DRAFT" \
  --requests 48 --max-num-seqs 48 --width 7 --context-cap 32 --output "$DSPARK_RUN/admission/cap32-c48.json"
python -m tools.dspark_admission_check --target "$DSPARK_TARGET" --draft "$DSPARK_DRAFT" \
  --requests 48 --max-num-seqs 48 --width 7 --context-cap 8 --output "$DSPARK_RUN/admission/cap8-c48.json"
python -m tools.dspark_serving_check --target "$DSPARK_TARGET" --draft "$DSPARK_DRAFT" \
  --widths 1,2,4,7 --prefix-widths 2,7 --output-dir "$DSPARK_RUN/serving"
python -m tools.dspark_step_profile --target "$DSPARK_TARGET" --draft "$DSPARK_DRAFT" \
  --widths 0,1,2,4,7 --concurrency 1 --output-dir "$DSPARK_RUN/profile-c1"
python -m tools.dspark_perf_bench --target "$DSPARK_TARGET" --draft "$DSPARK_DRAFT" --repo . \
  --widths 1,2,4,7 --buckets 128x128,1024x128,128x512 --concurrency 1,4 --repetitions 5 \
  --async-reference --output-dir "$DSPARK_RUN/bench"
python -m tools.dspark_stochastic_check --target "$DSPARK_TARGET" --draft "$DSPARK_DRAFT" \
  --width 7 --samples 8000 --control-max-num-seqs 16 --output-dir "$DSPARK_RUN/stochastic"
```

The serving harness exits nonzero on any failure and prints `GATE PASS`
otherwise; `summary.json` lists every scenario's verdict per server and each
`kN.json` keeps the tokens, the tie probes and the `/metrics` deltas. The
benchmark writes `results.json` with every repetition, the paired summaries
and the gate verdict per cell; the profiler writes one result per width and a
`summary.json`. Results and the qualification criteria are in the progress
record (M4b, M4c, M4d, M5). The stochastic check compares the speculative and
target-only engines' sampled distributions position by position, runs the
mixed greedy/stochastic/logprobs/penalty/short-budget/stop-token batch and the
seeded-reproducibility repeat, and with `--control-max-num-seqs` also reports
the target-only-versus-target-only statistics at another batch size as the
numerics floor. The tools bind the default `VLLM_USE_V2_MODEL_RUNNER=0`,
source-built kernels and offline Hugging Face access themselves.

## Remaining implementation and acceptance plan

Follow the existing architecture and small staged PR pattern: extend the current
platform, runner, cache and verifier owners, add a focused regression, then attach
real-model evidence and update progress. Existing scaffolding for ragged caps,
cleanup, request eligibility and memory admission must be extended and qualified,
not counted as a completed M4 serving feature by inspection alone.

| Next part | Concrete implementation and completion evidence |
| --- | --- |
| M4a: target/verification parity | Done on M5 Max under the recorded contract: both witnesses are ties or target-unstable prefixes, no cache/layout defect found, DSpark behaved correctly in every trace. Remaining: repeat the natural-workload gate on the M4 machine when available; the paged path's higher junk-basin frequency at absolute 207 is an open numerics observation, not a gate. |
| M4b: fixed-K admission | Done on M5 Max: configurable context cap (`VLLM_METAL_DSPARK_MAX_CONTEXTS`) budgeted by the planner, per-step draft cap (`VLLM_METAL_DSPARK_MAX_DRAFTS_PER_STEP`) with least-recently-drafted rotation, per-row caps proven against independent rows, and `tools/dspark_admission_check.py` real-engine evidence (see the progress record). Remaining: repeat on the M4 machine when available. |
| M4c: serving semantics | Done on M5 Max: `tools/dspark_serving_check.py` drives real `vllm serve` processes (multiprocess engine core) for K=0 and K=1/2/4/7, with and without prefix caching, through output limits 1/2/31/128, natural EOS, stop strings, the platform's `min_tokens` rejection, streaming, long prompts, staggered arrivals, a mid-stream disconnect, `logprobs` and sampled requests, and prefix repeats; parity is judged by the M4a tie rule from the K=0 server's own logprobs; positive draft work and idle metrics are asserted. Results in the progress record. Remaining: repeat on the M4 machine when available. |
| M4d: fixed-K performance | Done on M5 Max: `tools/dspark_step_profile.py` attributes per-step draft, verify, context and host costs in process; `tools/dspark_perf_bench.py` runs the paired streamed HTTP protocol (three buckets, C=1/4, five alternating repetitions, bootstrap intervals, SLO goodput, asynchronous target-only reference). Gate met at C=1 for K=1/2/4/7 on 128-token inputs and K=2/4 on 1,024-token inputs; not met at C=4 (progress record). Remaining: repeat on the M4 machine; the multi-row target verification cost is the M7 profiling target and the reason M6 must bypass at higher concurrency. |
| M5: stochastic verification | Done on M5 Max: `vllm_metal/v1/dspark/sampling.py` owns the exact float32 proposal distributions (temperature, top-k, top-p with the Metal sampler's mask semantics), inverse-CDF sampling with explicit finite-precision rules, the residual and bonus draws and the per-request proposal/acceptance/target streams; the proposer keeps a `DSparkProposal` record per scheduled draft and the controller's `verify` dispatches greedy and stochastic requests per row. Tests: enumerated two-position tiny-vocabulary oracle, a 20,000-draw chi-square gate with a negative control, zero-support and truncated proposals, stream isolation, scheduler-clipped drafts, fail-closed records; `tools/dspark_stochastic_check.py` compares the speculative and target-only engines' output distributions and runs a mixed greedy/stochastic/logprobs/penalty/short-budget batch on the real pair (progress record). Penalties, logprobs, allowed/bad tokens, `min_p`, `logit_bias` and structured output remain target-only with an observable reason. Remaining: repeat on the M4 machine. |
| M6: calibrated planning | Implement confidence/Markov semantics, recording with correct censored survival labels, per-position STS fit/apply, disjoint calibration/evaluation data and recipe-matched manifests. Report ECE, Brier, reliability and uncertainty. Fit separate measured draft/verify/host cost curves; make causal admission and K=0 bypass decisions before affected candidates are sampled. Test against a pure planner oracle, including early stopping and history cleanup. |
| M7: production hardening | Profile before kernel optimization; qualify actual HTTP mixed arrivals, streaming, disconnects/timeouts, prefix and memory pressure. Run **at least one hour AND at least 10,000 completed/cancelled requests**, with memory/resource/fallback and latency evidence. Validate packaged deployment, not just in-process instrumented checkers. |
| M8: more standalone pairs | After 4B correctness, evaluate Qwen3-8B, then 14B, then Gemma4-12B where capacity and backend support permit. Pin target/tokenizer/draft, implement family-specific adapters, repeat every relevant parity/precision/lifecycle/stochastic/calibration/performance gate per serving recipe. |

The full [specification](dspark.md) remains authoritative for detailed gates and
algorithmic constraints. M6's planner must optimize useful emitted tokens over
**total** step cost, including the decision whether to draft at all. Choosing
K=0 after running the backbone does not recover that cost. Never use future
sampled descendant tokens to retrospectively choose a stochastic prefix length.

For Gemma4, tiny draft parity is insufficient: target embedding scaling, global
heads, partial RoPE, sandwich norms, scalar/softcap behavior, sliding versus full
attention and KV sharing need model-specific preservation. Do not enable a new
family because its name or vocabulary resembles a tested family.

### Hardware-bounded experiment strategy

Use the 48 GB machine to improve coverage, not to infer unlimited concurrency.
Start with the pinned 4B pair and the small gates above. For each larger pair,
estimate target/draft resident weights, conversion peak, target KV, complete
context/capture/workspace and allocator reserve before downloading/loading.
Start a candidate **quantized target with max context 2,048 and C=1**, then expand
to C=2/4/8 only after measured capacity and correctness pass. Target quantization
is part of the qualified pair, not an interchangeable performance switch.

Thirty-two 8,192-token draft contexts alone require approximately 5 GiB before
padded copies or other workspaces. Do not start with 8,192/C=32. Keep OS and other
process headroom, monitor memory pressure and swap, and stop an unsafe expansion
rather than relying on SSD swap as serving capacity. The 2 TB SSD is useful for
pinned snapshots, numerical fixtures and profiles; it does not increase RAM.

A performance claim needs predefined workload/SLOs, at least five independent
measured repetitions with alternating baseline/speculative order after warmup,
and paired uncertainty. The proposed spec gate is at least 10% median benefit on
a declared workload with a paired 95% confidence interval excluding no benefit;
adaptive goodput and p95 should stay within 5% of target-only, or invoke an
explicit bypass/narrower support envelope. Inputs 128/1,024/4,096, outputs
32/128/512 and C=1/4/16 are workload buckets **as capacity allows**, not an
instruction to run every combination. Report skipped cells and their reason.

Use identical weights, tokenizer, requests and sampling for controlled
comparisons, and compare deployability with the best supported ordinary
target-only configuration, including asynchronous scheduling where supported.
Do not report only a deliberately slowed synchronous baseline or only favorable
short prompts. Current offline checkers expose neither a general concurrency
sweep nor a complete HTTP benchmark; implement the M4 harness first.

Integrated V4 uses a different integrated target/draft architecture with mHC,
MoE and sparse/compressed attention. It is not a loader extension for these
standalone weights. Keep it deferred on both available machines. For any valuable
experiment that does not fit or lacks a backend, record the pinned pair, estimated
bytes, missing capability, exact gate it would address and required future
hardware. Do not mark an unrun experiment passed or silently remove it from the
handoff. Preserve that queue after completing all feasible local experiments.

## Results, source transfer and reproducibility

Every new result set should record commit and source diff, machine/OS/SDK/kernel
path, dependencies, model/tokenizer/config hashes, quantization, model and batch
limits, budget and measured peaks, prompts or dataset revision/split, seeds,
sampling and stopping options, warmups/repetitions, exit status and full token
streams/counters. Separate numerical fixtures from calibration and held-out
performance data. Keep failed results beside successful ones. Use fresh run IDs;
never replace the immutable M3 summaries with a new machine's results.

All required source and concise evidence are in Git. For fuller diagnosis, a
separate **`dspark-evidence-20260910.tar.gz`** transfer archive was prepared from
the source machine's original JSON, logs and NPZ fixtures. It includes M0–M2,
M3 and initial tiny-oracle/audit output, with a per-file `MANIFEST.json` and README.
The adjacent `SHA256SUMS.txt` validates the compressed archive. This archive is a
local transfer artifact, not committed model data or a GitHub attachment. It
contains 227 evidence files and is 447,088,510 bytes (about 447 MB). Its SHA-256 is:

```text
5fc086b229f16858148a4506200db4d96354e3e1a94a73f4e20f9bca43519475
```

On the source machine its directory is:

```text
/Users/chuyuewang/.codex/automations/daily-vllm-metal-contribution-loop/dspark-handoff-20260910/
```

Copy the archive and its checksum file to the destination evidence directory,
then verify before extracting:

```bash
mkdir -p "$DSPARK_DATA/evidence"
cd "$DSPARK_DATA/evidence"
shasum -a 256 -c SHA256SUMS.txt
tar -xzf dspark-evidence-20260910.tar.gz
python - <<'PY'
import hashlib, json
from pathlib import Path
root = Path("dspark-evidence-20260910")
manifest = json.loads((root / "MANIFEST.json").read_text())
for item in manifest["files"]:
    path = root / item["path"]
    assert path.stat().st_size == item["bytes"], path
    with path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"], path
print("Verified evidence files:", len(manifest["files"]))
PY
cd "$HOME/Development/vllm-metal"
```

Archive directories preserve failures and intermediate runs as well as final
results. Consult the M3 qualification document when deciding what passed:

| Archive location | Interpretation |
| --- | --- |
| `m0-m2/` | Contract/capture/lifecycle logs and original result JSON; old memory fractions are historical |
| `m3/soak/` | Earlier 128-round sustained run |
| `m3/final-soak/`, `m3/precision-final/` | Final-source fault/capacity and precision evidence, including raw numerical fixtures |
| `m3/preemption-short/` | Passing short actual-preemption case |
| `m3/preemption-shared/` | Failed extended preemption with resource/drain evidence |
| `m3/capacity/` | Includes long-output diagnostics; consult each worker config rather than treating the directory name as a passing capacity gate |
| `m3/full-capacity/` | Separate full-context allocation probe |
| `m3/preemption/` | Earlier attempt that did not actually preempt; not qualifying preemption evidence |
| `initial-audit/` | Tiny reference and initial audit logs/results |

Archived worker configs and log strings retain original absolute paths. **Regenerate
worker configs with the public checker CLI** using the new model directories;
do not run old `--worker-config` files unchanged. No virtual environments, native
build outputs, authentication helpers, network reports or model checkpoint files
are included. Re-clone the pinned official source and transfer/download model
snapshots separately. Use fresh GitHub/Hugging Face authentication on the new
computer; never copy credentials or machine-specific authentication helpers.

Original source-machine locations, if additional evidence is needed:

```text
Repository: /Users/chuyuewang/Desktop/RESEARCH/Better Agentic Browser/vllm_metal_dspark
M0-M2: /Users/chuyuewang/.codex/automations/daily-vllm-metal-contribution-loop/dspark-implementation-20260909
M3: /Users/chuyuewang/.codex/automations/daily-vllm-metal-contribution-loop/dspark-m3-20260909
Initial research: /Users/chuyuewang/.codex/automations/daily-vllm-metal-contribution-loop/dspark-research-20260909
Target snapshot: /Users/chuyuewang/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25
Draft snapshot: /Users/chuyuewang/.cache/huggingface/hub/models--deepseek-ai--dspark_qwen3_4b_block7/snapshots/3457dff1417cb84927f6098a5fcb7cee85c934b7
```

## Contribution workflow after migration

Use focused branches from the updated `Dspark` base, for example
`dspark-m4-target-parity` or `dspark-m4-serving-validation`. Never include `codex`
or `agent` in new branch names. Preserve the contributor's identity and required
DCO sign-off; do not add an AI/agent coauthor trailer. The source checkout used
**Chuyue Wang <stevenwang0805@outlook.com>**. Apply this repository's normal
contribution and native-kernel style; SGLang-specific paths/skills are not the
vLLM Metal integration workflow.

For each coherent part: implement, inspect the diff, run relevant regressions
and real-model gates, update progress/evidence, commit with `git commit -s`, push
the descriptive branch, and open a PR into **`mhdimo/vllm-metal:Dspark`**. State
the concrete problem, resulting behavior, exact tested head and unresolved gates.
Obtain appropriate collaborator review for behavior/contract decisions. Merge
only the tested head with the normal merge workflow; verify the resulting tree
and fast-forward the local integration branch. Do not bundle the entire remaining
roadmap into one unreviewable PR or claim upstream acceptance from a fork merge.

At the handoff baseline, the fork's CI workflow only targets `main`; the six
`Dspark` implementation PRs had no hosted CI runs. Local validation is explicit
evidence, not a hosted green check or independent maintainer approval. Retain
logs and investigate required checks/branch policy on each future PR; do not
bypass protections to meet a milestone label. Preserve all deferred experiments
and the exact scope of any remaining limitation in each progress update.

## First destination session

1. Clone `Dspark`, verify the baseline ancestry, read this handoff and the M3
   failures, and install the pinned fresh arm64 environment.
2. Verify the two model snapshots and the transferred evidence; record hardware,
   SDK, kernel path and dependency state.
3. Run the ordered small correctness/resource checks, then both stored-prefix
   replays and the full unresolved failure commands. Record changed M5 Max
   behavior without reclassifying the old failures.
4. Begin M4a with the first-divergence evidence. Keep M5 stochastic work, M6
   planning and larger model expansion behind their prerequisite gates.
5. Carry forward the remaining experiments and update the progress record after
   every tested, reviewed implementation part.
