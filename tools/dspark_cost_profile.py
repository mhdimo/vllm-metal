# SPDX-License-Identifier: Apache-2.0
"""Measure DSpark step costs over a (requests, width) grid and write the cost model.

Each cell runs the in-process step profiler (``tools.dspark_step_profile``)
at one drafted width for one concurrency: the target forward with
verification and sampling, the batched draft backbone and the host
bookkeeping are timed per scheduler step with the decode pipeline disabled.
The median costs of every cell form the cost artifact the adaptive planner
interpolates (``vllm_metal.v1.dspark.planner.CostModel``), together with the
model-pair manifest, the machine and MLX identity, the profiled context
length and the per-cell p95 as the uncertainty record. Widths must include
0 (the undrafted step) at every request level. Run it alone on an idle
machine.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from vllm_metal.v1.dspark.calibration import CalibrationManifest
from vllm_metal.v1.dspark.planner import CostModel, CostSample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--requests", default="1,2,4,8,16")
    parser.add_argument("--widths", default="0,1,2,3,4,5,7")
    parser.add_argument("--output-length", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--memory-fraction", default="0.22")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    requests = [int(v) for v in args.requests.split(",") if v]
    widths = [int(v) for v in args.widths.split(",") if v]
    if 0 not in widths:
        parser.error("--widths must include 0")
    env = dict(
        os.environ,
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
        VLLM_METAL_USE_PAGED_ATTENTION="1",
        VLLM_METAL_MEMORY_FRACTION=str(args.memory_fraction),
        VLLM_METAL_BUILD_FROM_SOURCE="1",
        VLLM_METAL_DECODE_PIPELINE="0",
        HF_HUB_OFFLINE="1",
    )
    samples: list[CostSample] = []
    manifest = None
    device = None
    for concurrency in requests:
        for width in widths:
            config = {
                "target": str(args.target.resolve()),
                "draft": str(args.draft.resolve()),
                "width": width,
                "concurrency": concurrency,
                "output_length": args.output_length,
                "max_model_len": args.max_model_len,
                "max_num_batched_tokens": args.max_num_batched_tokens,
            }
            path = args.output_dir / f"k{width}-c{concurrency}.json"
            path.write_text(json.dumps(config, indent=2) + "\n")
            with path.with_suffix(".log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "tools.dspark_step_profile",
                        "--worker-config",
                        str(path),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=3600,
                )
            result = json.loads(path.with_suffix(".result.json").read_text())
            if result.get("manifest"):
                manifest = CalibrationManifest.from_dict(result["manifest"])
            device = result.get("device", device)
            ms = result["ms_per_step"]
            host = sum(
                ms[name].get("p50", 0.0)
                for name in ("ingest_host", "context_eval", "propose_other")
            )
            samples.append(
                CostSample(
                    requests=concurrency,
                    width=width,
                    rows=concurrency * (width + 1),
                    target_ms=ms["target"]["p50"],
                    draft_ms=ms["draft"].get("p50", 0.0),
                    host_ms=host,
                    steps=result["steps"],
                    target_p95_ms=ms["target"]["p95"],
                )
            )
            print(
                f"requests {concurrency} width {width}: target {samples[-1].target_ms:.1f} ms "
                f"(p95 {samples[-1].target_p95_ms:.1f}), draft {samples[-1].draft_ms:.1f} ms, "
                f"host {host:.2f} ms over {result['steps']} steps",
                flush=True,
            )
    assert manifest is not None, (
        "at least one drafted cell is required for the manifest"
    )
    model = CostModel(
        manifest=manifest,
        samples=samples,
        machine={
            "platform": platform.platform(),
            "device": device,
            "memory_fraction": str(args.memory_fraction),
            "decode_pipeline": "disabled",
        },
        # Natural prompts plus half the output: the typical context length
        # the cells were measured at.
        context_tokens=args.output_length // 2 + 64,
    )
    output = args.output_dir / "cost.json"
    output.write_text(model.to_json())
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
