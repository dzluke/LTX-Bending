from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, TypedDict

import torch
from bending_functions import add_scalar, invert, multiply_scalar, reflect, rotate
from ltx_core.types import LatentState
from ltx_pipelines import DistilledPipeline
from ltx_pipelines.utils.bending import make_network_bending_loop
from ltx_pipelines.utils.media_io import encode_video
from main import load_config, save_frames_and_collect, validate_config

BendFunctionName = Literal[
    "add_scalar",
    "multiply_scalar",
    "invert",
    "reflect",
    "rotate",
]


class BendSpec(TypedDict):
    name: str
    function: BendFunctionName
    params: dict[str, float | int | bool]
    steps: list[int]


OUTPUT_ROOT = Path("outputs/batch")
OVERWRITE_EXISTING = False

# Configure your batch sweep here in Python code.
EXPERIMENTS: list[BendSpec] = [
    {"name": "add_2_steps4_8", "function": "add_scalar", "params": {"value": 2.0}, "steps": [4, 8]},
    {"name": "mul_5_step4", "function": "multiply_scalar", "params": {"factor": 5.0}, "steps": [4]},
    {"name": "invert_step6", "function": "invert", "params": {}, "steps": [6]},
    {"name": "reflect_w_step7", "function": "reflect", "params": {"dim": -1}, "steps": [7]},
    {"name": "rotate90_step8", "function": "rotate", "params": {"k": 1}, "steps": [8]},
]


def apply_bend(latent: torch.Tensor, fn_name: BendFunctionName, params: dict[str, float | int | bool]) -> torch.Tensor:
    if fn_name == "add_scalar":
        value = float(params.get("value", 0.0))
        return add_scalar(latent, value=value)
    if fn_name == "multiply_scalar":
        factor = float(params.get("factor", 1.0))
        return multiply_scalar(latent, factor=factor)
    if fn_name == "invert":
        return invert(latent)
    if fn_name == "reflect":
        dim = int(params.get("dim", -1))
        return reflect(latent, dim=dim)
    if fn_name == "rotate":
        k = int(params.get("k", 1))
        return rotate(latent, k=k)
    raise ValueError(f"Unsupported bend function: {fn_name}")


def build_bending(spec: BendSpec):
    target_steps = set(spec["steps"])

    def bending(video_state: LatentState, step_idx: int) -> LatentState:
        if step_idx not in target_steps:
            return video_state
        print(
            f"  Applying '{spec['function']}' at step {step_idx} with params={spec['params']}",
        )
        bent = apply_bend(video_state.latent, spec["function"], spec["params"])
        return replace(video_state, latent=bent)

    return bending


def safe_run_name(name: str) -> str:
    keep = "-_."
    return "".join(c if c.isalnum() or c in keep else "_" for c in name)


@torch.inference_mode()
def main() -> None:
    base_cfg = load_config()
    validate_config(base_cfg)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    run_manifest: list[dict[str, Any]] = []

    for run_idx, spec in enumerate(EXPERIMENTS, start=1):
        run_name = f"{run_idx:03d}_{safe_run_name(spec['name'])}"
        run_dir = OUTPUT_ROOT / run_name
        frames_dir = run_dir / "frames"
        video_path = run_dir / "video.mp4"

        if run_dir.exists() and not OVERWRITE_EXISTING:
            print(f"[{run_idx}/{len(EXPERIMENTS)}] Skipping existing run: {run_name}")
            continue

        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{run_idx}/{len(EXPERIMENTS)}] Running: {run_name}")
        denoising_loop = make_network_bending_loop(build_bending(spec))
        pipeline = DistilledPipeline(
            distilled_checkpoint_path=base_cfg.distilled_checkpoint_path,
            gemma_root=base_cfg.gemma_root,
            spatial_upsampler_path=base_cfg.spatial_upsampler_path,
            loras=base_cfg.loras,
            quantization=base_cfg.quantization_policy(),
            torch_compile=base_cfg.torch_compile,
        )

        video_chunks, audio = pipeline(
            prompt=base_cfg.prompt,
            seed=base_cfg.seed,
            height=base_cfg.height,
            width=base_cfg.width,
            num_frames=base_cfg.num_frames,
            frame_rate=base_cfg.frame_rate,
            images=[],
            tiling_config=None,
            enhance_prompt=base_cfg.enhance_prompt,
            streaming_prefetch_count=1,
            denoising_loop=denoising_loop,
        )

        video_tensor = save_frames_and_collect(video_chunks, frames_dir)
        encode_video(
            video=video_tensor,
            fps=int(base_cfg.frame_rate),
            audio=audio,
            output_path=str(video_path),
            video_chunks_number=1,
        )

        run_info = {
            "run": run_name,
            "function": spec["function"],
            "params": spec["params"],
            "steps": spec["steps"],
            "frames_dir": str(frames_dir),
            "video_path": str(video_path),
        }
        run_manifest.append(run_info)

        with open(run_dir / "run.json", "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2)

    with open(OUTPUT_ROOT / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    print(f"Completed {len(run_manifest)} run(s). Manifest: {(OUTPUT_ROOT / 'manifest.json').resolve()}")


if __name__ == "__main__":
    main()
