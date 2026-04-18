from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

import torch
from bending_functions import (
    add_scalar,
    exponential,
    invert,
    logarithm,
    multiply_scalar,
    power,
    reflect,
    rotate,
)
from ltx_core.types import LatentState
from ltx_pipelines import DistilledPipeline
from ltx_pipelines.utils.bending import make_network_bending_loop
from ltx_pipelines.utils.media_io import encode_video
from main import load_config, save_frames_and_collect, validate_config

BendFunctionName = Literal[
    "add_scalar",
    "exponential",
    "logarithm",
    "multiply_scalar",
    "power",
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
OVERWRITE_EXISTING = True
SAVE_VIDEOS_ONLY = True

# Configure your batch sweep here in Python code.
EXPERIMENTS: list[BendSpec] = [
    # {"name": "add_2_step4", "function": "add_scalar", "params": {"value": 2.0}, "steps": [4]},
    # {"name": "add_2_steps4_8", "function": "add_scalar", "params": {"value": 2.0}, "steps": [4, 8]},
    # {"name": "mul_5_step2", "function": "multiply_scalar", "params": {"factor": 5.0}, "steps": [2]},
    # {"name": "mul_5_step7", "function": "multiply_scalar", "params": {"factor": 5.0}, "steps": [7]},
    # {"name": "exp_step6", "function": "exponential", "params": {}, "steps": [6]},
    # {"name": "log_bias3_step6", "function": "logarithm", "params": {"eps": 1e-6}, "steps": [6]},
    # # {"name": "pow_1_5_step7", "function": "power", "params": {"exponent": 1.5}, "steps": [7]},
    # {"name": "invert_step6", "function": "invert", "params": {}, "steps": [6]},
    # {"name": "reflect_w_step7", "function": "reflect", "params": {"dim": -1}, "steps": [7]},
    # {"name": "rotate90_step8", "function": "rotate", "params": {"k": 1}, "steps": [8]},
    {"name": "add_-2_step4", "function": "add_scalar", "params": {"value": -2.0}, "steps": [4]},
    {"name": "add_2_steps1", "function": "add_scalar", "params": {"value": 2.0}, "steps": [1]},
    {"name": "mul_10_step2", "function": "multiply_scalar", "params": {"factor": 10.0}, "steps": [2]},
    {"name": "mul_10_step7", "function": "multiply_scalar", "params": {"factor": 10.0}, "steps": [7]},
    {"name": "mul_0p5_step7", "function": "multiply_scalar", "params": {"factor": 0.5}, "steps": [7]},
    {"name": "mul_0p5_step7", "function": "multiply_scalar", "params": {"factor": 0.5}, "steps": [11]},
    {"name": "exp_step2", "function": "exponential", "params": {}, "steps": [2]},
    {"name": "log_bias3_step2", "function": "logarithm", "params": {"eps": 1e-6}, "steps": [2]},
    {"name": "invert_step2", "function": "invert", "params": {}, "steps": [2]},
    {"name": "reflect_w_step2", "function": "reflect", "params": {"dim": -1}, "steps": [2]},
    # {"name": "rotate90_step8", "function": "rotate", "params": {"k": 1}, "steps": [8]},
]


def apply_bend(latent: torch.Tensor, fn_name: BendFunctionName, params: dict[str, float | int | bool]) -> torch.Tensor:
    if fn_name == "add_scalar":
        value = float(params.get("value", 0.0))
        return add_scalar(latent, value=value)
    if fn_name == "exponential":
        return exponential(latent)
    if fn_name == "logarithm":
        eps = float(params.get("eps", 1e-8))
        return logarithm(latent, eps=eps)
    if fn_name == "multiply_scalar":
        factor = float(params.get("factor", 1.0))
        return multiply_scalar(latent, factor=factor)
    if fn_name == "power":
        exponent = float(params.get("exponent", 1.0))
        return power(latent, exponent=exponent)
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


def create_batch_run_dir(root: Path) -> Path:
    # Create a unique folder per script execution so outputs never overwrite prior runs.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"run_{timestamp}"
    run_dir = root / base_name
    suffix = 1

    while run_dir.exists():
        run_dir = root / f"{base_name}_{suffix:02d}"
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


@torch.inference_mode()
def main() -> None:
    base_cfg = load_config()
    validate_config(base_cfg)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    batch_run_dir = create_batch_run_dir(OUTPUT_ROOT)

    # Load model components once, then swap the denoising loop for each run.
    pipeline = DistilledPipeline(
        distilled_checkpoint_path=base_cfg.distilled_checkpoint_path,
        gemma_root=base_cfg.gemma_root,
        spatial_upsampler_path=base_cfg.spatial_upsampler_path,
        loras=base_cfg.loras,
        quantization=base_cfg.quantization_policy(),
        torch_compile=base_cfg.torch_compile,
    )

    run_manifest: list[dict[str, Any]] = []

    for run_idx, spec in enumerate(EXPERIMENTS, start=1):
        run_name = f"{run_idx:03d}_{safe_run_name(spec['name'])}"
        if SAVE_VIDEOS_ONLY:
            video_path = batch_run_dir / f"{run_name}.mp4"
            if video_path.exists() and not OVERWRITE_EXISTING:
                print(f"[{run_idx}/{len(EXPERIMENTS)}] Skipping existing video: {video_path.name}")
                continue
        else:
            run_dir = batch_run_dir / run_name
            frames_dir = run_dir / "frames"
            video_path = run_dir / "video.mp4"

            if run_dir.exists() and not OVERWRITE_EXISTING:
                print(f"[{run_idx}/{len(EXPERIMENTS)}] Skipping existing run: {run_name}")
                continue

            run_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{run_idx}/{len(EXPERIMENTS)}] Running: {run_name}")

        denoising_loop = make_network_bending_loop(build_bending(spec))
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
            denoising_loop=denoising_loop
        )

        if SAVE_VIDEOS_ONLY:
            chunks = tuple(video_chunks)
            if not chunks:
                raise RuntimeError(f"Pipeline returned no video chunks for run: {run_name}")
            video_tensor = torch.cat(chunks, dim=2)
        else:
            video_tensor = save_frames_and_collect(video_chunks, frames_dir)

        encode_video(
            video=video_tensor,
            fps=int(base_cfg.frame_rate),
            audio=audio,
            output_path=str(video_path),
            video_chunks_number=1,
        )

        if not SAVE_VIDEOS_ONLY:
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

    if SAVE_VIDEOS_ONLY:
        print(f"Completed video-only batch in: {batch_run_dir.resolve()}")
    else:
        manifest_path = batch_run_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(run_manifest, f, indent=2)

        print(f"Completed {len(run_manifest)} run(s). Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
