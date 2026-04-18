from __future__ import annotations

import json
import math
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


class BendSpec(TypedDict, total=False):
    name: str
    function: BendFunctionName
    params: dict[str, float | int | bool]
    steps: list[int]
    # Per-parameter keypoints over normalized video progress t in [0, 1].
    # Example: {"value": [(0.0, 0.0), (1.0, 5.0)]}
    param_schedule: dict[str, list[tuple[float, float]]]
    # Per-parameter piecewise constant segments over t in [0, 1].
    # Example: {"factor": [(0.0, 0.5, 1.0), (0.5, 1.0, 5.0)]}
    param_segments: dict[str, list[tuple[float, float, float]]]


OUTPUT_ROOT = Path("outputs/batch")
OVERWRITE_EXISTING = True
SAVE_VIDEOS_ONLY = True

# Configure your batch sweep here in Python code.
EXPERIMENTS: list[BendSpec] = [
    # {"name": "add_-2_step4", "function": "add_scalar", "params": {"value": -2.0}, "steps": [4]},
    # {"name": "add_2_steps1", "function": "add_scalar", "params": {"value": 2.0}, "steps": [1]},
    # {"name": "mul_10_step2", "function": "multiply_scalar", "params": {"factor": 10.0}, "steps": [2]},
    # {"name": "mul_10_step7", "function": "multiply_scalar", "params": {"factor": 10.0}, "steps": [7]},
    # {"name": "mul_0p5_step7", "function": "multiply_scalar", "params": {"factor": 0.5}, "steps": [7]},
    # {"name": "mul_0p5_step11", "function": "multiply_scalar", "params": {"factor": 0.5}, "steps": [11]},
    # {"name": "exp_step2", "function": "exponential", "params": {}, "steps": [2]},
    # {"name": "log_bias3_step2", "function": "logarithm", "params": {"eps": 1e-6}, "steps": [2]},
    # {"name": "invert_step2", "function": "invert", "params": {}, "steps": [2]},
    # {"name": "reflect_w_step2", "function": "reflect", "params": {"dim": -1}, "steps": [2]},
    {
        "name": "add_ramp_0_to_5_step7",
        "function": "add_scalar",
        "params": {"value": 0.0},
        "steps": [7],
        "param_schedule": {"value": [(0.0, 0.0), (1.0, 5.0)]},
    },
    {
        "name": "add_ramp_0_to_5_step2",
        "function": "add_scalar",
        "params": {"value": 0.0},
        "steps": [2],
        "param_schedule": {"value": [(0.0, 0.0), (1.0, 5.0)]},
    },
    # {
    #     "name": "mul_segments_step7",
    #     "function": "multiply_scalar",
    #     "params": {"factor": 1.0},
    #     "steps": [7],
    #     "param_segments": {"factor": [(0.0, 0.5, 1.0), (0.5, 1.0, 5.0)]},
    # },
]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalize_time_progress(video_state: LatentState) -> torch.Tensor:
    if video_state.positions is None:
        raise ValueError("video_state.positions is required for param scheduling")

    positions = video_state.positions
    if positions.ndim == 5:
        time_axis = positions[0, 0, :, 0, 0]
    elif positions.ndim == 4:
        # Common layout in this repo: (B, 3, tokens, 2) with [start, end) bounds.
        time_axis = positions[0, 0, :, 0]
    elif positions.ndim == 3:
        time_axis = positions[0, 0, :]
    elif positions.ndim == 2:
        time_axis = positions[0, :]
    else:
        raise ValueError(f"Unsupported positions shape for scheduling: {tuple(positions.shape)}")

    time_axis = time_axis.to(video_state.latent.device, dtype=torch.float32)
    t_min = torch.min(time_axis)
    t_max = torch.max(time_axis)
    if torch.isclose(t_min, t_max):
        return torch.zeros_like(time_axis)
    return (time_axis - t_min) / (t_max - t_min)


def _parameter_vector_from_keypoints(
    t: torch.Tensor,
    keypoints: list[tuple[float, float]],
) -> torch.Tensor:
    if len(keypoints) == 0:
        raise ValueError("param_schedule keypoints cannot be empty")

    sorted_points = sorted((float(tp), float(v)) for tp, v in keypoints)
    if any((not math.isfinite(tp) or not math.isfinite(v)) for tp, v in sorted_points):
        raise ValueError("param_schedule contains non-finite values")

    if len(sorted_points) == 1:
        return torch.full_like(t, sorted_points[0][1])

    x = torch.tensor([tp for tp, _ in sorted_points], dtype=torch.float32, device=t.device)
    y = torch.tensor([v for _, v in sorted_points], dtype=torch.float32, device=t.device)

    idx = torch.searchsorted(x, t, right=True)
    idx = idx.clamp(min=1, max=len(sorted_points) - 1)

    x0 = x[idx - 1]
    x1 = x[idx]
    y0 = y[idx - 1]
    y1 = y[idx]

    denom = (x1 - x0).clamp_min(1e-8)
    alpha = ((t - x0) / denom).clamp(0.0, 1.0)
    out = y0 + alpha * (y1 - y0)

    out = torch.where(t <= x[0], y[0], out)
    out = torch.where(t >= x[-1], y[-1], out)
    return out


def _parameter_vector_from_segments(
    t: torch.Tensor,
    segments: list[tuple[float, float, float]],
    default_value: float,
) -> torch.Tensor:
    if len(segments) == 0:
        raise ValueError("param_segments cannot be empty")

    out = torch.full_like(t, float(default_value))
    for start_t, end_t, value in segments:
        if not math.isfinite(float(value)):
            raise ValueError("param_segments contains non-finite values")
        start = _clamp01(float(start_t))
        end = _clamp01(float(end_t))
        if end <= start:
            continue

        mask = (t >= start) & (t < end)
        if math.isclose(end, 1.0, abs_tol=1e-6):
            mask = mask | (t == 1.0)
        out = torch.where(mask, torch.tensor(float(value), device=t.device), out)
    return out


def _resolve_parameter(
    video_state: LatentState,
    spec: BendSpec,
    param_name: str,
    default_value: float,
) -> float | torch.Tensor:
    schedule = spec.get("param_schedule")
    segments = spec.get("param_segments")

    has_schedule = bool(schedule and param_name in schedule)
    has_segments = bool(segments and param_name in segments)

    if has_schedule and has_segments:
        raise ValueError(
            f"Bend '{spec['name']}' param '{param_name}' defines both param_schedule and param_segments",
        )
    if not has_schedule and not has_segments:
        return default_value

    t = _normalize_time_progress(video_state)
    if has_schedule:
        return _parameter_vector_from_keypoints(t, schedule[param_name])
    return _parameter_vector_from_segments(t, segments[param_name], default_value)


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


def apply_bend_scheduled(
    latent: torch.Tensor,
    fn_name: BendFunctionName,
    params: dict[str, float | int | bool | torch.Tensor],
) -> torch.Tensor:
    def _expand_over_token_axis(values: torch.Tensor) -> torch.Tensor:
        """Broadcast per-token values across the latent tensor regardless of latent rank."""
        token_count = int(values.numel())
        token_axis = None
        for axis in range(1, latent.ndim):
            if latent.shape[axis] == token_count:
                token_axis = axis
                break
        if token_axis is None:
            raise ValueError(
                f"Cannot align scheduled parameter of length {token_count} with latent shape {tuple(latent.shape)}",
            )

        view_shape = [1] * latent.ndim
        view_shape[token_axis] = token_count
        return values.view(*view_shape).to(device=latent.device, dtype=latent.dtype)

    if fn_name == "add_scalar":
        value = params.get("value", 0.0)
        if isinstance(value, torch.Tensor):
            return latent + _expand_over_token_axis(value)
        return add_scalar(latent, value=float(value))
    if fn_name == "multiply_scalar":
        factor = params.get("factor", 1.0)
        if isinstance(factor, torch.Tensor):
            return latent * _expand_over_token_axis(factor)
        return multiply_scalar(latent, factor=float(factor))

    fallback_params: dict[str, float | int | bool] = {}
    for key, value in params.items():
        if isinstance(value, torch.Tensor):
            continue
        fallback_params[key] = value
    return apply_bend(latent, fn_name, fallback_params)


def build_bending(spec: BendSpec):
    target_steps = set(spec["steps"])

    def bending(video_state: LatentState, step_idx: int) -> LatentState:
        if step_idx not in target_steps:
            return video_state

        resolved_params: dict[str, float | int | bool | torch.Tensor] = dict(spec["params"])
        if spec.get("param_schedule") or spec.get("param_segments"):
            for name, value in list(spec["params"].items()):
                if isinstance(value, bool):
                    continue
                resolved_params[name] = _resolve_parameter(video_state, spec, name, float(value))

        print(f"  Applying '{spec['function']}' at step {step_idx}")
        bent = apply_bend_scheduled(video_state.latent, spec["function"], resolved_params)
        return replace(video_state, latent=bent)

    return bending


def safe_run_name(name: str) -> str:
    keep = "-_."
    return "".join(c if c.isalnum() or c in keep else "_" for c in name)


def create_batch_run_dir(root: Path) -> Path:
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

    pipeline = DistilledPipeline(
        distilled_checkpoint_path=base_cfg.distilled_checkpoint_path,
        gemma_root=base_cfg.gemma_root,
        spatial_upsampler_path=base_cfg.spatial_upsampler_path,
        loras=base_cfg.loras,
        quantization=base_cfg.quantization_policy(),
        torch_compile=base_cfg.torch_compile,
    )

    run_manifest: list[dict[str, Any]] = []

    baseline_name = "000_baseline"
    if SAVE_VIDEOS_ONLY:
        baseline_video_path = batch_run_dir / f"{baseline_name}.mp4"
        baseline_frames_dir: Path | None = None
        baseline_run_dir: Path | None = None
        if baseline_video_path.exists() and not OVERWRITE_EXISTING:
            print(f"[baseline] Skipping existing video: {baseline_video_path.name}")
            baseline_video_path = None
    else:
        baseline_run_dir = batch_run_dir / baseline_name
        baseline_frames_dir = baseline_run_dir / "frames"
        baseline_video_path = baseline_run_dir / "video.mp4"
        if baseline_run_dir.exists() and not OVERWRITE_EXISTING:
            print(f"[baseline] Skipping existing run: {baseline_name}")
            baseline_video_path = None
        else:
            baseline_run_dir.mkdir(parents=True, exist_ok=True)

    if baseline_video_path is not None:
        print("[baseline] Running: 000_baseline")
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
            denoising_loop=None,
        )

        if SAVE_VIDEOS_ONLY:
            chunks = tuple(video_chunks)
            if not chunks:
                raise RuntimeError("Pipeline returned no video chunks for run: 000_baseline")
            baseline_video_tensor = torch.cat(chunks, dim=2)
        else:
            assert baseline_frames_dir is not None
            baseline_video_tensor = save_frames_and_collect(video_chunks, baseline_frames_dir)

        encode_video(
            video=baseline_video_tensor,
            fps=int(base_cfg.frame_rate),
            audio=audio,
            output_path=str(baseline_video_path),
            video_chunks_number=1,
        )

        if not SAVE_VIDEOS_ONLY:
            assert baseline_run_dir is not None
            assert baseline_frames_dir is not None
            baseline_info = {
                "run": baseline_name,
                "function": "baseline",
                "params": {},
                "steps": [],
                "frames_dir": str(baseline_frames_dir),
                "video_path": str(baseline_video_path),
            }
            run_manifest.append(baseline_info)

            with open(baseline_run_dir / "run.json", "w", encoding="utf-8") as f:
                json.dump(baseline_info, f, indent=2)

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
            denoising_loop=denoising_loop,
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
