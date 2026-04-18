from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

import torch

from bending_functions import add_scalar, multiply_scalar
from ltx_core.types import LatentState
from ltx_pipelines import DistilledPipeline
from ltx_pipelines.utils.bending import make_network_bending_loop
from ltx_pipelines.utils.media_io import decode_audio_from_file, encode_video
from main import load_config, validate_config

BendFunctionName = Literal["add_scalar", "multiply_scalar"]


class AudioBendSpec(TypedDict):
    name: str
    function: BendFunctionName
    params: dict[str, float | int | bool]
    steps: list[int]
    param_name: str
    param_min: float
    param_max: float
    feature: Literal["rms"]


OUTPUT_ROOT = Path("outputs/audio2video")
OVERWRITE_EXISTING = True


class Audio2VideoConfig:
    """Script-local settings for audio-driven bends.

    This does not replace or override model/runtime settings from config.yaml.
    """

    audio_path: Path = Path("media/audio/saxophone.aiff")
    feature: Literal["rms"] = "rms"
    # Optional preview length for quick tests. Use None to process full audio length.
    quick_test_seconds: float | None = None
    output_root: Path = OUTPUT_ROOT
    overwrite_existing: bool = OVERWRITE_EXISTING


AUDIO2VIDEO_CONFIG = Audio2VideoConfig()

# Configure your audio-driven bend experiments here.
EXPERIMENTS: list[AudioBendSpec] = [
    {
        "name": "rms_add_step7",
        "function": "add_scalar",
        "params": {"value": 0.0},
        "steps": [7],
        "param_name": "value",
        "param_min": 0.0,
        "param_max": 0.3,
        "feature": "rms",
    },
    {
        "name": "rms_add_step2",
        "function": "add_scalar",
        "params": {"value": 0.0},
        "steps": [2],
        "param_name": "value",
        "param_min": 0.0,
        "param_max": 0.3,
        "feature": "rms",
    },
]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _safe_run_name(name: str) -> str:
    keep = "-_ ."
    return "".join(c if c.isalnum() or c in keep else "_" for c in name).replace(" ", "_")


def _create_batch_run_dir(root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"run_{timestamp}"
    run_dir = root / base_name
    suffix = 1

    while run_dir.exists():
        run_dir = root / f"{base_name}_{suffix:02d}"
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _normalize_time_progress(video_state: LatentState) -> torch.Tensor:
    if video_state.positions is None:
        raise ValueError("video_state.positions is required for audio-driven scheduling")

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


def _interp_1d(query_x: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.numel() != y.numel():
        raise ValueError("x and y must have the same length")
    if x.numel() == 0:
        raise ValueError("Cannot interpolate from empty arrays")
    if x.numel() == 1:
        return torch.full_like(query_x, y[0])

    idx = torch.searchsorted(x, query_x, right=True)
    idx = idx.clamp(min=1, max=x.numel() - 1)

    x0 = x[idx - 1]
    x1 = x[idx]
    y0 = y[idx - 1]
    y1 = y[idx]

    alpha = ((query_x - x0) / (x1 - x0).clamp_min(1e-8)).clamp(0.0, 1.0)
    out = y0 + alpha * (y1 - y0)
    out = torch.where(query_x <= x[0], y[0], out)
    out = torch.where(query_x >= x[-1], y[-1], out)
    return out


def _compute_rms_per_frame(audio_waveform: torch.Tensor, sample_rate: int, num_frames: int, frame_rate: float) -> torch.Tensor:
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    if frame_rate <= 0:
        raise ValueError("frame_rate must be > 0")

    # audio_waveform shape is expected as (channels, samples).
    if audio_waveform.ndim != 2:
        raise ValueError(f"audio_waveform must have shape (channels, samples), got {tuple(audio_waveform.shape)}")

    mono = audio_waveform.mean(dim=0).to(dtype=torch.float32)
    sample_count = int(mono.numel())

    # Recommended frame-sized RMS window: roughly one displayed frame in audio time.
    window = max(1, int(round(sample_rate / frame_rate)))
    hop = max(1, window // 2)  # 50% overlap.

    if sample_count == 0:
        return torch.zeros(num_frames, dtype=torch.float32)

    if sample_count < window:
        padded = torch.zeros(window, dtype=torch.float32)
        padded[:sample_count] = mono
        mono = padded
        sample_count = int(mono.numel())

    starts = list(range(0, max(sample_count - window, 0) + 1, hop))
    last_start = sample_count - window
    if starts[-1] != last_start:
        starts.append(last_start)

    rms_values = []
    rms_times = []
    for start in starts:
        end = start + window
        segment = mono[start:end]
        rms = torch.sqrt(torch.mean(segment * segment).clamp_min(0.0))
        center_sample = start + (window * 0.5)
        center_time = center_sample / float(sample_rate)
        rms_values.append(rms)
        rms_times.append(center_time)

    rms_curve = torch.stack(rms_values)
    rms_t = torch.tensor(rms_times, dtype=torch.float32)

    frame_times = torch.arange(num_frames, dtype=torch.float32) / float(frame_rate)
    return _interp_1d(frame_times, rms_t, rms_curve)


def _scale_min_max(values: torch.Tensor, out_min: float, out_max: float) -> torch.Tensor:
    if not torch.isfinite(values).all():
        raise ValueError("Feature values contain NaN or inf")

    lo = torch.min(values)
    hi = torch.max(values)
    out_min_f = float(out_min)
    out_max_f = float(out_max)

    if math.isclose(float(hi - lo), 0.0, abs_tol=1e-12):
        mid = (out_min_f + out_max_f) * 0.5
        return torch.full_like(values, mid)

    normalized = (values - lo) / (hi - lo)
    return out_min_f + normalized * (out_max_f - out_min_f)


def _parameter_vector_from_frame_values(video_state: LatentState, frame_values: torch.Tensor) -> torch.Tensor:
    if frame_values.ndim != 1:
        raise ValueError("frame_values must be a 1D tensor")

    t = _normalize_time_progress(video_state)
    if frame_values.numel() == 1:
        return torch.full_like(t, float(frame_values[0]))

    x = torch.linspace(0.0, 1.0, steps=int(frame_values.numel()), device=t.device, dtype=torch.float32)
    y = frame_values.to(device=t.device, dtype=torch.float32)
    return _interp_1d(t, x, y)


def _apply_bend(
    latent: torch.Tensor,
    fn_name: BendFunctionName,
    params: dict[str, float | int | bool | torch.Tensor],
) -> torch.Tensor:
    def _expand_over_token_axis(values: torch.Tensor) -> torch.Tensor:
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

    raise ValueError(f"Unsupported bend function for audio scheduling: {fn_name}")


def _build_audio_bending(spec: AudioBendSpec, per_frame_param_values: torch.Tensor):
    target_steps = set(spec["steps"])
    param_name = spec["param_name"]

    def bending(video_state: LatentState, step_idx: int) -> LatentState:
        if step_idx not in target_steps:
            return video_state

        params: dict[str, float | int | bool | torch.Tensor] = dict(spec["params"])
        params[param_name] = _parameter_vector_from_frame_values(video_state, per_frame_param_values)

        print(f"  Applying '{spec['function']}' at step {step_idx}")
        bent = _apply_bend(video_state.latent, spec["function"], params)
        return replace(video_state, latent=bent)

    return bending


def _ensure_stereo_audio(audio):
    waveform = audio.waveform
    if waveform.ndim == 3:
        # decode_audio_from_file commonly returns (1, channels, samples)
        if waveform.shape[0] != 1:
            raise ValueError(f"Unsupported batched audio shape: {tuple(waveform.shape)}")
        waveform = waveform.squeeze(0)
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim != 2:
        raise ValueError(f"Unsupported audio waveform shape: {tuple(waveform.shape)}")

    # Convert to channel-first [channels, samples] before stereo normalization.
    if waveform.shape[0] not in (1, 2) and waveform.shape[1] in (1, 2):
        waveform = waveform.T

    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2, :]

    # _write_audio expects either [samples, 2] or [2, samples], but [samples, 2]
    # avoids ambiguous reshaping and packs correctly for interleaved stereo.
    waveform = waveform.T.contiguous()

    if waveform.ndim != 2 or waveform.shape[1] != 2:
        raise ValueError(f"Failed to normalize to stereo, got shape: {tuple(waveform.shape)}")
    return replace(audio, waveform=waveform)


def _waveform_to_channels_samples(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 3:
        if waveform.shape[0] != 1:
            raise ValueError(f"Unsupported batched waveform shape: {tuple(waveform.shape)}")
        waveform = waveform.squeeze(0)
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim != 2:
        raise ValueError(f"Unsupported waveform shape: {tuple(waveform.shape)}")

    # If it's [samples, channels], transpose to [channels, samples].
    if waveform.shape[0] not in (1, 2) and waveform.shape[1] in (1, 2):
        waveform = waveform.T
    return waveform


def _mux_audio_with_ffmpeg(video_path: Path, audio_path: Path, output_path: Path, duration_seconds: float) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed or not available in PATH")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-t",
        f"{duration_seconds:.6f}",
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        details = stderr if stderr else stdout
        raise RuntimeError(f"ffmpeg mux failed (exit {exc.returncode}): {details}") from exc


@torch.inference_mode()
def main() -> None:
    audio_cfg = AUDIO2VIDEO_CONFIG
    audio_path = audio_cfg.audio_path.expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
    if audio_cfg.feature != "rms":
        raise ValueError(f"Unsupported audio feature: {audio_cfg.feature}. Only 'rms' is supported.")
    if audio_cfg.quick_test_seconds is not None and float(audio_cfg.quick_test_seconds) <= 0.0:
        raise ValueError("quick_test_seconds must be > 0 when provided")

    cfg = load_config()
    validate_config(cfg)

    audio_cfg.output_root.mkdir(parents=True, exist_ok=True)
    batch_run_dir = _create_batch_run_dir(audio_cfg.output_root)

    full_audio = decode_audio_from_file(
        path=str(audio_path),
        device=torch.device("cpu"),
        start_time=0.0,
        max_duration=None,
    )
    if full_audio is None:
        raise RuntimeError(f"Could not decode audio from: {audio_path}")

    audio_duration_seconds = float(full_audio.waveform.shape[-1]) / float(full_audio.sampling_rate)
    target_duration_seconds = audio_duration_seconds
    if audio_cfg.quick_test_seconds is not None:
        target_duration_seconds = min(target_duration_seconds, float(audio_cfg.quick_test_seconds))

    # Match generated video length to audio-driven target duration.
    target_num_frames = max(1, int(round(target_duration_seconds * float(cfg.frame_rate))))
    target_duration_seconds = float(target_num_frames) / float(cfg.frame_rate)

    source_audio = decode_audio_from_file(
        path=str(audio_path),
        device=torch.device("cpu"),
        start_time=0.0,
        max_duration=target_duration_seconds,
    )
    if source_audio is None:
        raise RuntimeError(f"Could not decode trimmed audio from: {audio_path}")
    source_audio = _ensure_stereo_audio(source_audio)

    print(
        "Timeline settings: "
        f"audio_seconds={audio_duration_seconds:.3f}, "
        f"target_seconds={target_duration_seconds:.3f}, "
        f"num_frames={target_num_frames}, "
        f"fps={float(cfg.frame_rate):.3f}"
    )

    waveform = _waveform_to_channels_samples(source_audio.waveform)
    rms_per_frame = _compute_rms_per_frame(
        audio_waveform=waveform,
        sample_rate=int(source_audio.sampling_rate),
        num_frames=target_num_frames,
        frame_rate=float(cfg.frame_rate),
    )

    print(
        "RMS feature stats: "
        f"min={float(torch.min(rms_per_frame)):.6f}, "
        f"max={float(torch.max(rms_per_frame)):.6f}, "
        f"first={float(rms_per_frame[0]):.6f}, "
        f"last={float(rms_per_frame[-1]):.6f}"
    )

    pipeline = DistilledPipeline(
        distilled_checkpoint_path=cfg.distilled_checkpoint_path,
        gemma_root=cfg.gemma_root,
        spatial_upsampler_path=cfg.spatial_upsampler_path,
        loras=cfg.loras,
        quantization=cfg.quantization_policy(),
        torch_compile=cfg.torch_compile,
    )

    run_manifest: list[dict[str, Any]] = []

    for run_idx, spec in enumerate(EXPERIMENTS, start=1):
        run_name = f"{run_idx:03d}_{_safe_run_name(spec['name'])}"
        run_dir = batch_run_dir / run_name
        video_path = run_dir / "video.mp4"
        silent_video_path = run_dir / "video_no_audio.mp4"

        if run_dir.exists() and not audio_cfg.overwrite_existing:
            print(f"[{run_idx}/{len(EXPERIMENTS)}] Skipping existing run: {run_name}")
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{run_idx}/{len(EXPERIMENTS)}] Running: {run_name}")

        scaled_param_values = _scale_min_max(
            rms_per_frame,
            out_min=float(spec["param_min"]),
            out_max=float(spec["param_max"]),
        )
        print(
            f"  Scaled {spec['param_name']} stats: "
            f"min={float(torch.min(scaled_param_values)):.6f}, "
            f"max={float(torch.max(scaled_param_values)):.6f}"
        )

        denoising_loop = make_network_bending_loop(_build_audio_bending(spec, scaled_param_values))
        video_chunks, _ltx_generated_audio = pipeline(
            prompt=cfg.prompt,
            seed=cfg.seed,
            height=cfg.height,
            width=cfg.width,
            num_frames=target_num_frames,
            frame_rate=cfg.frame_rate,
            images=[],
            tiling_config=None,
            enhance_prompt=cfg.enhance_prompt,
            streaming_prefetch_count=1,
            denoising_loop=denoising_loop,
        )

        chunks = tuple(video_chunks)
        if not chunks:
            raise RuntimeError(f"Pipeline returned no video chunks for run: {run_name}")
        video_tensor = torch.cat(chunks, dim=2)

        encode_video(
            video=video_tensor,
            fps=int(cfg.frame_rate),
            audio=None,
            output_path=str(silent_video_path),
            video_chunks_number=1,
        )

        _mux_audio_with_ffmpeg(
            video_path=silent_video_path,
            audio_path=audio_path,
            output_path=video_path,
            duration_seconds=target_duration_seconds,
        )

        run_info = {
            "run": run_name,
            "audio_path": str(audio_path),
            "audio_duration_seconds": audio_duration_seconds,
            "target_duration_seconds": target_duration_seconds,
            "num_frames": target_num_frames,
            "frame_rate": float(cfg.frame_rate),
            "feature": spec["feature"],
            "function": spec["function"],
            "param_name": spec["param_name"],
            "param_min": spec["param_min"],
            "param_max": spec["param_max"],
            "params": spec["params"],
            "steps": spec["steps"],
            "video_path": str(video_path),
        }
        run_manifest.append(run_info)

        with open(run_dir / "run.json", "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2)

    manifest_path = batch_run_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    print(f"Completed {len(run_manifest)} run(s). Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
