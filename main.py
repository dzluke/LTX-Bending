from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image

from ltx_pipelines import DistilledPipeline
from ltx_pipelines.utils.media_io import encode_video


@dataclass
class Config:
	# Required model paths
	distilled_checkpoint_path: str = "/absolute/path/to/ltx2_distilled.safetensors"
	spatial_upsampler_path: str = "/absolute/path/to/spatial_upsampler.safetensors"
	gemma_root: str = "/absolute/path/to/gemma"

	# Prompt / generation settings
	prompt: str = "A cinematic portrait of a fox in a misty forest at sunrise"
	seed: int = 42
	width: int = 768
	height: int = 512

	# Set num_frames > 1 for actual motion video.
	num_frames: int = 49
	frame_rate: float = 24.0

	# Output
	output_frames_dir: str = "outputs/frames"
	output_video_path: str = "outputs/generated.mp4"

	# Optional extras
	enhance_prompt: bool = False
	torch_compile: bool = False
	loras: list = field(default_factory=list)
	quantization: object | None = None


def validate_config(cfg: Config) -> None:
	required_paths = [
		("distilled_checkpoint_path", cfg.distilled_checkpoint_path),
		("spatial_upsampler_path", cfg.spatial_upsampler_path),
		("gemma_root", cfg.gemma_root),
	]
	missing = [name for name, value in required_paths if not Path(value).exists()]
	if missing:
		joined = ", ".join(missing)
		raise FileNotFoundError(
			f"Missing required path(s): {joined}. Update Config at the top of main.py."
		)


def save_frames_and_collect(video_chunks, frames_dir: str) -> torch.Tensor:
	frames_path = Path(frames_dir)
	frames_path.mkdir(parents=True, exist_ok=True)

	chunks: list[torch.Tensor] = []
	frame_index = 0
	for chunk in video_chunks:
		if chunk.ndim != 4:
			raise RuntimeError(f"Unexpected video chunk shape: {tuple(chunk.shape)}")
		if chunk.dtype != torch.uint8:
			chunk = chunk.clamp(0, 255).to(torch.uint8)

		chunks.append(chunk)
		for frame in chunk:
			frame_path = frames_path / f"frame_{frame_index:05d}.png"
			Image.fromarray(frame.cpu().numpy(), mode="RGB").save(frame_path)
			frame_index += 1

	if not chunks:
		raise RuntimeError("The pipeline returned no video chunks.")

	video_tensor = torch.cat(chunks, dim=0)
	print(f"Saved {video_tensor.shape[0]} frames to {frames_path.resolve()}")
	return video_tensor


def main() -> None:
	cfg = Config()
	validate_config(cfg)

	pipeline = DistilledPipeline(
		distilled_checkpoint_path=cfg.distilled_checkpoint_path,
		gemma_root=cfg.gemma_root,
		spatial_upsampler_path=cfg.spatial_upsampler_path,
		loras=cfg.loras,
		quantization=cfg.quantization,
		torch_compile=cfg.torch_compile,
	)

	video_chunks, audio = pipeline(
		prompt=cfg.prompt,
		seed=cfg.seed,
		height=cfg.height,
		width=cfg.width,
		num_frames=cfg.num_frames,
		frame_rate=cfg.frame_rate,
		images=[],
		tiling_config=None,
		enhance_prompt=cfg.enhance_prompt,
		streaming_prefetch_count=None,
	)

	video_tensor = save_frames_and_collect(video_chunks, cfg.output_frames_dir)

	video_path = Path(cfg.output_video_path)
	video_path.parent.mkdir(parents=True, exist_ok=True)
	encode_video(
		video=video_tensor,
		fps=int(cfg.frame_rate),
		audio=audio,
		output_path=str(video_path),
		video_chunks_number=1,
	)
	print(f"Saved video to {video_path.resolve()}")


if __name__ == "__main__":
	main()
