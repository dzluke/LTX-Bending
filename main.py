from __future__ import annotations

import os
from dataclasses import replace

from ltx_core.types import LatentState
from ltx_pipelines.utils.bending import make_network_bending_loop
from ltx_pipelines.utils.samplers import euler_denoising_loop

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path
from typing import Literal

import torch
import yaml
from PIL import Image
from pydantic import BaseModel

from ltx_pipelines import DistilledPipeline
from ltx_pipelines.utils.media_io import encode_video
from ltx_core.quantization import QuantizationPolicy  # noqa: E402  (must follow ltx_pipelines to avoid circular import)

CONFIG_PATH = Path(__file__).parent / "config.yaml"

QuantizationName = Literal["fp8_cast", "fp8_scaled_mm"]


class Config(BaseModel):
	model_config = {"extra": "forbid"}

	distilled_checkpoint_path: str
	spatial_upsampler_path: str
	gemma_root: str

	prompt: str = "A cinematic portrait of a fox in a misty forest at sunrise"
	seed: int = 42
	width: int = 768
	height: int = 512

	num_frames: int = 49
	frame_rate: float = 24.0

	output_frames_dir: str = "outputs/frames"
	output_video_path: str = "outputs/generated.mp4"

	enhance_prompt: bool = False
	torch_compile: bool = False
	loras: list = []
	quantization: QuantizationName | None = "fp8_cast"

	def quantization_policy(self) -> QuantizationPolicy | None:
		if self.quantization is None:
			return None
		return getattr(QuantizationPolicy, self.quantization)()


def load_config(path: Path = CONFIG_PATH) -> Config:
	if not path.exists():
		raise FileNotFoundError(
			f"Config file not found at {path}. Copy config.example.yaml to config.yaml and edit it.",
		)
	with open(path) as f:
		data = yaml.safe_load(f) or {}
	return Config(**data)


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
			f"Missing required path(s): {joined}. Update {CONFIG_PATH.name}.",
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


@torch.inference_mode()
def main() -> None:
	cfg = load_config()
	validate_config(cfg)

	denoising_loop = euler_denoising_loop

	def bending(video_state: LatentState, step_idx: int) -> LatentState:
		if step_idx == 4:
			print("Applying bending at step 4!")
			return replace(video_state, latent=video_state.latent * 2.0)
		return video_state

	denoising_loop2 = make_network_bending_loop(bending)

	pipeline = DistilledPipeline(
		distilled_checkpoint_path=cfg.distilled_checkpoint_path,
		gemma_root=cfg.gemma_root,
		spatial_upsampler_path=cfg.spatial_upsampler_path,
		loras=cfg.loras,
		quantization=cfg.quantization_policy(),
		torch_compile=cfg.torch_compile,
		denoising_loop=denoising_loop2,
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
		streaming_prefetch_count=1,
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
