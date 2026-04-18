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
from urllib.parse import urlparse

import torch
import yaml
from huggingface_hub import hf_hub_download, snapshot_download
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


def _is_local_path(path_or_ref: str) -> bool:
	return path_or_ref.startswith(("/", "./", "../", "~"))


def _parse_hf_reference(path_or_ref: str) -> tuple[str, str | None, str | None]:
	revision: str | None = None

	if path_or_ref.startswith("hf://"):
		parts = [p for p in path_or_ref.removeprefix("hf://").strip("/").split("/") if p]
		if parts and parts[0] in {"models", "datasets", "spaces"}:
			parts = parts[1:]
	elif path_or_ref.startswith(("http://", "https://")):
		parsed = urlparse(path_or_ref)
		if parsed.netloc != "huggingface.co":
			raise ValueError("Only huggingface.co URLs are supported for remote model references.")
		parts = [p for p in parsed.path.strip("/").split("/") if p]
		if len(parts) >= 4 and parts[2] in {"resolve", "blob", "tree"}:
			revision = parts[3]
			parts = [parts[0], parts[1], *parts[4:]]
	else:
		parts = [p for p in path_or_ref.strip("/").split("/") if p]

	if len(parts) < 2:
		raise ValueError("Hugging Face references must include at least owner/repo.")

	repo_id = f"{parts[0]}/{parts[1]}"
	subpath = "/".join(parts[2:]) if len(parts) > 2 else None
	return repo_id, subpath, revision


def _resolve_model_reference(path_or_ref: str, *, expect_dir: bool, field_name: str) -> str:
	local_path = Path(path_or_ref).expanduser()
	if local_path.exists():
		return str(local_path.resolve())

	if _is_local_path(path_or_ref):
		raise FileNotFoundError(f"{field_name} does not exist: {path_or_ref}")

	try:
		repo_id, subpath, revision = _parse_hf_reference(path_or_ref)
	except ValueError as exc:
		raise FileNotFoundError(
			f"{field_name} does not exist: {path_or_ref}. Expected a local path or Hugging Face reference.",
		) from exc

	if expect_dir:
		allow_patterns: list[str] | None = None
		if subpath:
			allow_patterns = [f"{subpath}/**", f"{subpath}/*"]
		local_repo = Path(
			snapshot_download(
				repo_id=repo_id,
				revision=revision,
				allow_patterns=allow_patterns,
			)
		).resolve()
		resolved_dir = (local_repo / subpath).resolve() if subpath else local_repo
		if not resolved_dir.exists() or not resolved_dir.is_dir():
			raise FileNotFoundError(
				f"{field_name} resolved to a non-directory path: {resolved_dir}",
			)
		return str(resolved_dir)

	if not subpath:
		raise ValueError(
			f"{field_name} must include a file path. Use owner/repo/path/to/file or a huggingface.co resolve URL.",
		)

	resolved_file = hf_hub_download(repo_id=repo_id, filename=subpath, revision=revision)
	return str(Path(resolved_file).resolve())


def resolve_config_paths(cfg: Config) -> Config:
	resolved = {
		"distilled_checkpoint_path": _resolve_model_reference(
			cfg.distilled_checkpoint_path,
			expect_dir=False,
			field_name="distilled_checkpoint_path",
		),
		"spatial_upsampler_path": _resolve_model_reference(
			cfg.spatial_upsampler_path,
			expect_dir=False,
			field_name="spatial_upsampler_path",
		),
		"gemma_root": _resolve_model_reference(
			cfg.gemma_root,
			expect_dir=True,
			field_name="gemma_root",
		),
	}
	return cfg.model_copy(update=resolved)


def load_config(path: Path = CONFIG_PATH) -> Config:
	if not path.exists():
		raise FileNotFoundError(
			f"Config file not found at {path}. Copy config.example.yaml to config.yaml and edit it.",
		)
	with open(path) as f:
		data = yaml.safe_load(f) or {}
	cfg = Config(**data)
	return resolve_config_paths(cfg)


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
		denoising_loop=denoising_loop,
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
