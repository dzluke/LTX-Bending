from __future__ import annotations

import asyncio
import logging
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from fastapi.concurrency import run_in_threadpool

from bending_functions import add_scalar, invert, multiply_scalar, reflect, rotate
from ltx_core.types import LatentState
from ltx_pipelines import DistilledPipeline
from ltx_pipelines.utils.bending import VideoBendingFn, make_network_bending_loop
from ltx_pipelines.utils.helpers import cleanup_memory
from ltx_pipelines.utils.media_io import encode_video

from ltx_server.schemas import BendFunctionName, BendSpec, GenerateRequest
from ltx_server.storage import generation_dir, video_path

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Owns the single shared DistilledPipeline and serializes generation requests."""

    def __init__(self, pipeline: DistilledPipeline) -> None:
        self._pipeline = pipeline
        self._lock = asyncio.Lock()

    async def generate(self, gen_id: str, req: GenerateRequest) -> None:
        async with self._lock:
            try:
                await run_in_threadpool(self._run_sync, gen_id, req)
            finally:
                cleanup_memory()

    def _run_sync(self, gen_id: str, req: GenerateRequest) -> None:
        bend_fn = compile_bending_specs(req.bending_ops)
        loop = make_network_bending_loop(bend_fn)

        with torch.inference_mode():
            video_chunks, audio = self._pipeline(
                prompt=req.prompt,
                seed=req.seed,
                height=req.height,
                width=req.width,
                num_frames=req.num_frames,
                frame_rate=req.frame_rate,
                images=[],
                tiling_config=None,
                enhance_prompt=req.enhance_prompt,
                streaming_prefetch_count=req.streaming_prefetch_count,
                denoising_loop=loop,
            )

            out_dir = generation_dir(gen_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="frames_", dir=out_dir) as frames_dir:
                video_tensor = _collect_chunks(video_chunks, Path(frames_dir))

            encode_video(
                video=video_tensor,
                fps=int(req.frame_rate),
                audio=audio,
                output_path=str(video_path(gen_id)),
                video_chunks_number=1,
            )


def _collect_chunks(video_chunks: Any, frames_dir: Path) -> torch.Tensor:
    del frames_dir
    chunks: list[torch.Tensor] = []
    for chunk in video_chunks:
        if chunk.ndim != 4:
            raise RuntimeError(f"Unexpected video chunk shape: {tuple(chunk.shape)}")
        if chunk.dtype != torch.uint8:
            chunk = chunk.clamp(0, 255).to(torch.uint8)
        chunks.append(chunk)
    if not chunks:
        raise RuntimeError("The pipeline returned no video chunks.")
    return torch.cat(chunks, dim=0)


def apply_bend(
    latent: torch.Tensor,
    fn_name: BendFunctionName,
    params: dict[str, Any],
) -> torch.Tensor:
    if fn_name == "add_scalar":
        return add_scalar(latent, value=float(params.get("value", 0.0)))
    if fn_name == "multiply_scalar":
        return multiply_scalar(latent, factor=float(params.get("factor", 1.0)))
    if fn_name == "invert":
        return invert(latent)
    if fn_name == "reflect":
        return reflect(latent, dim=int(params.get("dim", -1)))
    if fn_name == "rotate":
        return rotate(latent, k=int(params.get("k", 1)))
    raise ValueError(f"Unsupported bend function: {fn_name}")


def compile_bending_specs(specs: list[BendSpec]) -> VideoBendingFn:
    """Compile BendSpecs into a VideoBendingFn. Specs with overlapping steps run in list order."""
    by_step: dict[int, list[BendSpec]] = {}
    for spec in specs:
        for step in spec.steps:
            by_step.setdefault(step, []).append(spec)

    def bending(state: LatentState, step_idx: int) -> LatentState:
        step_specs = by_step.get(step_idx)
        if not step_specs:
            return state
        latent = state.latent
        for spec in step_specs:
            latent = apply_bend(latent, spec.function, spec.params)
        return replace(state, latent=latent)

    return bending
