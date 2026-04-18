from __future__ import annotations

import asyncio
import logging
import tempfile
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from fastapi.concurrency import run_in_threadpool

from bending_functions import add_scalar, invert, multiply_scalar, reflect, rotate
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput
from ltx_core.types import LatentState
from ltx_pipelines import DistilledPipeline
from ltx_pipelines.utils.bending import VideoBendingFn, make_network_bending_loop
from ltx_pipelines.utils.helpers import cleanup_memory
from ltx_pipelines.utils.media_io import encode_video

from ltx_server.schemas import BendFunctionName, BendSpec, GenerateRequest
from ltx_server.storage import generation_dir, update_phase, video_path

logger = logging.getLogger(__name__)

PROMPT_CACHE_MAX = 64


class PipelineRunner:
    """Owns the single shared DistilledPipeline and serializes generation requests."""

    def __init__(self, pipeline: DistilledPipeline) -> None:
        self._pipeline = pipeline
        self._lock = asyncio.Lock()
        self._prompt_cache: OrderedDict[tuple[str, bool], EmbeddingsProcessorOutput] = OrderedDict()

    def _get_prompt_embedding(
        self, prompt: str, enhance_prompt: bool, streaming_prefetch_count: int | None
    ) -> EmbeddingsProcessorOutput:
        key = (prompt, enhance_prompt)
        cached = self._prompt_cache.get(key)
        if cached is not None:
            self._prompt_cache.move_to_end(key)
            logger.info("prompt cache hit (size=%d)", len(self._prompt_cache))
        else:
            logger.info("prompt cache miss; encoding (size=%d)", len(self._prompt_cache))
            (ctx,) = self._pipeline.prompt_encoder(
                [prompt],
                enhance_first_prompt=enhance_prompt,
                enhance_prompt_image=None,
                streaming_prefetch_count=streaming_prefetch_count,
            )
            cached = _embedding_to(ctx, torch.device("cpu"))
            self._prompt_cache[key] = cached
            while len(self._prompt_cache) > PROMPT_CACHE_MAX:
                self._prompt_cache.popitem(last=False)
        return _embedding_to(cached, self._pipeline.device)

    async def generate(self, gen_id: str, req: GenerateRequest) -> None:
        async with self._lock:
            try:
                await run_in_threadpool(self._run_sync, gen_id, req)
            finally:
                cleanup_memory()

    def _run_sync(self, gen_id: str, req: GenerateRequest) -> None:
        logger.info(
            "[%s] generate: prompt=%r size=%sx%s frames=%s ops=%d",
            gen_id, req.prompt[:60], req.width, req.height, req.num_frames, len(req.bending_ops),
        )

        # Wrap the user-supplied bending with per-step progress reporting.
        # DistilledPipeline uses this loop for stage 1 only (8 steps).
        STAGE_1_STEPS = 8
        bend_fn = compile_bending_specs(req.bending_ops)

        def progress_bend(state: LatentState, step_idx: int) -> LatentState:
            update_phase(gen_id, "stage_1", step=step_idx, total=STAGE_1_STEPS)
            return bend_fn(state, step_idx)

        loop = make_network_bending_loop(progress_bend)

        def report_phase(phase: str) -> None:
            logger.info("[%s] phase: %s", gen_id, phase)
            update_phase(gen_id, phase)

        with torch.inference_mode():
            update_phase(gen_id, "encoding_prompt")
            prompt_embedding = self._get_prompt_embedding(
                req.prompt, req.enhance_prompt, req.streaming_prefetch_count
            )

            logger.info("[%s] running pipeline...", gen_id)
            video_chunks, audio = self._pipeline(
                prompt=prompt_embedding,
                seed=req.seed,
                height=req.height,
                width=req.width,
                num_frames=req.num_frames,
                frame_rate=req.frame_rate,
                images=[],
                tiling_config=None,
                enhance_prompt=False,
                streaming_prefetch_count=req.streaming_prefetch_count,
                denoising_loop=loop,
                progress=report_phase,
            )

            out_dir = generation_dir(gen_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="frames_", dir=out_dir) as frames_dir:
                update_phase(gen_id, "decoding")
                logger.info("[%s] collecting video chunks...", gen_id)
                video_tensor = _collect_chunks(video_chunks, Path(frames_dir))

            update_phase(gen_id, "encoding_video", step=video_tensor.shape[0])
            logger.info("[%s] encoding %d frames to mp4...", gen_id, video_tensor.shape[0])
            encode_video(
                video=video_tensor,
                fps=int(req.frame_rate),
                audio=audio,
                output_path=str(video_path(gen_id)),
                video_chunks_number=1,
            )
            logger.info("[%s] done", gen_id)


def _embedding_to(ctx: EmbeddingsProcessorOutput, device: torch.device) -> EmbeddingsProcessorOutput:
    return EmbeddingsProcessorOutput(
        video_encoding=ctx.video_encoding.to(device),
        audio_encoding=ctx.audio_encoding.to(device) if ctx.audio_encoding is not None else None,
        attention_mask=ctx.attention_mask.to(device),
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
