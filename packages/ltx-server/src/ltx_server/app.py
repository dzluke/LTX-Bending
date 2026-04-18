from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from ltx_pipelines import DistilledPipeline  # noqa: E402

from main import load_config, validate_config  # noqa: E402
from ltx_server.pipeline_runner import PipelineRunner  # noqa: E402
from ltx_server.schemas import GenerateRequest  # noqa: E402
from ltx_server.storage import (  # noqa: E402
    GENERATIONS_ROOT,
    clear_stale_running,
    create_generation,
    list_generations,
    mark_done,
    mark_error,
    new_generation_id,
    read_generation,
    video_path,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    validate_config(cfg)
    GENERATIONS_ROOT.mkdir(parents=True, exist_ok=True)
    stale = clear_stale_running()
    if stale:
        logger.info("Cleared %d stale 'running' generation(s) from a previous process.", stale)

    logger.info("Loading DistilledPipeline (this may take a while)...")
    pipeline = DistilledPipeline(
        distilled_checkpoint_path=cfg.distilled_checkpoint_path,
        gemma_root=cfg.gemma_root,
        spatial_upsampler_path=cfg.spatial_upsampler_path,
        loras=cfg.loras,
        quantization=cfg.quantization_policy(),
        torch_compile=cfg.torch_compile,
    )
    app.state.runner = PipelineRunner(pipeline)
    logger.info("Pipeline ready.")
    yield


app = FastAPI(lifespan=lifespan)

_cors_origins = os.environ.get("LTX_CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict:
    return {"ok": True}


@app.get("/api/generations")
async def api_list_generations() -> list[dict]:
    return list_generations()


@app.get("/api/generations/{gen_id}")
async def api_get_generation(gen_id: str) -> dict:
    entry = read_generation(gen_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="not found")
    return entry


@app.get("/api/generations/{gen_id}/video")
async def api_get_video(gen_id: str):
    path = video_path(gen_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="video not available")
    return FileResponse(path, media_type="video/mp4")


async def _run_generation(runner: PipelineRunner, gen_id: str, req: GenerateRequest) -> None:
    try:
        await runner.generate(gen_id, req)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Generation %s failed", gen_id)
        mark_error(gen_id, f"{type(exc).__name__}: {exc}")
    else:
        mark_done(gen_id)


@app.post("/api/generate")
async def api_generate(req: GenerateRequest) -> dict:
    gen_id = new_generation_id()
    config_snapshot = req.model_dump()
    create_generation(gen_id, config_snapshot)

    runner: PipelineRunner = app.state.runner
    asyncio.create_task(_run_generation(runner, gen_id, req))

    return {
        "id": gen_id,
        "status": "running",
        "video_url": f"/api/generations/{gen_id}/video",
    }
