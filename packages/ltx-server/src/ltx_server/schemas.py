from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

BendFunctionName = Literal[
    "add_scalar",
    "multiply_scalar",
    "invert",
    "reflect",
    "rotate",
]


class BendSpec(BaseModel):
    """Matches the shape used by batch.py/BendSpec so configs round-trip across entry points."""

    model_config = {"extra": "allow"}

    name: str = ""
    function: BendFunctionName
    params: dict[str, Any] = Field(default_factory=dict)
    steps: list[int] = Field(default_factory=list)


class GenerateRequest(BaseModel):
    model_config = {"extra": "forbid"}

    prompt: str
    seed: int = 42
    width: int = 768
    height: int = 512
    num_frames: int = 49
    frame_rate: float = 24.0
    enhance_prompt: bool = False
    streaming_prefetch_count: int | None = 1
    bending_ops: list[BendSpec] = Field(default_factory=list)


class GenerationStatus(BaseModel):
    state: Literal["running", "done", "error"]
    started_at: str
    finished_at: str | None = None
    error: str | None = None


class GenerationSummary(BaseModel):
    id: str
    config: dict[str, Any]
    status: GenerationStatus
