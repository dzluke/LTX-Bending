from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GENERATIONS_ROOT = Path(os.environ.get("LTX_GENERATIONS_DIR", "generations")).resolve()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _elapsed_since(iso_ts: str) -> float:
    try:
        started = datetime.fromisoformat(iso_ts)
    except ValueError:
        return 0.0
    return max(0.0, (datetime.now(timezone.utc) - started).total_seconds())


def _close_current_phase(status: dict[str, Any]) -> None:
    """Record the duration of the phase currently in `status` into `phase_durations`."""
    phase = status.get("phase")
    started = status.get("phase_started_at")
    if not phase or not started:
        return
    durations = status.setdefault("phase_durations", {})
    durations[phase] = round(durations.get(phase, 0.0) + _elapsed_since(started), 3)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON via temp-file + rename so readers never see a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def _read_json_tolerant(path: Path) -> dict[str, Any] | None:
    """Read JSON, returning None if the file is missing or transiently unreadable."""
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def new_generation_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    return f"{ts}_{secrets.token_hex(3)}"


def generation_dir(gen_id: str) -> Path:
    return GENERATIONS_ROOT / gen_id


def create_generation(gen_id: str, config: dict[str, Any]) -> Path:
    path = generation_dir(gen_id)
    path.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path / "config.json", config)
    write_status(gen_id, {"state": "running", "started_at": _now_iso()})
    return path


def write_status(gen_id: str, status: dict[str, Any]) -> None:
    _atomic_write_json(generation_dir(gen_id) / "status.json", status)


def mark_done(gen_id: str) -> None:
    path = generation_dir(gen_id) / "status.json"
    status = _read_json_tolerant(path) or {"started_at": _now_iso()}
    _close_current_phase(status)
    status["state"] = "done"
    status["finished_at"] = _now_iso()
    status.pop("phase", None)
    status.pop("phase_started_at", None)
    _atomic_write_json(path, status)


def mark_error(gen_id: str, message: str) -> None:
    path = generation_dir(gen_id) / "status.json"
    status = _read_json_tolerant(path) or {"started_at": _now_iso()}
    _close_current_phase(status)
    status["state"] = "error"
    status["finished_at"] = _now_iso()
    status["error"] = message
    status.pop("phase_started_at", None)
    _atomic_write_json(path, status)


def update_phase(gen_id: str, phase: str, step: int | None = None, total: int | None = None) -> None:
    path = generation_dir(gen_id) / "status.json"
    status = _read_json_tolerant(path)
    if status is None:
        return
    if status.get("phase") != phase:
        _close_current_phase(status)
        status["phase"] = phase
        status["phase_started_at"] = _now_iso()
        status.pop("step", None)
        status.pop("total", None)
    if step is not None:
        status["step"] = step
    if total is not None:
        status["total"] = total
    _atomic_write_json(path, status)


def read_generation(gen_id: str) -> dict[str, Any] | None:
    path = generation_dir(gen_id)
    if not path.is_dir():
        return None
    config = _read_json_tolerant(path / "config.json")
    if config is None:
        return None
    status = _read_json_tolerant(path / "status.json") or {"state": "error", "started_at": ""}
    return {"id": gen_id, "config": config, "status": status}


def clear_stale_running() -> int:
    """On server startup, mark any generations stuck in 'running' as errored.

    These are leftovers from a previous process — no running generation survives
    a restart because the pipeline lock/task lives in memory.
    """
    if not GENERATIONS_ROOT.exists():
        return 0
    count = 0
    for child in GENERATIONS_ROOT.iterdir():
        if not child.is_dir():
            continue
        status_path = child / "status.json"
        status = _read_json_tolerant(status_path)
        if status and status.get("state") == "running":
            _close_current_phase(status)
            status["state"] = "error"
            status["finished_at"] = _now_iso()
            status["error"] = "server restarted before generation finished"
            status.pop("phase", None)
            status.pop("phase_started_at", None)
            _atomic_write_json(status_path, status)
            count += 1
    return count


def list_generations() -> list[dict[str, Any]]:
    if not GENERATIONS_ROOT.exists():
        return []
    entries: list[dict[str, Any]] = []
    for child in GENERATIONS_ROOT.iterdir():
        if not child.is_dir():
            continue
        entry = read_generation(child.name)
        if entry is not None:
            entries.append(entry)
    entries.sort(key=lambda e: e["id"], reverse=True)
    return entries


def video_path(gen_id: str) -> Path:
    return generation_dir(gen_id) / "video.mp4"
