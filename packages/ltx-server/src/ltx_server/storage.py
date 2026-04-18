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


def new_generation_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    return f"{ts}_{secrets.token_hex(3)}"


def generation_dir(gen_id: str) -> Path:
    return GENERATIONS_ROOT / gen_id


def create_generation(gen_id: str, config: dict[str, Any]) -> Path:
    path = generation_dir(gen_id)
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config, indent=2))
    write_status(gen_id, {"state": "running", "started_at": _now_iso()})
    return path


def write_status(gen_id: str, status: dict[str, Any]) -> None:
    (generation_dir(gen_id) / "status.json").write_text(json.dumps(status, indent=2))


def mark_done(gen_id: str) -> None:
    path = generation_dir(gen_id) / "status.json"
    status = json.loads(path.read_text()) if path.exists() else {"started_at": _now_iso()}
    status["state"] = "done"
    status["finished_at"] = _now_iso()
    path.write_text(json.dumps(status, indent=2))


def mark_error(gen_id: str, message: str) -> None:
    path = generation_dir(gen_id) / "status.json"
    status = json.loads(path.read_text()) if path.exists() else {"started_at": _now_iso()}
    status["state"] = "error"
    status["finished_at"] = _now_iso()
    status["error"] = message
    path.write_text(json.dumps(status, indent=2))


def read_generation(gen_id: str) -> dict[str, Any] | None:
    path = generation_dir(gen_id)
    if not path.is_dir():
        return None
    config_path = path / "config.json"
    status_path = path / "status.json"
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text())
    status = json.loads(status_path.read_text()) if status_path.exists() else {"state": "error", "started_at": ""}
    return {"id": gen_id, "config": config, "status": status}


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
