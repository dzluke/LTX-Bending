# ltx-server

FastAPI server wrapping the LTX-2 distilled pipeline with per-request network-bending parameters.

Run from the repo root:

```
uv run uvicorn ltx_server.app:app --port 8000
```

Reads `config.yaml` at the repo root on startup (via `main.load_config`) to locate model checkpoints.
