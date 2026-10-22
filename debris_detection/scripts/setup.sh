#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PIP_CACHE_DIR="$PWD/.data/pip-cache"
export npm_config_cache="$PWD/.data/npm-cache"
command -v ffmpeg >/dev/null || { echo 'Install FFmpeg first.'; exit 1; }
command -v ffprobe >/dev/null || { echo 'Install ffprobe first.'; exit 1; }
PYTHON_BIN="/usr/bin/python3"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python3)"
fi
if [ ! -x .venv/bin/python ]; then "$PYTHON_BIN" -m venv .venv; fi
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python scripts/verify_models.py
if [ ! -f .env ]; then cp .env.example .env; fi
if [ ! -f .data/models/yolo11n-seg.onnx ]; then
  if [ ! -x .model-venv/bin/python ]; then "$PYTHON_BIN" -m venv .model-venv; fi
  .model-venv/bin/python -m pip install -r requirements-export.txt
  .model-venv/bin/python scripts/export_model.py
fi
(cd frontend && npm ci && npm run build)
echo 'Ready. Edit .env then run: bash scripts/start.sh'
