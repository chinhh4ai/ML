#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
    echo "Error: Virtual environment not found at $PWD/.venv" >&2
    echo "Please run 'bash scripts/setup.sh' to set up the environment first." >&2
    exit 1
fi
exec .venv/bin/python -m uvicorn backend.main:app --host "${APP_HOST:-0.0.0.0}" --port 5050 --reload

