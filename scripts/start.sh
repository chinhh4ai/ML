#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m uvicorn backend.main:app --host "${APP_HOST:-127.0.0.1}" --port 5076
