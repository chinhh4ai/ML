#!/usr/bin/env bash
# ========================================================
# Start Gauge Detection & Reading Application
# ========================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Prefer hip1 environment where dependencies are installed
PYTHON_BIN="/home/chinhh/miniconda3/envs/hip1/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

# Load port from .env or default to 5070
PORT=$(grep -E "^PORT=" .env 2>/dev/null | cut -d '=' -f 2 || echo "5070")
PORT="${PORT:-5070}"

HOST=$(grep -E "^HOST=" .env 2>/dev/null | cut -d '=' -f 2 || echo "0.0.0.0")
HOST="${HOST:-0.0.0.0}"

# Automatically clean up any zombie or previous process occupying this port
OLD_PIDS=$(lsof -t -i :"$PORT" 2>/dev/null)
if [ -n "$OLD_PIDS" ]; then
    echo "⚠️  Port ${PORT} is currently occupied by PID(s): ${OLD_PIDS}. Auto-clearing old process..."
    kill -9 ${OLD_PIDS} 2>/dev/null || true
    sleep 0.5
fi

echo "=========================================================="
echo " Starting Industrial Gauge Detection & Reading AI System"
echo " Host: http://${HOST}:${PORT}"
echo " Python: ${PYTHON_BIN}"
echo " Active Provider: $(grep -E '^AI_PROVIDER=' .env 2>/dev/null || echo 'AI_PROVIDER=gemini')"
echo "=========================================================="

exec "$PYTHON_BIN" -m uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
