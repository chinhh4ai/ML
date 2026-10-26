#!/usr/bin/env bash
# ========================================================
# Start HIP Detection Unified Platform (Port 5075)
# ========================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_DIR="$(dirname "$SCRIPT_DIR")"
cd "$SCRIPT_DIR"

PYTHON_BIN="/home/chinhh/miniconda3/envs/hip1/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

PORT="${PORT:-5075}"
HOST="${HOST:-0.0.0.0}"
DEBRIS_PORT=5076
GAUGE_PORT=5077

cleanup() {
    echo ""
    echo "Stopping HIP Detection sub-services..."
    kill $(jobs -p) 2>/dev/null || true
    # Clear occupied ports
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    fuser -k "${DEBRIS_PORT}/tcp" 2>/dev/null || true
    fuser -k "${GAUGE_PORT}/tcp" 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Clean existing processes on ports
fuser -k "${PORT}/tcp" 2>/dev/null || true
fuser -k "${DEBRIS_PORT}/tcp" 2>/dev/null || true
fuser -k "${GAUGE_PORT}/tcp" 2>/dev/null || true
sleep 0.5

echo "=========================================================="
echo " Starting Debris Detection Service (Port ${DEBRIS_PORT})"
echo "=========================================================="
(
    cd "${ML_DIR}/debris_detection"
    exec "${ML_DIR}/debris_detection/.venv/bin/python" -m uvicorn backend.main:app --host 127.0.0.1 --port "$DEBRIS_PORT"
) &

echo "=========================================================="
echo " Starting Gauge Detection Service (Port ${GAUGE_PORT})"
echo "=========================================================="
(
    cd "${ML_DIR}/gauge_detection"
    exec "$PYTHON_BIN" -m uvicorn app.main:app --host 127.0.0.1 --port "$GAUGE_PORT"
) &

echo "Waiting for backend services to initialize..."
for i in {1..20}; do
    if curl -s "http://127.0.0.1:${DEBRIS_PORT}/api/health" >/dev/null 2>&1 && curl -s "http://127.0.0.1:${GAUGE_PORT}/api/config" >/dev/null 2>&1; then
        echo "Sub-services are up!"
        break
    fi
    sleep 0.5
done

echo "=========================================================="
echo " Starting Unified HIP Detection Gateway"
echo " URL: http://${HOST}:${PORT}"
echo " Pages: DEBRIS DETECTION | GAUGE DETECTION"
echo "=========================================================="

export DEBRIS_BACKEND_URL="http://127.0.0.1:${DEBRIS_PORT}"
export GAUGE_BACKEND_URL="http://127.0.0.1:${GAUGE_PORT}"

exec "$PYTHON_BIN" -m uvicorn main:app --host "$HOST" --port "$PORT"
