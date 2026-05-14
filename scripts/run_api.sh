#!/bin/bash

# ChurnGuard AI - API Server Runner
# Usage: ./scripts/run_api.sh

cd "$(dirname "$0")/.."

echo "========================================================================"
echo "ChurnGuard AI - Starting API Server"
echo "========================================================================"

# Activate virtual environment
source venv/bin/activate

# Check if MLflow is running
if ! curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "Warning: MLflow server not detected at http://localhost:5000"
    echo "   Model loading will fail in Milestone 6.2"
    echo "   Start MLflow: mlflow server --backed-store_uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000"
    echo ""
fi

API_PORT="${API_PORT:-8000}"
if ss -tln 2>/dev/null | grep -qE ":${API_PORT}\s"; then
    echo "ERROR: Port ${API_PORT} is already in use (another API server or app is listening)."
    echo "Stop that process (Ctrl+C in its terminal), or free the port, for example:"
    echo "  fuser -k ${API_PORT}/tcp"
    echo "Or start on a different port:"
    echo "  API_PORT=8001 ./scripts/run_api.sh"
    exit 1
fi

# Run API server
uvicorn src.api.main:app \
    --host "${API_HOST:-0.0.0.0}" \
    --port "${API_PORT}" \
    --reload \
    --log-level info


echo ""
echo "API server stopped"