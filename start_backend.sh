#!/bin/bash
# Get script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "[VeriPaper Backend] Checking port 8000..."
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    PORT_PID=$(lsof -t -i:8000 -sTCP:LISTEN)
    echo "Port 8000 is already in use by process PID $PORT_PID."
    echo "Backend appears to be already running. Run ./stop_backend.sh first."
    exit 1
fi

# Detect Python command
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    if command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo "Python is not installed or not in environment PATH."
        exit 1
    fi
fi

if [ ! -d ".venv" ]; then
    echo "[VeriPaper Backend] Creating virtual environment (.venv)..."
    $PYTHON_CMD -m venv --system-site-packages .venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment."
        exit 1
    fi
    echo "[VeriPaper Backend] Activating environment and installing dependencies..."
    source .venv/bin/activate
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Failed to install dependencies."
        exit 1
    fi
else
    echo "[VeriPaper Backend] Activating existing virtual environment..."
    source .venv/bin/activate
fi

echo "[VeriPaper Backend] Starting FastAPI server on port 8000 in background..."
nohup python -u -m uvicorn src.app:app --host 127.0.0.1 --port 8000 > backend.log 2>&1 &

# Loop to wait for startup and capture the PID
BACKEND_PID=""
for i in {1..10}; do
    if [ -z "$BACKEND_PID" ]; then
        sleep 2
        BACKEND_PID=$(lsof -t -i:8000 -sTCP:LISTEN 2>/dev/null)
    fi
done

if [ ! -z "$BACKEND_PID" ]; then
    echo "$BACKEND_PID" > .backend.pid
    echo "[VeriPaper Backend] Backend running at http://localhost:8000 (PID: $BACKEND_PID)."
    echo "[VeriPaper Backend] Run ./stop_backend.sh to stop it."
else
    echo "[VeriPaper Backend] Failed to start backend or bind to port 8000."
    echo "[VeriPaper Backend] Please check backend.log for details."
    exit 1
fi
