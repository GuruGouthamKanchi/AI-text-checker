#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

STOPPED=0

if [ -f .backend.pid ]; then
    PID=$(cat .backend.pid)
    if [ ! -z "$PID" ]; then
        if kill -0 $PID 2>/dev/null; then
            echo "[VeriPaper Backend] Stopping process PID $PID..."
            kill -9 $PID 2>/dev/null
            STOPPED=1
        else
            echo "[VeriPaper Backend] PID file found but process $PID is not running."
        fi
    fi
    rm .backend.pid
else
    echo "[VeriPaper Backend] No .backend.pid file found. Checking port 8000..."
fi

# Safety net: forcefully terminate any remaining process on port 8000
PORT_PID=$(lsof -t -i:8000 -sTCP:LISTEN 2>/dev/null)
if [ ! -z "$PORT_PID" ]; then
    echo "[VeriPaper Backend] Found process PID $PORT_PID listening on port 8000. Forcefully stopping..."
    kill -9 $PORT_PID 2>/dev/null
    STOPPED=1
fi

if [ $STOPPED -eq 1 ]; then
    echo "[VeriPaper Backend] Backend stopped successfully."
else
    echo "[VeriPaper Backend] Backend doesn't appear to be running."
fi
