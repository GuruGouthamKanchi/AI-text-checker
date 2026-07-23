#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

STOPPED=0

if [ -f .frontend.pid ]; then
    PID=$(cat .frontend.pid)
    if [ ! -z "$PID" ]; then
        if kill -0 $PID 2>/dev/null; then
            echo "[VeriPaper Frontend] Stopping process PID $PID..."
            kill -9 $PID 2>/dev/null
            STOPPED=1
        else
            echo "[VeriPaper Frontend] PID file found but process $PID is not running."
        fi
    fi
    rm .frontend.pid
else
    echo "[VeriPaper Frontend] No .frontend.pid file found. Checking port 3000..."
fi

# Safety net: forcefully terminate any remaining process on port 3000
PORT_PID=$(lsof -t -i:3000 -sTCP:LISTEN 2>/dev/null)
if [ ! -z "$PORT_PID" ]; then
    echo "[VeriPaper Frontend] Found process PID $PORT_PID listening on port 3000. Forcefully stopping..."
    kill -9 $PORT_PID 2>/dev/null
    STOPPED=1
fi

if [ $STOPPED -eq 1 ]; then
    echo "[VeriPaper Frontend] Frontend stopped successfully."
else
    echo "[VeriPaper Frontend] Frontend doesn't appear to be running."
fi
