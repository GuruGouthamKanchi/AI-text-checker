#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "[VeriPaper Frontend] Checking port 3000..."
if lsof -Pi :3000 -sTCP:LISTEN -t >/dev/null ; then
    PORT_PID=$(lsof -t -i:3000 -sTCP:LISTEN)
    echo "Port 3000 is already in use by process PID $PORT_PID."
    echo "Frontend appears to be already running. Run ./stop_frontend.sh first."
    exit 1
fi

cd frontend
if [ ! -d "node_modules" ]; then
    echo "[VeriPaper Frontend] Installing dependencies (this may take a few minutes)..."
    npm install
    if [ $? -ne 0 ]; then
        echo "Failed to install frontend dependencies."
        exit 1
    fi
fi

echo "[VeriPaper Frontend] Starting Next.js server on port 3000 in background..."
nohup npm run dev > ../frontend.log 2>&1 &

cd ..

# Loop to wait for Next.js server to bind to port 3000
FRONTEND_PID=""
for i in {1..15}; do
    if [ -z "$FRONTEND_PID" ]; then
        sleep 2
        FRONTEND_PID=$(lsof -t -i:3000 -sTCP:LISTEN 2>/dev/null)
    fi
done

if [ ! -z "$FRONTEND_PID" ]; then
    echo "$FRONTEND_PID" > .frontend.pid
    echo "[VeriPaper Frontend] Frontend running at http://localhost:3000 (PID: $FRONTEND_PID)."
    echo "[VeriPaper Frontend] Run ./stop_frontend.sh to stop it."
else
    echo "[VeriPaper Frontend] Failed to start frontend or bind to port 3000."
    echo "[VeriPaper Frontend] Please check frontend.log for details."
    exit 1
fi
