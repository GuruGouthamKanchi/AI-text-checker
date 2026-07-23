#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=========================================="
echo "  Starting VeriPaper AI Forensic Services"
echo "=========================================="
echo ""

chmod +x start_backend.sh start_frontend.sh stop_backend.sh stop_frontend.sh stop_all.sh

./start_backend.sh
if [ $? -ne 0 ]; then
    echo "[VeriPaper All] Failed to start backend. Aborting startup."
    exit 1
fi

echo ""
./start_frontend.sh
if [ $? -ne 0 ]; then
    echo "[VeriPaper All] Failed to start frontend."
    exit 1
fi

echo ""
echo "=========================================="
echo "  All services started successfully!"
echo ""
echo "  - Frontend: http://localhost:3000"
echo "  - Backend API: http://localhost:8000"
echo "  - Stop: ./stop_all.sh"
echo "=========================================="
