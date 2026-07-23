#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=========================================="
echo "  Stopping VeriPaper AI Forensic Services"
echo "=========================================="
echo ""

chmod +x stop_backend.sh stop_frontend.sh

./stop_frontend.sh
echo ""
./stop_backend.sh

echo ""
echo "=========================================="
echo "  All services stopped successfully!"
echo "=========================================="
