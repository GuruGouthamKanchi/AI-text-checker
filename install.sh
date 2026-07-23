#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "==================================================="
echo "  VeriPaper AI Forensic Tool - Setup Installer"
echo "==================================================="
echo ""

echo "[VeriPaper Setup] Checking prerequisites..."

# 1. Check Python 3.9+
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" &> /dev/null; then
        PYTHON_CMD="python3"
    fi
fi

if [ -z "$PYTHON_CMD" ] && command -v python &> /dev/null; then
    if python -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" &> /dev/null; then
        PYTHON_CMD="python"
    fi
fi

if [ -z "$PYTHON_CMD" ]; then
    echo "[Error] Python 3.9 or later was not found on your system."
    echo "Please install Python 3.9 or later from https://www.python.org/downloads/"
    echo "and ensure it is added to your PATH, then re-run this script."
    exit 1
fi

PY_VER=$($PYTHON_CMD --version)
echo "[VeriPaper Setup] Found $PY_VER"

# 2. Check Node.js
if ! command -v node &> /dev/null; then
    echo "[Error] Node.js was not found on your system."
    echo "Please install Node.js from https://nodejs.org/ (LTS version recommended)"
    echo "and ensure it is added to your PATH, then re-run this script."
    exit 1
fi
NODE_VER=$(node --version)
echo "[VeriPaper Setup] Found Node.js: $NODE_VER"

# 3. Check npm
if ! command -v npm &> /dev/null; then
    echo "[Error] npm was not found. Please ensure Node.js is correctly installed with npm."
    exit 1
fi
NPM_VER=$(npm --version)
echo "[VeriPaper Setup] Found npm version: $NPM_VER"

# 4. Backend Setup
echo ""
echo "[VeriPaper Setup] Setting up backend environment..."
if [ ! -d ".venv" ]; then
    echo "[VeriPaper Setup] Creating virtual environment (.venv)..."
    $PYTHON_CMD -m venv --system-site-packages .venv
    if [ $? -ne 0 ]; then
        echo "[Error] Failed to create virtual environment."
        exit 1
    fi
else
    echo "[VeriPaper Setup] Virtual environment (.venv) already exists."
fi

echo "[VeriPaper Setup] Installing Python dependencies (this may take a few minutes)..."
source .venv/bin/activate
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[Error] Failed to install Python dependencies. Please check the error above."
    exit 1
fi

# 5. Check Model Weights
echo ""
echo "[VeriPaper Setup] Checking model weights..."
if [ ! -f "models/roberta-sentence-academic-v2/model.safetensors" ]; then
    echo "[VeriPaper Setup] Note: Local model weights not found at models/roberta-sentence-academic-v2/model.safetensors."
    echo "[VeriPaper Setup] The app will run in SIMULATION MODE unless you download and place the weights there."
else
    echo "[VeriPaper Setup] Model weights found successfully."
fi

# 6. Frontend Setup
echo ""
echo "[VeriPaper Setup] Setting up frontend..."
cd frontend
echo "[VeriPaper Setup] Installing frontend dependencies (this may take a few minutes)..."
npm install
if [ $? -ne 0 ]; then
    echo "[Error] Failed to install frontend dependencies. Please check the error above."
    cd ..
    exit 1
fi
cd ..

# 7. Make script files executable
chmod +x start_backend.sh start_frontend.sh stop_backend.sh stop_frontend.sh start_all.sh stop_all.sh install.sh 2>/dev/null

echo ""
echo "==================================================="
echo "  Setup complete!"
echo "==================================================="
echo "  To start the app:"
echo "    1. Run ./start_backend.sh"
echo "    2. Run ./start_frontend.sh"
echo "    (Or run ./start_all.sh)"
echo ""
echo "  To stop the app:"
echo "    Run ./stop_backend.sh and ./stop_frontend.sh (or ./stop_all.sh)"
echo ""
echo "  The app will be available at http://localhost:3000 once both are running."
echo "==================================================="
