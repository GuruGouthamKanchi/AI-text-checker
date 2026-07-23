@echo off
cd /d "%~dp0"

echo ===================================================
echo  VeriPaper AI Forensic Tool - Setup Installer
echo ===================================================
echo.

echo [VeriPaper Setup] Checking prerequisites...

:: 1. Check Python 3.9+
set PYTHON_CMD=
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if %errorlevel%==0 (
    set PYTHON_CMD=python
) else (
    py -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
    if %errorlevel%==0 (
        set PYTHON_CMD=py
    )
)

if not defined PYTHON_CMD (
    echo [Error] Python 3.9 or later was not found on your PATH.
    echo Please install Python 3.9 or later from https://www.python.org/downloads/
    echo and ensure "Add Python to PATH" is checked during installation, then re-run this script.
    exit /b 1
)

for /f "delims=" %%v in ('%PYTHON_CMD% --version') do echo [VeriPaper Setup] Found %%v

:: 2. Check Node.js
set NODE_VER=
for /f "delims=" %%v in ('node --version 2^>nul') do set NODE_VER=%%v
if not defined NODE_VER (
    echo [Error] Node.js was not found on your PATH.
    echo Please install Node.js from https://nodejs.org/ - LTS version recommended.
    echo and ensure it is added to your PATH, then re-run this script.
    exit /b 1
)
echo [VeriPaper Setup] Found Node.js: %NODE_VER%

:: 3. Check npm
set NPM_VER=
for /f "delims=" %%v in ('npm --version 2^>nul') do set NPM_VER=%%v
if not defined NPM_VER (
    echo [Error] npm was not found. Please ensure Node.js is correctly installed with npm.
    exit /b 1
)
echo [VeriPaper Setup] Found npm version: %NPM_VER%

:: 4. Backend Setup
echo.
echo [VeriPaper Setup] Setting up backend environment...
if not exist .venv (
    echo [VeriPaper Setup] Creating virtual environment .venv...
    %PYTHON_CMD% -m venv --system-site-packages .venv
    if %errorlevel% neq 0 (
        echo [Error] Failed to create virtual environment.
        exit /b 1
    )
) else (
    echo [VeriPaper Setup] Virtual environment .venv already exists.
)

echo [VeriPaper Setup] Installing Python dependencies...
call .venv\Scripts\activate
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [Error] Failed to install Python dependencies. Please check the error above.
    exit /b 1
)

:: 5. Check Model Weights
echo.
echo [VeriPaper Setup] Checking model weights...
if not exist models\roberta-sentence-academic-v2\model.safetensors (
    echo [VeriPaper Setup] Note: Local model weights not found at models\roberta-sentence-academic-v2\model.safetensors.
    echo [VeriPaper Setup] The app will run in SIMULATION MODE unless you download and place the weights there.
) else (
    echo [VeriPaper Setup] Model weights found successfully.
)

:: 6. Frontend Setup
echo.
echo [VeriPaper Setup] Setting up frontend...
cd frontend
echo [VeriPaper Setup] Installing frontend dependencies...
call npm install
if %errorlevel% neq 0 (
    echo [Error] Failed to install frontend dependencies. Please check the error above.
    cd ..
    exit /b 1
)
cd ..

echo.
echo ===================================================
echo  Setup complete!
echo ===================================================
echo  To start the app:
echo    1. Run start_backend.bat
echo    2. Run start_frontend.bat
echo    (Or double-click start_all.bat)
echo.
echo  To stop the app:
echo    Run stop_backend.bat and stop_frontend.bat (or stop_all.bat)
echo.
echo  The app will be available at http://localhost:3000 once both are running.
echo ===================================================
