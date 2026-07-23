@echo off
cd /d "%~dp0"

echo [VeriPaper Backend] Checking port 8000...
powershell -Command "(Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue).OwningProcess" > .port8000.tmp 2>nul
set PORT_PID=
set /p PORT_PID=<.port8000.tmp
if exist .port8000.tmp del .port8000.tmp

if defined PORT_PID (
    echo Port 8000 is already in use by process PID %PORT_PID%.
    echo Backend appears to be already running. Run stop_backend.bat first.
    exit /b 1
)

if not exist .venv (
    echo [VeriPaper Backend] Creating virtual environment .venv...
    python -m venv --system-site-packages .venv
    if errorlevel 1 (
        echo Failed to create virtual environment. Make sure Python is installed and in your PATH.
        exit /b 1
    )
    echo [VeriPaper Backend] Activating environment and installing dependencies...
    call .venv\Scripts\activate
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install dependencies. Check your network connection.
        exit /b 1
    )
) else (
    echo [VeriPaper Backend] Activating existing virtual environment...
    call .venv\Scripts\activate
)

echo [VeriPaper Backend] Starting FastAPI server on port 8000 in background...
start /B cmd /c "python -u -m uvicorn src.app:app --host 127.0.0.1 --port 8000 > backend.log 2>&1"

:: Loop to wait for startup and capture the PID
set BACKEND_PID=
set RETRIES=0
:loop_backend
powershell -Command "(Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue).OwningProcess" > .port8000.tmp 2>nul
set BACKEND_PID=
set /p BACKEND_PID=<.port8000.tmp
if exist .port8000.tmp del .port8000.tmp

if defined BACKEND_PID goto backend_ok
set /a RETRIES+=1
if %RETRIES% geq 15 (
    echo [VeriPaper Backend] Failed to start backend or bind to port 8000.
    echo [VeriPaper Backend] Please check backend.log for details.
    exit /b 1
)
:: Use ping instead of timeout to avoid input redirection errors
ping 127.0.0.1 -n 3 > nul
goto loop_backend

:backend_ok
echo %BACKEND_PID% > .backend.pid
echo [VeriPaper Backend] Backend running at http://localhost:8000 (PID: %BACKEND_PID%).
echo [VeriPaper Backend] Run stop_backend.bat to stop it.
