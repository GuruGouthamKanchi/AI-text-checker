@echo off
cd /d "%~dp0"

echo [VeriPaper Frontend] Checking port 3000...
powershell -Command "(Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue).OwningProcess" > .port3000.tmp 2>nul
set PORT_PID=
set /p PORT_PID=<.port3000.tmp
if exist .port3000.tmp del .port3000.tmp

if defined PORT_PID (
    echo Port 3000 is already in use by process PID %PORT_PID%.
    echo Frontend appears to be already running. Run stop_frontend.bat first.
    exit /b 1
)

cd frontend
if not exist node_modules (
    echo [VeriPaper Frontend] Installing dependencies...
    call npm install
    if errorlevel 1 (
        echo Failed to install frontend dependencies. Check Node.js installation and network.
        exit /b 1
    )
)

echo [VeriPaper Frontend] Starting Next.js server on port 3000 in background...
start /B cmd /c "npm run dev > ..\frontend.log 2>&1"

cd ..

:: Loop to wait for Next.js server to bind to port 3000
set FRONTEND_PID=
set RETRIES=0
:loop_frontend
powershell -Command "(Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue).OwningProcess" > .port3000.tmp 2>nul
set FRONTEND_PID=
set /p FRONTEND_PID=<.port3000.tmp
if exist .port3000.tmp del .port3000.tmp

if defined FRONTEND_PID goto frontend_ok
set /a RETRIES+=1
if %RETRIES% geq 15 (
    echo [VeriPaper Frontend] Failed to start frontend or bind to port 3000.
    echo [VeriPaper Frontend] Please check frontend.log for details.
    exit /b 1
)
:: Use ping instead of timeout to avoid input redirection errors
ping 127.0.0.1 -n 3 > nul
goto loop_frontend

:frontend_ok
echo %FRONTEND_PID% > .frontend.pid
echo [VeriPaper Frontend] Frontend running at http://localhost:3000 (PID: %FRONTEND_PID%).
echo [VeriPaper Frontend] Run stop_frontend.bat to stop it.
