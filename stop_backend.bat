@echo off
cd /d "%~dp0"

set STOPPED=0

if exist .backend.pid (
    set /p PID=<.backend.pid
    if defined PID (
        set PID=%PID: =%
        tasklist /FI "PID eq %PID%" | findstr /I "%PID%" > nul
        if not errorlevel 1 (
            echo [VeriPaper Backend] Stopping process PID %PID%...
            taskkill /PID %PID% /F > nul 2>&1
            set STOPPED=1
        ) else (
            echo [VeriPaper Backend] PID file found but process %PID% is not running.
        )
    )
    del .backend.pid
) else (
    echo [VeriPaper Backend] No .backend.pid file found. Checking port 8000...
)

:: Safety net: check if anything is still listening on port 8000
powershell -Command "(Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue).OwningProcess" > .port8000.tmp 2>nul
set PORT_PID=
set /p PORT_PID=<.port8000.tmp
if exist .port8000.tmp del .port8000.tmp

if defined PORT_PID (
    echo [VeriPaper Backend] Found process PID %PORT_PID% listening on port 8000. Forcefully stopping...
    taskkill /PID %PORT_PID% /F > nul 2>&1
    set STOPPED=1
)

if %STOPPED%==1 (
    echo [VeriPaper Backend] Backend stopped successfully.
) else (
    echo [VeriPaper Backend] Backend doesn't appear to be running.
)
