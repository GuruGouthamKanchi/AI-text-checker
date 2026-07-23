@echo off
cd /d "%~dp0"

set STOPPED=0

if exist .frontend.pid (
    set /p PID=<.frontend.pid
    if defined PID (
        set PID=%PID: =%
        tasklist /FI "PID eq %PID%" | findstr /I %PID% > nul
        if not errorlevel 1 (
            echo [VeriPaper Frontend] Stopping process PID %PID%...
            taskkill /PID %PID% /F > nul 2>&1
            set STOPPED=1
        ) else (
            echo [VeriPaper Frontend] PID file found but process %PID% is not running.
        )
    )
    del .frontend.pid
) else (
    echo [VeriPaper Frontend] No .frontend.pid file found. Checking port 3000...
)

:: Safety net: check if anything is still listening on port 3000
powershell -Command "(Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue).OwningProcess" > .port3000.tmp 2>nul
set PORT_PID=
set /p PORT_PID=<.port3000.tmp
if exist .port3000.tmp del .port3000.tmp

if defined PORT_PID (
    echo [VeriPaper Frontend] Found process PID %PORT_PID% listening on port 3000. Forcefully stopping...
    taskkill /PID %PORT_PID% /F > nul 2>&1
    set STOPPED=1
)

if %STOPPED%==1 (
    echo [VeriPaper Frontend] Frontend stopped successfully.
) else (
    echo [VeriPaper Frontend] Frontend doesn't appear to be running.
)
