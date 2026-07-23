@echo off
cd /d "%~dp0"

echo ==========================================
echo  Starting VeriPaper AI Forensic Services
echo ==========================================
echo.

call start_backend.bat
if errorlevel 1 (
    echo [VeriPaper All] Failed to start backend. Aborting startup.
    exit /b 1
)

echo.
call start_frontend.bat
if errorlevel 1 (
    echo [VeriPaper All] Failed to start frontend.
    exit /b 1
)

echo.
echo ==========================================
echo  All services started successfully!
echo.
echo  - Frontend: http://localhost:3000
echo  - Backend API: http://localhost:8000
echo  - Stop: Double-click stop_all.bat
echo ==========================================
