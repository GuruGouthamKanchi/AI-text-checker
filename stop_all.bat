@echo off
cd /d "%~dp0"

echo ==========================================
echo  Stopping VeriPaper AI Forensic Services
echo ==========================================
echo.

call stop_frontend.bat
echo.
call stop_backend.bat

echo.
echo ==========================================
echo  All services stopped successfully!
echo ==========================================
