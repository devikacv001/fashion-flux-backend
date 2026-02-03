@echo off
echo ==========================================
echo   Fashion Flux Backend Server
echo ==========================================
echo.

cd /d "%~dp0"

echo Starting server on http://127.0.0.1:8080
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn app:app --host 127.0.0.1 --port 8080

pause
