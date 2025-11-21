@echo off
echo ====================================
echo Restarting FastAPI Backend
echo ====================================
echo.
echo Killing existing backend...
taskkill /FI "WindowTitle eq FastAPI Backend*" /T /F 2>nul
timeout /t 2 /nobreak >nul

echo.
echo Starting backend with new trained model...
cd /d r:\SSF\Next_Gen_Loan
start "FastAPI Backend" cmd /k ".venv\Scripts\activate.bat && uvicorn src.api.main:app --reload --port 8000"

timeout /t 3 /nobreak >nul
echo.
echo ====================================
echo Backend restarted successfully!
echo ====================================
echo.
echo API Documentation: http://localhost:8000/docs
echo Health Check: http://localhost:8000/health
echo.
pause
