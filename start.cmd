@echo off
echo ====================================
echo Next-Gen Loan Origination System
echo Two-Window Architecture
echo ====================================
echo.

REM Check if databases exist
if not exist "data\users.db" (
    echo [1/4] Seeding database with demo users...
    python scripts\seed_users.py
    if errorlevel 1 (
        echo ERROR: Failed to seed database
        pause
        exit /b 1
    )
) else (
    echo [1/4] Database already seeded (skipping)
)

echo.
echo [2/4] Starting FastAPI Backend on port 8000...
echo.
start "FastAPI Backend" cmd /k "uvicorn src.api.main:app --reload --port 8000"

timeout /t 3 /nobreak >nul

echo [3/4] Starting User Portal on port 3000...
echo.
start "User Portal" cmd /k "cd user-portal && npm run dev"

timeout /t 2 /nobreak >nul

echo.
echo [4/4] All services started!
echo.
echo ====================================
echo   ACCESS YOUR APPLICATIONS
echo ====================================
echo.
echo Backend API (Swagger Docs):
echo   http://localhost:8000/docs
echo.
echo User Portal (Loan Officers):
echo   http://localhost:3000
echo   Username: loan_officer
echo   Password: officer123
echo.
echo Admin Dashboard (Coming Soon):
echo   http://localhost:3001
echo   Username: admin
echo   Password: admin123
echo.
echo ====================================
echo Press any key to stop all services...
pause >nul

taskkill /FI "WindowTitle eq FastAPI Backend*" /T /F
taskkill /FI "WindowTitle eq User Portal*" /T /F
taskkill /FI "WindowTitle eq Admin Dashboard*" /T /F

echo.
echo All services stopped.
pause
