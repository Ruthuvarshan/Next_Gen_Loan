@echo off
SETLOCAL EnableDelayedExpansion

REM Force use of Java 17 instead of Java 25
set "JAVA_HOME=C:\Program Files\Java\jdk-17"
set "PATH=%JAVA_HOME%\bin;%PATH%"

REM Unset SPARK_HOME to use PySpark's bundled Spark (avoids version conflicts)
set SPARK_HOME=

REM Set Hadoop username to avoid Java compatibility issues
set HADOOP_USER_NAME=%USERNAME%

REM Set Java compatibility options for Java 17
set "_JAVA_OPTIONS=--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/sun.nio.cs=ALL-UNNAMED --add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/sun.util.calendar=ALL-UNNAMED --add-opens=java.base/jdk.internal.ref=ALL-UNNAMED"

set "SPARK_SUBMIT_OPTS=--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/sun.nio.cs=ALL-UNNAMED --add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/sun.util.calendar=ALL-UNNAMED --add-opens=java.base/jdk.internal.ref=ALL-UNNAMED"

echo ===============================================================================
echo               SPARK UI DEMO - Credit Risk Model Training
echo ===============================================================================
echo.
echo This script will:
echo  1. Activate Python virtual environment
echo  2. Generate sample data (if needed)
echo  3. Run Spark training with UI monitoring
echo.
echo Spark UI will be available at: http://localhost:4040
echo.
echo Note: Using Java 17 from %JAVA_HOME%
echo Note: Using Hadoop user: %HADOOP_USER_NAME%
echo ===============================================================================
echo.

REM Change to project directory
cd /d "%~dp0"

REM Activate virtual environment
echo [1/3] Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo OK: Virtual environment activated
echo.

REM Check if sample data exists
echo [2/3] Checking for sample data...
if not exist "data\sample\loan_data_sample.csv" (
    echo Sample data not found. Generating...
    python scripts\generate_sample_data.py
    if errorlevel 1 (
        echo ERROR: Failed to generate sample data
        pause
        exit /b 1
    )
) else (
    echo OK: Sample data already exists
)
echo.

REM Run Spark UI demo
echo [3/3] Starting Spark UI Demo...
echo.
echo ===============================================================================
echo   Spark UI will open at: http://localhost:4040
echo   Press Ctrl+C to stop the demo
echo ===============================================================================
echo.
python scripts\spark_ui_demo.py

echo.
echo ===============================================================================
echo Demo completed or interrupted
echo ===============================================================================
pause
