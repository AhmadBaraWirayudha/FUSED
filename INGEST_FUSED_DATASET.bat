@echo off
setlocal
chcp 65001 >nul 2>&1
title FUSED - Ingest Dataset
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo FUSED is not installed yet. Running setup first...
    call SETUP_FUSED.bat
    if errorlevel 1 exit /b 1
)

set /p dataset="Dataset path (.jsonl or .jsonl.gz): "
if "%dataset%"=="" exit /b 0
set /p db="Database path [data\supervised.db]: "
if "%db%"=="" set "db=data\supervised.db"

"%~dp0.venv\Scripts\python.exe" hybrid_cli.py --mode ingest-supervised --dataset "%dataset%" --db "%db%" --batch-size 500
if errorlevel 1 (
    echo.
    echo Ingestion failed. Read the error above.
    pause
    exit /b 1
)

echo.
echo Dataset ingestion completed.
pause
endlocal
