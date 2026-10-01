@echo off
setlocal
chcp 65001 >nul 2>&1
title FUSED - Generate 100M Token Dataset
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo FUSED is not installed yet. Running setup first...
    call SETUP_FUSED.bat
    if errorlevel 1 exit /b 1
)

if not exist "data\tb10" mkdir "data\tb10"
echo.
echo ============================================================
echo             FUSED 100M-TOKEN DATA GENERATOR
echo ============================================================
echo This creates about 100 million estimated tokens.
echo The resulting .gz file is a stress-test dataset, not a Git file.
echo.
set /p confirm="Type YES to continue: "
if /I not "%confirm%"=="YES" exit /b 0

"%~dp0.venv\Scripts\python.exe" hybrid_cli.py --mode generate-max-supervised --dataset data\tb10\synthetic_supervised_max_100m.jsonl.gz --token-budget 100000000 --target-tokens-per-record 512 --seed 20261001 --train-ratio 0.8
if errorlevel 1 (
    echo.
    echo Generation failed. Read the error above.
    pause
    exit /b 1
)

echo.
echo 100M-token dataset generation completed.
pause
endlocal
