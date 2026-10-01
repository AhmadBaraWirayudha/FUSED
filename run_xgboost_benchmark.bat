@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1
title XGBoost Benchmark: metatune vs Optuna
cd /d "%~dp0"

rem --- Locate a Python interpreter -------------------------------------------
set "PY_CMD="
where python >nul 2>&1
if %errorlevel%==0 set "PY_CMD=python"
if not defined PY_CMD (
    where python3 >nul 2>&1
    if %errorlevel%==0 set "PY_CMD=python3"
)
if not defined PY_CMD (
    where py >nul 2>&1
    if %errorlevel%==0 set "PY_CMD=py"
)
if not defined PY_CMD (
    echo.
    echo   Could not find Python on your PATH.
    echo   Install Python 3.10+ from https://www.python.org/downloads/
    echo   and make sure "Add python.exe to PATH" is checked during setup.
    echo.
    pause
    exit /b 1
)

if not exist "run_xgboost_benchmark.py" (
    echo.
    echo   run_xgboost_benchmark.py wasn't found next to this .bat file.
    echo   Make sure run_xgboost_benchmark.bat stays in the project's root folder.
    echo.
    pause
    exit /b 1
)

echo ============================================================
echo   XGBoost Benchmark -- metatune vs Optuna
echo   Real diabetes data + controllable synthetic data
echo   (using: !PY_CMD!)
echo ============================================================
echo.
echo   First run installs any missing benchmark dependencies
echo   (numpy, scipy, xgboost, optuna, scikit-learn, ...), then
echo   runs both problems ours vs Optuna and writes a report.
echo   Takes a few minutes.
echo.

!PY_CMD! run_xgboost_benchmark.py
if errorlevel 1 (
    echo.
    echo   Something went wrong -- see the error above.
    echo.
    pause
    exit /b 1
)

echo.
echo   Done. Opening the report...
if exist "XGBOOST_BENCHMARK_REPORT.md" (
    start "" notepad "XGBOOST_BENCHMARK_REPORT.md"
)
echo.
pause
endlocal
exit /b 0
