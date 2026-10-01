@echo off
setlocal
chcp 65001 >nul 2>&1
title FUSED - First Time Setup
cd /d "%~dp0"

set "PY_CMD="
where python >nul 2>&1
if %errorlevel%==0 set "PY_CMD=python"
if not defined PY_CMD (
    where py >nul 2>&1
    if %errorlevel%==0 set "PY_CMD=py -3"
)
if not defined PY_CMD (
    echo.
    echo Python 3.10+ was not found.
    echo Install Python 3.10+ and enable "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

%PY_CMD% -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 (
    echo Python 3.10 or newer is required.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating isolated FUSED environment...
    %PY_CMD% -m venv .venv
    if errorlevel 1 goto failed
)

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%VENV_PY%" goto failed

echo Installing FUSED core + UI dependencies...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel --disable-pip-version-check
if errorlevel 1 echo Pip upgrade skipped; continuing with the installed pip/setuptools.
"%VENV_PY%" -m pip install -e . --no-build-isolation --disable-pip-version-check
if errorlevel 1 goto failed
"%VENV_PY%" -m pip install -r app\requirements.txt -r requirements-dev.txt --disable-pip-version-check
if errorlevel 1 goto failed

echo.
echo ============================================================
echo FUSED setup completed.
echo Next step: double-click START_FUSED_UI.bat
echo ============================================================
pause
exit /b 0

:failed
echo.
echo FUSED setup failed. Read the error above.
pause
exit /b 1
