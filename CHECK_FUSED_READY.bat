@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo FUSED is not installed. Double-click SETUP_FUSED.bat first.
    pause
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -c "from app_streamlit_core import readiness; import json, pathlib; print(json.dumps(readiness(pathlib.Path('.').resolve()), indent=2))"
pause
