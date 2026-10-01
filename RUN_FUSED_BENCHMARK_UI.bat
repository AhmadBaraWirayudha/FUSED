@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    call SETUP_FUSED.bat
    if errorlevel 1 exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -m streamlit run app\streamlit_app.py
endlocal
