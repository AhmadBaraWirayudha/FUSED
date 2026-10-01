@echo off
setlocal
chcp 65001 >nul 2>&1
title FUSED - AI Research Studio
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" goto setup
"%~dp0.venv\Scripts\python.exe" -c "import streamlit" >nul 2>&1
if errorlevel 1 goto setup
goto launch

:setup
echo FUSED UI is not installed or needs repair. Running setup...
call SETUP_FUSED.bat
if errorlevel 1 exit /b 1

:launch
"%~dp0.venv\Scripts\python.exe" -m streamlit run app\streamlit_app.py
endlocal
