@echo off
setlocal
cd /d "%~dp0"
set /p question="Type your question: "
if "%question%"=="" set "question=Solve the integral of 2x from 0 to 4."
python hybrid_cli.py --mode pipeline --text "%question%"
pause
endlocal
