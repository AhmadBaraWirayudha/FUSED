@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo FUSED is not installed. Double-click SETUP_FUSED.bat first.
    pause
    exit /b 1
)
set "PY=%~dp0.venv\Scripts\python.exe"
echo ============================================================
echo FUSED - ALL SAFE CHECKS
 echo Evaluation + learning + UI-targeted tests + 156 smoke checks.
echo No Gemini API call is required.
echo ============================================================
if not exist "var\evaluation" mkdir "var\evaluation"
"%PY%" hybrid_cli.py --mode evaluate --dataset data\evaluation\v1.jsonl --output var\evaluation\latest.json
if errorlevel 1 goto failed
"%PY%" hybrid_cli.py --mode evaluate-learning --dataset data\evaluation\learning_v1.jsonl --output var\evaluation\learning_latest.json
if errorlevel 1 goto failed
"%PY%" -m pytest -q tests/test_persistent_index.py tests/test_large_data.py tests/test_cli.py tests/test_evaluation.py tests/test_fractal_routing.py tests/test_language_model_interface.py tests/test_gemini_backend.py tests/test_learning_evaluation.py tests/test_synthetic_data.py tests/test_tb12_ui.py
if errorlevel 1 goto failed
"%PY%" tests/test_smoke.py
if errorlevel 1 goto failed
echo.
echo ALL SAFE CHECKS PASSED.
pause
exit /b 0
:failed
echo.
echo ONE OR MORE CHECKS FAILED. Read the output above.
pause
exit /b 1
