@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1
title FUSED - One Click Launcher
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    set "PY_CMD="
    where python >nul 2>&1
    if %errorlevel%==0 set "PY_CMD=python"
    if not defined PY_CMD (
        where py >nul 2>&1
        if %errorlevel%==0 set "PY_CMD=py -3"
    )
    if not defined PY_CMD (
        echo Python 3.10+ was not found. Install Python and run this file again.
        pause
        exit /b 1
    )
    echo First run detected. Installing FUSED...
    call SETUP_FUSED.bat
    if errorlevel 1 exit /b 1
)

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
"%VENV_PY%" -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Streamlit UI dependency is missing. Repairing the environment...
    call SETUP_FUSED.bat
    if errorlevel 1 exit /b 1
)

:menu
cls
echo ============================================================
echo                  FUSED - ONE CLICK LAUNCHER
echo ============================================================
echo.
echo  1. Open FUSED web UI
echo 2. Show live demo
echo 3. Ask one question
echo 4. Teach one answer
echo 5. Fixed benchmark
echo 6. Learning benchmark
echo 7. Generate 10,000 cases
echo 8. Generate 100M-token stress data
echo 9. Ingest a dataset
echo 10. Check retrieval index
echo 11. Rebuild retrieval index
echo 12. Install Semantic RAG + FAISS
echo 13. Run targeted tests
echo 14. Run all safe checks
echo 15. Run full test command
echo 16. Open beginner tutorial
echo 17. Open zero-to-benchmark tutorial
echo 18. Check installation readiness
echo 19. Exit
echo.
set /p choice="Choose 1-19: "

if "!choice!"=="1" goto ui
if "!choice!"=="2" goto demo
if "!choice!"=="3" goto ask
if "!choice!"=="4" goto teach
if "!choice!"=="5" goto eval
if "!choice!"=="6" goto evallearning
if "!choice!"=="7" goto gen10k
if "!choice!"=="8" goto gen100m
if "!choice!"=="9" goto ingest
if "!choice!"=="10" goto indexstatus
if "!choice!"=="11" goto rebuildindex
if "!choice!"=="12" goto installsemantic
if "!choice!"=="13" goto targeted
if "!choice!"=="14" goto allchecks
if "!choice!"=="15" goto release
if "!choice!"=="16" goto tutorial
if "!choice!"=="17" goto zero
if "!choice!"=="18" goto ready
if "!choice!"=="19" goto end
goto menu

:ui
"%VENV_PY%" -m streamlit run app\streamlit_app.py
pause
goto menu

:demo
"%VENV_PY%" demo_showcase.py --pause
pause
goto menu

:ask
set "question="
set /p question="Type your question: "
if "!question!"=="" goto menu
"%VENV_PY%" hybrid_cli.py --mode pipeline --text "!question!"
pause
goto menu

:teach
set "question="
set "answer="
set /p question="Question to teach: "
set /p answer="Correct answer: "
if "!question!"=="" goto menu
if "!answer!"=="" goto menu
"%VENV_PY%" hybrid_cli.py --mode teach --text "!question!" --ideal-output "!answer!"
pause
goto menu

:eval
if not exist "var\evaluation" mkdir "var\evaluation"
"%VENV_PY%" hybrid_cli.py --mode evaluate --dataset data\evaluation\v1.jsonl --output var\evaluation\latest.json
pause
goto menu

:evallearning
if not exist "var\evaluation" mkdir "var\evaluation"
"%VENV_PY%" hybrid_cli.py --mode evaluate-learning --dataset data\evaluation\learning_v1.jsonl --output var\evaluation\learning_latest.json
pause
goto menu

:gen10k
if not exist "var\generated" mkdir "var\generated"
"%VENV_PY%" hybrid_cli.py --mode generate-supervised --dataset var\generated\synthetic_10000.jsonl --count 10000 --seed 42 --train-ratio 0.8 --target-tokens-per-record 32
pause
goto menu

:gen100m
cls
echo This creates about 100 million estimated tokens.
echo The archive already contains a reference TB10 corpus.
echo.
set /p confirm="Type YES to generate a new large corpus: "
if /I not "!confirm!"=="YES" goto menu
if not exist "var\generated" mkdir "var\generated"
"%VENV_PY%" hybrid_cli.py --mode generate-max-supervised --dataset var\generated\synthetic_max_100m.jsonl.gz --token-budget 100000000 --target-tokens-per-record 512 --seed 20261001 --train-ratio 0.8
pause
goto menu

:ingest
set "dataset="
set "db="
set /p dataset="Dataset path (.jsonl or .jsonl.gz): "
if "!dataset!"=="" goto menu
set /p db="Database path [data\supervised.db]: "
if "!db!"=="" set "db=data\supervised.db"
"%VENV_PY%" hybrid_cli.py --mode ingest-supervised --dataset "!dataset!" --db "!db!" --batch-size 500
pause
goto menu

:indexstatus
set "db="
set /p db="Database path [data\engine.db]: "
if "!db!"=="" set "db=data\engine.db"
"%VENV_PY%" hybrid_cli.py --mode index-status --config config.yaml --db "!db!"
pause
goto menu

:rebuildindex
set "db="
set /p db="Database path [data\engine.db]: "
if "!db!"=="" set "db=data\engine.db"
"%VENV_PY%" hybrid_cli.py --mode rebuild-index --config config.yaml --db "!db!"
pause
goto menu

:installsemantic
"%VENV_PY%" -m pip install -e ".[semantic,faiss]" --no-build-isolation
pause
goto menu

:targeted
"%VENV_PY%" -m pytest -q tests/test_persistent_index.py tests/test_large_data.py tests/test_cli.py tests/test_evaluation.py tests/test_fractal_routing.py tests/test_language_model_interface.py tests/test_gemini_backend.py tests/test_learning_evaluation.py tests/test_synthetic_data.py tests/test_tb12_ui.py
pause
goto menu

:allchecks
call RUN_FUSED_ALL_CHECKS.bat
goto menu

:release
"%VENV_PY%" verify_release.py
pause
goto menu

:alltests
"%VENV_PY%" run_tests.py
pause
goto menu

:tutorial
start "" notepad "%~dp0docs\BEGINNER_TUTORIAL.md"
pause
goto menu

:zero
start "" notepad "%~dp0docs\TB12_ZERO_TO_BENCHMARK.md"
pause
goto menu

:ready
call CHECK_FUSED_READY.bat
goto menu

:end
endlocal
exit /b 0
