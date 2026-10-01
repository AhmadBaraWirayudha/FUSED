@echo off
cd /d "%~dp0"
if not exist "var\evaluation" mkdir "var\evaluation"
python hybrid_cli.py --mode evaluate-learning --dataset data/evaluation/learning_v1.jsonl --output var/evaluation/learning_latest.json
pause
