@echo off
cd /d "%~dp0"
if not exist "var\evaluation" mkdir "var\evaluation"
python hybrid_cli.py --mode evaluate --dataset data/evaluation/v1.jsonl --output var/evaluation/latest.json
pause
