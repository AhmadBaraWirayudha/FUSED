@echo off
cd /d "%~dp0"
python hybrid_cli.py --mode index-status --config config.yaml
pause
