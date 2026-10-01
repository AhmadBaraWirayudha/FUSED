@echo off
cd /d "%~dp0"
python hybrid_cli.py --mode rebuild-index --config config.yaml
pause
