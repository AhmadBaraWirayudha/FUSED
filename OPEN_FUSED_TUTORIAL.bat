@echo off
cd /d "%~dp0"
if exist "docs\BEGINNER_TUTORIAL.md" start "" notepad "docs\BEGINNER_TUTORIAL.md"
if not exist "docs\BEGINNER_TUTORIAL.md" echo Tutorial file not found.
pause
