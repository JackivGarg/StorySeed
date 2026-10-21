@echo off
"%~dp0.venv\Scripts\python.exe" "%~dp0scripts\shutdown_after_training.py" --cancel
pause
