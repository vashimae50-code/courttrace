@echo off
cd /d "%~dp0"
start "" http://localhost:8765/
python serve.py 8765
if errorlevel 1 ( echo Python 3 is required. Install it from https://www.python.org/ and tick "Add python.exe to PATH". & pause )
