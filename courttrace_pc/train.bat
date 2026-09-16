@echo off
cd /d %~dp0
if "%~1"=="" ( echo Usage: train.bat labels.zip [more.zip ...] & pause & exit /b 1 )
venv\Scripts\python train_players.py %* --epochs 60 --imgsz 960
pause
