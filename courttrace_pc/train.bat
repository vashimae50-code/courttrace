@echo off
setlocal
cd /d "%~dp0"
if not "%CT_KEEP%"=="1" (
  set CT_KEEP=1
  start "CourtTrace training" cmd /k ""%~f0" %*"
  exit /b
)
echo ==== CourtTrace training ====
echo Folder: %CD%
echo Log   : %CD%\train_log.txt
if "%~1"=="" (
  echo.
  echo Drop a labels ZIP file - courttrace_labels_XXXX.zip - onto train.bat, or run:
  echo    train.bat labels.zip
  echo.
  pause
  exit /b 1
)
if not exist "venv\Scripts\python.exe" (
  echo venv not found. Run setup.bat first.
  pause
  exit /b 1
)
"venv\Scripts\python.exe" -u train_players.py %* --epochs 60 --imgsz 960
echo.
echo ==== finished - exit code %ERRORLEVEL% - details in train_log.txt ====
pause
