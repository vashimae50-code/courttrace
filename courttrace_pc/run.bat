@echo off
cd /d "%~dp0"
if not exist venv (echo 先に setup.bat を実行してください & pause & exit /b 1)
echo === コートトレース PC解析（YOLO + ByteTrack、CPUで実行） ===
set /p VIDEO=動画ファイルのパス（ドラッグ＆ドロップ可）: 
set /p PROJ=Webアプリで保存したJSONのパス（ドラッグ＆ドロップ可）: 
set VIDEO=%VIDEO:"=%
set PROJ=%PROJ:"=%
set OUT=%PROJ:.json=%.tracked.json
echo 出力: %OUT%
venv\Scripts\python track_yolo.py --video "%VIDEO%" --project "%PROJ%" --out "%OUT%" --stride 3 --imgsz 1280
echo.
echo 終了しました。Webアプリの「プロジェクト読込」で %OUT% を開いてください。
pause
