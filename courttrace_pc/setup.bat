@echo off
cd /d "%~dp0"
echo === コートトレース PC解析: 初回セットアップ（数分かかります） ===
where python >nul 2>nul || (echo Python が見つかりません。https://www.python.org/ からインストールしてください & pause & exit /b 1)
if not exist venv python -m venv venv
venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\python -m pip install ultralytics opencv-python numpy
venv\Scripts\python -m pip install "git+https://github.com/facebookresearch/sam2.git"
echo.
echo セットアップ完了。run.bat で解析を実行できます。
pause
