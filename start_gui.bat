@echo off
setlocal
cd /d "%~dp0"

REM 观鸟记录 .xls 导出依赖（须与下方 python 为同一解释器）
python -c "import xlutils" 2>nul
if errorlevel 1 (
    echo [Birdy] Installing xlrd xlwt xlutils for record export...
    python -m pip install xlrd xlwt xlutils
    if errorlevel 1 (
        echo [Birdy] Failed to install Excel dependencies. Try:
        echo   python -m pip install -r requirements.txt
        pause
        exit /b 1
    )
)

cd src
echo Starting Birdy Skill GUI...
python birdy_gui.py
