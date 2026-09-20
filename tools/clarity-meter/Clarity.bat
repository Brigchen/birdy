@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PYTHONPATH=%~dp0;%PYTHONPATH%"

python -c "import PyQt5, cv2, numpy" 2>nul
if errorlevel 1 (
    echo [clarity-meter] Missing dependencies. Run:
    echo   cd /d "%~dp0"
    echo   python -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo Starting clarity-meter...
python -m clarity_meter %*
set "ERR=%ERRORLEVEL%"
if not "%ERR%"=="0" pause
exit /b %ERR%
