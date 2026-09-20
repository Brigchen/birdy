@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PYTHONPATH=%~dp0;%PYTHONPATH%"

python -c "import PyQt5, cv2, numpy, PIL, matplotlib" 2>nul
if errorlevel 1 (
    echo [MTF-meter] Missing dependencies. Run:
    echo   cd /d "%~dp0"
    echo   python -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo Starting MTF-meter...
python -m mtf_meter
set "ERR=%ERRORLEVEL%"
if not "%ERR%"=="0" pause
exit /b %ERR%
