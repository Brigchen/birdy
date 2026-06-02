@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PYTHONPATH=%~dp0;%~dp0birdy_runtime;%PYTHONPATH%"

if not exist "%~dp0birdy_runtime\gpx_track" (
    echo [BIRDY Track Map] Missing birdy_runtime. Run:
    echo   cd /d "%~dp0"
    echo   python sync_runtime.py
    pause
    exit /b 1
)

python -c "import PyQt5" 2>nul
if errorlevel 1 (
    echo [BIRDY Track Map] PyQt5 is required. Run:
    echo   cd /d "%~dp0"
    echo   python -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo Starting BIRDY Track Map...
python -m birdy_track_map
set "ERR=%ERRORLEVEL%"
if not "%ERR%"=="0" pause
exit /b %ERR%
