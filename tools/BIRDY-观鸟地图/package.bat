@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo Syncing birdy_runtime...
python sync_runtime.py
if errorlevel 1 exit /b 1

echo.
echo Ready to zip and share this folder:
echo   %~dp0
echo.
pause
