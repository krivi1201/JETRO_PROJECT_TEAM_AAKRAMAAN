@echo off
cd /d "%~dp0"
echo Installing requirements (first run only takes a few minutes)...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Installation failed. Check that Python 3.9+ is installed and on PATH.
  pause
  exit /b 1
)
echo.
echo Setup complete. Start the dashboard with start_dashboard.bat
pause
