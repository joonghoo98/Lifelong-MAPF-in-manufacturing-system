@echo off
rem Double-click to watch the AGV simulation live. It asks for the map, vehicles and jobs per minute.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo .venv\Scripts\python.exe was not found next to this file: create the virtual environment first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m agvtb.tools.launch
echo.
pause
endlocal
