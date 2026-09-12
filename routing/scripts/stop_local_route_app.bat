@echo off
setlocal
cd /d "%~dp0\..\.."
if not exist routing\logs\application\server.pid (
  echo Local route application is not running.
  exit /b 0
)
set /p APP_PID=<routing\logs\application\server.pid
powershell -NoProfile -Command "Set-Content -LiteralPath 'routing\logs\application\stop.request' -Value 'graceful'; for ($i=0; $i -lt 60; $i++) { if (-not (Get-Process -Id %APP_PID% -ErrorAction SilentlyContinue)) { exit 0 }; Start-Sleep -Milliseconds 250 }; exit 1"
if errorlevel 1 (
  echo [WARN] Graceful shutdown timed out; stopping process %APP_PID%.
  powershell -NoProfile -Command "Stop-Process -Id %APP_PID% -ErrorAction Stop"
)
if errorlevel 1 (
  echo [ERROR] Could not stop local application.
  exit /b 1
)
echo Local route application stopped.
python routing\scripts\manage_local_model.py stop
endlocal
