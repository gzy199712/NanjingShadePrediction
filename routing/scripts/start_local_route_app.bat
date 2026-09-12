@echo off
setlocal
cd /d "%~dp0\..\.."
python routing\scripts\manage_local_model.py start
if errorlevel 1 (
  echo [ERROR] Local Ollama/Qwen3 service could not be started.
  exit /b 1
)
python -m workflows.routing_navigation check
if errorlevel 1 (
  echo [ERROR] Frozen inputs or local application checks failed.
  exit /b 1
)
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue) { exit 1 }"
if errorlevel 1 (
  echo [ERROR] Port 8765 is already in use.
  exit /b 1
)
echo Local application: http://127.0.0.1:8765
echo The service is local-only. No firewall rule will be created.
python routing\app\backend\run_local_server.py
endlocal
