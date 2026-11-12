@echo off
rem Double-click to start the backend and frontend (runs start-dev.ps1).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-dev.ps1"
pause
