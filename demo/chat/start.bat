@echo off
setlocal
set PYTHONIOENCODING=utf-8

echo Starting Chat System
echo ==========================

:: Ensure logs directory exists
if not exist logs mkdir logs

:: Define ports
set BACKEND_PORT_1=8100
set BACKEND_PORT_2=8200
set FRONTEND_PORT=4000

:: Check and kill ports
call :KillPort %BACKEND_PORT_1%
call :KillPort %BACKEND_PORT_2%
call :KillPort %FRONTEND_PORT%

echo Cleaning old processes...
:: Use wmic to kill process by command line pattern if needed, but port killing is usually sufficient.
:: Or explicitly kill python/node if they are hanging without ports (less reliable on Windows without precise pid tracking)

echo Cleanup completed.
echo.

:: Start backend API
echo Starting backend API...
start /B "DeepAnalyze Backend" cmd /c "python backend.py > logs\backend.log 2>&1"
:: Windows doesn't easily give us the PID of the started background process without external tools or complex PowerShell.
:: We will rely on port checking or tasklist if needed, but for now simple start is fine.
echo Backend started in background.
echo API running on: http://localhost:8200
echo File service running on: http://localhost:8100

:: Wait for backend to initialize
timeout /t 3 /nobreak >nul

:: Start frontend
echo.
echo Starting React frontend...
cd frontend
start /B "DeepAnalyze Frontend" cmd /c "npm run dev -- -p %FRONTEND_PORT% > ..\logs\frontend.log 2>&1"
cd ..
echo Frontend started in background.
echo Frontend running on: http://localhost:%FRONTEND_PORT%

echo.
echo All services started successfully.
echo.
echo Service URLs:
echo   Backend API:  http://localhost:8200
echo   Frontend:     http://localhost:%FRONTEND_PORT%
echo   File Service: http://localhost:8100
echo.
echo Log files:
echo   Backend: logs\backend.log
echo   Frontend: logs\frontend.log
echo.
echo Stop services: run stop.bat
goto :eof

:KillPort
set port=%1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":"%port% ^| findstr "LISTENING"') do (
    echo Port %port% is in use by PID %%a. Killing...
    taskkill /F /PID %%a >nul 2>&1
)
goto :eof
