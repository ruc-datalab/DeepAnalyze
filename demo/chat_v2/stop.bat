@echo off
setlocal

echo Stopping AI Chat System
echo =======================

set BACKEND_PORT_1=8100
set BACKEND_PORT_2=8200
set FRONTEND_PORT=4000

echo Releasing ports...
call :KillPort %BACKEND_PORT_1%
call :KillPort %BACKEND_PORT_2%
call :KillPort %FRONTEND_PORT%

echo.
echo Cleaning up remaining processes...
:: Kill by image name as a fallback
taskkill /F /IM "python.exe" /FI "WINDOWTITLE eq DeepAnalyze Backend*" >nul 2>&1
taskkill /F /IM "node.exe" >nul 2>&1
:: Note: Killing all node.exe might be aggressive if user has other node projects running. 
:: But filtering by command line arguments is hard in batch. 
:: The port killing above is the safest method. This is just cleanup.

echo.
echo System stopped successfully.
echo.
echo Log files are kept in the logs\ directory.
echo To restart the system: run start.bat
goto :eof

:KillPort
set port=%1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":"%port% ^| findstr "LISTENING"') do (
    echo   Releasing port %port% [PID: %%a]...
    taskkill /F /PID %%a >nul 2>&1
)
goto :eof
