@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 goto failed
)
fc /b "requirements.txt" ".venv\requirements-installed.txt" >nul 2>nul
if errorlevel 1 (
    ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
    if errorlevel 1 goto failed
    copy /y "requirements.txt" ".venv\requirements-installed.txt" >nul
)
if not exist ".env" copy /y ".env.example" ".env" >nul
".venv\Scripts\python.exe" bot.py
if errorlevel 1 goto failed
exit /b 0
:failed
echo.
echo Setup or startup failed. See the message above and README.md.
pause
exit /b 1
