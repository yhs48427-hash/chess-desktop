@echo off
rem Chess win-probability desktop app. Double-click to start; close this window to stop.
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set "PYEXE="
set "PYARG="

if exist "%~dp0.venv\Scripts\python.exe" set "PYEXE=%~dp0.venv\Scripts\python.exe"
if not defined PYEXE (
  py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1 && (set "PYEXE=py" & set "PYARG=-3")
)
if not defined PYEXE (
  python -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1 && set "PYEXE=python"
)
if not defined PYEXE (
  if exist "%USERPROFILE%\anaconda3\python.exe" set "PYEXE=%USERPROFILE%\anaconda3\python.exe"
)
if not defined PYEXE (
  echo Python 3.9+ was not found.
  echo Install Python from https://www.python.org/downloads/ ^(check "Add python.exe to PATH"^) and run this again.
  pause
  exit /b 1
)

"%PYEXE%" %PYARG% start.py %*
if errorlevel 1 pause
endlocal
