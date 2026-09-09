@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0code"
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -m streamlit run 09_universal_system.py
) else if exist "%~dp0env_bishe\python.exe" (
    "%~dp0env_bishe\python.exe" -m streamlit run 09_universal_system.py
) else (
    python -m streamlit run 09_universal_system.py
)
if errorlevel 1 pause
