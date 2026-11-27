@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [AI Hand Lab] Creazione ambiente virtuale Python...
    python -m venv .venv
    if errorlevel 1 (
        echo Errore: assicurati che Python 3.10-3.12 sia installato e nel PATH.
        pause
        exit /b 1
    )
    echo [AI Hand Lab] Installazione librerie richieste...
    .venv\Scripts\pip.exe install --upgrade pip
    .venv\Scripts\pip.exe install -r requirements.txt
)

.venv\Scripts\python.exe main.py --fullscreen %*
