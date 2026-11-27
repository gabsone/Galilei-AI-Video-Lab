@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
)
.venv\Scripts\pip.exe install --upgrade pip
.venv\Scripts\pip.exe install -r requirements.txt pyinstaller

.venv\Scripts\pyinstaller.exe ^
    --noconfirm ^
    --clean ^
    --onedir ^
    --name "Galilei_Di_Palo_AI_Lab" ^
    --add-data "models;models" ^
    --add-data "assets;assets" ^
    --collect-all mediapipe ^
    main.py

echo.
echo Build completata in: dist\Galilei_Di_Palo_AI_Lab\Galilei_Di_Palo_AI_Lab.exe
pause
