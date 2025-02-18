@echo off

REM Set the directory for the virtual environment
set "VENV_DIR=%~dp0venv"

REM Check if the virtual environment already exists
if exist "%VENV_DIR%\Scripts\Python.exe" (
    echo Virtual environment already found at %VENV_DIR%.
    exit /b 1
) else (
    REM Create the virtual environment
    python -m venv --system-site-packages venv
    goto :activate
)

:activate
REM Activate the virtual environment
call "%VENV_DIR%\Scripts\activate.bat"

REM Upgrade pip
python -m pip install --upgrade pip

REM Install PyTorch and related packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM Install project requirements
pip install -r requirements.txt

REM End of script
echo Virtual environment setup complete.