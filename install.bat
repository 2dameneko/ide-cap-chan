@echo off

set "VENV_DIR=%~dp0%venv"

dir "%VENV_DIR%\Scripts\Python.exe"
if %ERRORLEVEL% == 0 goto :activate

python -m venv --system-site-packages venv

:activate
call "%VENV_DIR%\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

md 2tag