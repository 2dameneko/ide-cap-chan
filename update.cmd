@echo off

REM Pull the latest changes from the repository
git pull

REM Set default values for environment variables if not already defined
if not defined PYTHON (set PYTHON=python)
if defined GIT (set "GIT_PYTHON_GIT_EXECUTABLE=%GIT%")
if not defined VENV_DIR (set "VENV_DIR=%~dp0venv")

REM Check if the virtual environment exists
if exist "%VENV_DIR%\Scripts\Python.exe" (
    REM Activate the virtual environment
    call "%VENV_DIR%\Scripts\activate.bat"
    set PYTHON="%VENV_DIR%\Scripts\Python.exe"
    echo Virtual environment activated: %PYTHON%
) else (
    echo Virtual environment not found at %VENV_DIR%. Please create it first.
    exit /b 1
)

REM Install project requirements
pip install -r requirements.txt

REM End of script
echo Update complete.