@echo off
setlocal enabledelayedexpansion

REM Set the name of your virtual environment
set "VENV_NAME=chunkify_env"

REM Set the path to your Python installation (update this if needed)
set "PYTHON_PATH=python"

REM Check if Python is installed and in PATH
%PYTHON_PATH% --version >nul 2>&1
if errorlevel 1 (
    echo Python is not found. Please ensure Python is installed and added to your PATH.
    pause
    exit /b 1
)


REM Check if the virtual environment exists, create if it doesn't
if not exist "%VENV_NAME%\Scripts\activate.bat" (
    echo Creating new virtual environment: %VENV_NAME%
    %PYTHON_PATH% -m venv %VENV_NAME%
    if errorlevel 1 (
        echo Failed to create virtual environment. Please check your Python installation.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment %VENV_NAME% already exists.
)

REM Activate the virtual environment
call "%VENV_NAME%\Scripts\activate.bat"

REM Check if requirements.txt exists
if not exist requirements.txt (
    echo requirements.txt not found. Please create a requirements.txt file in the same directory as this script.
    pause
    exit /b 1
)

REM Upgrade pip to the latest version
python -m pip install --upgrade pip

REM Install packages from requirements.txt
echo Installing packages from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install some packages. Please check your internet connection and requirements.txt file.
    pause
    exit /b 1
)


REM Check if koboldcpp.exe exists, if not, check for koboldcpp_cu12.exe
if exist koboldcpp.exe (
    set "KOBOLD_EXE=koboldcpp.exe"
) else if exist koboldcpp_cu12.exe (
    set "KOBOLD_EXE=koboldcpp_cu12.exe"
) else (
    echo Neither koboldcpp.exe nor koboldcpp_cu12.exe found. Please ensure one of these files exists.
    pause
    exit /b 1
)

REM Launch your Python script
start %KOBOLD_EXE% --config chunkify.kcppt

python chunkify-gui.py
pause

REM Deactivate the virtual environment
deactivate

pause