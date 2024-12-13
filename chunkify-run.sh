#!/bin/bash

# Set the name of your virtual environment
VENV_NAME="chunkify_env"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Python is installed
if ! command_exists python3; then
    echo "Python 3 is not found. Please ensure Python 3 is installed and added to your PATH."
    exit 1
fi

# Check if the virtual environment exists, create if it doesn't
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating new virtual environment: $VENV_NAME"
    python3 -m venv "$VENV_NAME"
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Please check your Python installation."
        exit 1
    fi
else
    echo "Virtual environment $VENV_NAME already exists."
fi

# Activate the virtual environment
source "$VENV_NAME/bin/activate"

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "requirements.txt not found. Please create a requirements.txt file in the same directory as this script."
    exit 1
fi

# Upgrade pip to the latest version
python3 -m pip install --upgrade pip

# Install packages from requirements.txt
echo "Installing packages from requirements.txt..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install some packages. Please check your internet connection and requirements.txt file."
    exit 1
fi

# Determine the correct KoboldCPP binary based on the system
if [[ "$(uname)" == "Darwin" ]]; then
    if [[ "$(uname -m)" == "arm64" ]]; then
        KOBOLDCPP_BINARY="./koboldcpp-mac-arm64"
    else
        KOBOLDCPP_BINARY="./koboldcpp-mac-x64"
    fi
elif [[ "$(uname)" == "Linux" ]]; then
    KOBOLDCPP_BINARY="./koboldcpp-linux-x64"
else
    echo "Unsupported operating system. Please run on macOS or Linux."
    exit 1
fi

# Check if the KoboldCPP binary exists and is executable
if [ ! -x "$KOBOLDCPP_BINARY" ]; then
    echo "KoboldCPP binary not found or not executable. Please check the binary and its permissions."
    exit 1
fi

# Launch KoboldCPP
"$KOBOLDCPP_BINARY" --config chunkify.kcppt &

# Wait for KoboldCPP to start
echo "Waiting for KoboldCPP to start..."
while ! nc -z localhost 5001; do   
  sleep 1
done

# Launch the Python GUI script
python3 chunkify-gui.py

# Deactivate the virtual environment when the GUI is closed
deactivate

# Wait for user input before closing
read -p "Press Enter to exit..."
