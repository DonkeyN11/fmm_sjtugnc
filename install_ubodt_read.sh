#!/bin/bash

# Installation script for UBODT Read tool
# This script makes ubodt_read available system-wide

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create symlink in /usr/local/bin
if [ "$EUID" -eq 0 ]; then
    # Running as root
    ln -sf "$SCRIPT_DIR/ubodt_read" /usr/local/bin/ubodt_read
    echo "ubodt_read has been installed system-wide in /usr/local/bin/"
else
    # Running as regular user, install in ~/.local/bin
    mkdir -p ~/.local/bin
    ln -sf "$SCRIPT_DIR/ubodt_read" ~/.local/bin/ubodt_read

    # Check if ~/.local/bin is in PATH
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        echo "Adding ~/.local/bin to PATH..."
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        echo "Please run 'source ~/.bashrc' or restart your terminal to use ubodt_read from anywhere."
    else
        echo "ubodt_read has been installed in ~/.local/bin/"
    fi
fi

echo "Installation complete! You can now run 'ubodt_read' from any directory."