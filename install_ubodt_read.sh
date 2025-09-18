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

# Install cache manager script
ln -sf "$SCRIPT_DIR/ubodt_cache_manager" ~/.local/bin/ubodt_cache_manager

# Install daemon runner
ln -sf "$SCRIPT_DIR/build/ubodt_daemon_runner" ~/.local/bin/ubodt_daemon_runner

# Install client tool
ln -sf "$SCRIPT_DIR/build/ubodt_client" ~/.local/bin/ubodt_client

echo "Installation complete!"
echo "Available tools:"
echo "  ubodt_read          - Original UBODT reader"
echo "  ubodt_cache_manager - Cache manager (recommended)"
echo "  ubodt_client       - Direct daemon client"
echo "  ubodt_daemon_runner - Manual daemon runner"
echo ""
echo "Quick start:"
echo "  ubodt_cache_manager start <ubodt_file>"
echo "  ubodt_cache_manager status"
echo "  ubodt_cache_manager stop"