#!/bin/bash
# Install FMM tools to conda environment or local bin directory
# Usage: ./install_tools.sh

set -e

# Determine installation directory
if [ -n "$CONDA_PREFIX" ]; then
    INSTALL_DIR="$CONDA_PREFIX/bin"
    echo "Installing to conda environment: $INSTALL_DIR"
else
    INSTALL_DIR="$HOME/.local/bin"
    echo "Installing to user local bin: $INSTALL_DIR"
    mkdir -p "$INSTALL_DIR"
fi

# Check if build directory exists
if [ ! -d "build" ]; then
    echo "Error: build directory not found. Please run 'cmake' and 'make' first."
    exit 1
fi

# Copy executables
echo "Installing FMM tools..."
TOOLS="fmm cmm stmatch h3mm ubodt_gen ubodt_converter ubodt_daemon"

for tool in $TOOLS; do
    if [ -f "build/$tool" ]; then
        cp "build/$tool" "$INSTALL_DIR/"
        chmod +x "$INSTALL_DIR/$tool"
        echo "  ✓ Installed: $tool"
    else
        echo "  ⚠ Warning: $tool not found in build directory"
    fi
done

echo ""
echo "Installation complete!"
echo ""
echo "Installed tools:"
for tool in $TOOLS; do
    if [ -f "$INSTALL_DIR/$tool" ]; then
        echo "  - $tool"
    fi
done

echo ""
echo "You can now run these tools from any directory."
echo ""
echo "Example usage:"
echo "  ubodt_manager --help"
echo "  fmm --help"
echo "  cmm --help"
