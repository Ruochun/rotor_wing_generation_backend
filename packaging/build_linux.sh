#!/bin/bash
# Build script for Linux executables
# This script builds all three executables using PyInstaller

set -e  # Exit on any error

echo "========================================"
echo "Building Rotor Wing Generation Executables"
echo "========================================"
echo ""

# Check if pyinstaller is installed
if ! python3 -c "import PyInstaller" &> /dev/null; then
    echo "ERROR: PyInstaller is not installed!"
    echo "Please run: pip install pyinstaller"
    exit 1
fi

# Navigate to packaging directory
cd "$(dirname "$0")"

echo "Building generate_params..."
pyinstaller --clean --noconfirm generate_params.spec
echo ""

echo "Building generate_wing..."
pyinstaller --clean --noconfirm generate_wing.spec
echo ""

echo "Building analysis..."
pyinstaller --clean --noconfirm analysis.spec
echo ""

echo "========================================"
echo "Build completed successfully!"
echo "========================================"
echo ""
echo "Executables are located in: dist/"
echo "  - generate_params"
echo "  - generate_wing"
echo "  - analysis"
echo ""
echo "You can now distribute these executables to users."
echo "Users do NOT need to install Python or any dependencies."
