# Quick Start Guide for Packaging

This guide will help you quickly build and test the standalone executables.

## Prerequisites

Make sure you have:
- Python 3.7+ installed
- All dependencies installed

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Building Executables

### On Linux/macOS

```bash
cd packaging
./build_linux.sh
```

### On Windows

```cmd
cd packaging
build_windows.bat
```

## Testing the Executables

After building, test the executables:

```bash
cd packaging/dist

# Test 1: Generate parameters
./generate_params test_params.csv --n-wings 4 --overall-length 0.025

# Test 2: Generate wing geometry
./generate_wing test_params.csv --output test_wing.stl

# Test 3: Analyze performance
./analysis test_params.csv test_results.csv

# View results
cat test_results.csv
```

## Expected Output

After successful build:
- `dist/generate_params` (~7-8 MB)
- `dist/generate_wing` (~50-60 MB)
- `dist/analysis` (~7-8 MB)

## Distribution

Simply copy the executables from `dist/` folder and share them with users. They don't need Python or any dependencies installed!

## Example Workflow for End Users

End users who receive the executables can use them like this:

```bash
# Step 1: Generate parameters
./generate_params my_design.csv --n-wings 4 --max-twist-angle 45

# Step 2: Generate 3D model
./generate_wing my_design.csv --output my_wing.stl

# Step 3: Analyze performance
./analysis my_design.csv results.csv --rpm 3500
```

That's it! No Python, no pip, no dependencies needed.

## Troubleshooting

**Build fails with "PyInstaller not found"**
```bash
pip install pyinstaller
```

**Build fails with import errors**
```bash
pip install -r ../requirements.txt
```

**Executables don't run**
- On Linux: Make sure the file is executable: `chmod +x generate_params`
- Check that you're running on the same platform where you built (Linux binaries only work on Linux)

For more detailed information, see [README.md](README.md).
