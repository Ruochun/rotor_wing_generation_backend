# Packaging Guide

This directory contains the infrastructure to build standalone executables for the Rotor Wing Generation Backend tools.

## Overview

The packaging system uses [PyInstaller](https://pyinstaller.org/) to create standalone executables that bundle Python and all dependencies into single executable files. Users can run these executables without installing Python or any dependencies.

## Prerequisites

Before building executables, you need:

1. **Python 3.7+** installed on your system
2. **All runtime dependencies** installed:
   ```bash
   pip install -r requirements.txt
   ```
3. **PyInstaller** installed:
   ```bash
   pip install -r requirements-dev.txt
   ```

## Building Executables

### Windows

On Windows, run the batch script:

```cmd
cd packaging
build_windows.bat
```

This will create three executables in the `dist/` directory as folders (using --onedir mode for faster startup):
- `generate_params/` - contains `generate_params.exe` and dependencies
- `generate_wing/` - contains `generate_wing.exe` and dependencies
- `analysis/` - contains `analysis.exe` and dependencies

### Linux

On Linux, run the bash script:

```bash
cd packaging
./build_linux.sh
```

This will create three executables in the `dist/` directory as folders (using --onedir mode for faster startup):
- `generate_params/` - contains `generate_params` executable and dependencies
- `generate_wing/` - contains `generate_wing` executable and dependencies
- `analysis/` - contains `analysis` executable and dependencies

### macOS

The Linux build script also works on macOS:

```bash
cd packaging
./build_linux.sh
```

## What Gets Built

The build process creates three separate executables using --onedir mode for faster startup:

1. **generate_params** - Generates CSV parameter files from abstract design requirements
2. **generate_wing** - Generates 3D wing geometry (STL files) from parameter CSVs
3. **analysis** - Analyzes wing performance using Blade Element Momentum Theory

Each executable is created as a directory containing:
- Python interpreter
- All required Python packages (numpy, scipy, trimesh, manifold3d, networkx)
- The corresponding Python script
- The main executable file

The --onedir mode is used by default instead of --onefile because it provides faster startup times.

## Distribution

After building, you can distribute the executables to users. The executables work exactly like the Python scripts but don't require users to install anything.

### Windows Distribution

Share the folders from `dist/`:
- `generate_params/` folder (contains `generate_params.exe`)
- `generate_wing/` folder (contains `generate_wing.exe`)
- `analysis/` folder (contains `analysis.exe`)

Users can run the executables from the command line by navigating to each folder:
```cmd
cd generate_params
generate_params.exe input.csv --overall-length 0.03

cd ..\generate_wing
generate_wing.exe input.csv --output wing.stl

cd ..\analysis
analysis.exe input.csv results.csv --rpm 4000
```

Or by specifying the full path:
```cmd
dist\generate_params\generate_params.exe input.csv --overall-length 0.03
```

### Linux Distribution

Share the folders from `dist/`:
- `generate_params/` folder (contains `generate_params` executable)
- `generate_wing/` folder (contains `generate_wing` executable)
- `analysis/` folder (contains `analysis` executable)

Users can run the executables from the terminal by navigating to each folder:
```bash
cd generate_params
./generate_params input.csv --overall-length 0.03

cd ../generate_wing
./generate_wing input.csv --output wing.stl

cd ../analysis
./analysis input.csv results.csv --rpm 4000
```

Or by specifying the full path:
```bash
dist/generate_params/generate_params input.csv --overall-length 0.03
```

## File Structure

```
packaging/
├── README.md                 # This file
├── build_windows.bat         # Windows build script
├── build_linux.sh           # Linux/macOS build script
├── generate_params.spec     # PyInstaller spec for generate_params
├── generate_wing.spec       # PyInstaller spec for generate_wing
├── analysis.spec            # PyInstaller spec for analysis
├── build/                   # Temporary build files (gitignored)
└── dist/                    # Output executables (gitignored)
```

## Customizing the Build

### Modifying Spec Files

The `.spec` files control how PyInstaller builds the executables. You can customize:

- **Hidden imports**: Add modules that PyInstaller might miss
- **Data files**: Include additional files in the executable
- **Icons**: Set custom icons for Windows executables
- **Console mode**: Create GUI applications instead of console applications

Example: Adding a custom icon to `generate_params.spec`:

```python
exe = EXE(
    # ... other parameters ...
    icon='path/to/icon.ico',  # Add this line
)
```

### Build Options

The build scripts use these PyInstaller options:
- `--clean`: Remove temporary files before building
- `--noconfirm`: Overwrite output directory without asking

The spec files are configured to use `--onedir` mode by default (creating a directory with executable and dependencies) for faster startup times. This is preferred over `--onefile` mode which bundles everything into a single executable but has slower startup.

Additional useful options for customization:
- `--debug`: Enable debugging output
- `--log-level LEVEL`: Set logging verbosity

## Troubleshooting

### "PyInstaller not found"

Install PyInstaller:
```bash
pip install pyinstaller
```

### "Import errors in the executable"

Some packages need to be explicitly imported. Add them to the `hiddenimports` list in the `.spec` file:

```python
hiddenimports=[
    'trimesh.resources',
    'manifold3d',
    'your_missing_module',
],
```

### "Executable is too large"

The executables bundle all dependencies. The default --onedir mode already optimizes for faster startup. To reduce size:
1. Use `--exclude-module` to exclude unnecessary packages
2. Use UPX compression (already enabled with `upx=True`)

Note: The --onedir mode is now used by default instead of --onefile, which provides faster startup times at the cost of distributing a folder instead of a single file.

### "Different platforms"

**Important**: Executables are platform-specific:
- Windows executables only work on Windows
- Linux executables only work on Linux
- macOS executables only work on macOS

To support multiple platforms, build on each target platform separately.

## Testing the Executables

After building, test each executable from within its directory:

```bash
# Test generate_params
cd dist/generate_params
./generate_params input.csv --n-wings 4

# Test generate_wing (requires a CSV file)
cd ../generate_wing
./generate_wing input.csv --output test.stl

# Test analysis (requires a CSV file)
cd ../analysis
./analysis input.csv results.csv
```

Or test using full paths:
```bash
./dist/generate_params/generate_params input.csv --n-wings 4
./dist/generate_wing/generate_wing input.csv --output test.stl
./dist/analysis/analysis input.csv results.csv
```

## Continuous Integration

For automated builds, you can integrate these scripts into CI/CD pipelines:

### GitHub Actions Example

```yaml
- name: Build Windows Executables
  run: |
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    cd packaging
    build_windows.bat

- name: Upload Executables
  uses: actions/upload-artifact@v3
  with:
    name: windows-executables
    path: packaging/dist/*.exe
```

## Additional Resources

- [PyInstaller Documentation](https://pyinstaller.org/en/stable/)
- [PyInstaller Spec Files](https://pyinstaller.org/en/stable/spec-files.html)
- [PyInstaller Hooks](https://pyinstaller.org/en/stable/hooks.html)
