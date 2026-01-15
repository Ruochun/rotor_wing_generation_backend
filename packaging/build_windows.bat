@echo off
REM Build script for Windows executables
REM This script builds all three executables using PyInstaller

echo ========================================
echo Building Rotor Wing Generation Executables
echo ========================================
echo.

REM Check if pyinstaller is installed
python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo ERROR: PyInstaller is not installed!
    echo Please run: pip install pyinstaller
    exit /b 1
)

REM Navigate to packaging directory
cd /d "%~dp0"

echo Building generate_params.exe...
pyinstaller --clean --noconfirm generate_params.spec
if errorlevel 1 (
    echo ERROR: Failed to build generate_params.exe
    exit /b 1
)
echo.

echo Building generate_wing.exe...
pyinstaller --clean --noconfirm generate_wing.spec
if errorlevel 1 (
    echo ERROR: Failed to build generate_wing.exe
    exit /b 1
)
echo.

echo Building analysis.exe...
pyinstaller --clean --noconfirm analysis.spec
if errorlevel 1 (
    echo ERROR: Failed to build analysis.exe
    exit /b 1
)
echo.

echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo Executables are located in: dist\
echo   - generate_params.exe
echo   - generate_wing.exe
echo   - analysis.exe
echo.
echo You can now distribute these executables to users.
echo Users do NOT need to install Python or any dependencies.
