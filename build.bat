@echo off
REM Quick build script for Windows

echo 🔧 Building Gage R&R Analysis Executable...
echo ============================================

REM Check if virtual environment exists
if exist ".venv\Scripts\activate.bat" (
    echo ✓ Virtual environment found
    call .venv\Scripts\activate.bat
) else (
    echo ⚠️  Virtual environment not found. Creating...
    python -m venv .venv
    call .venv\Scripts\activate.bat
)

REM Install/upgrade required packages
echo 📦 Installing required packages...
pip install -r requirements.txt

REM Run the build script
echo 🏗️  Starting build process...
python build_executable.py

echo ✅ Build process completed!
pause
