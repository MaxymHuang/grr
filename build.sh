#!/bin/bash
# Quick build script for Unix/Linux/macOS

echo "🔧 Building Gage R&R Analysis Executable..."
echo "============================================"

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  Virtual environment not detected. Activating..."
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
        echo "✓ Virtual environment activated"
    else
        echo "❌ No virtual environment found. Please run: python -m venv .venv && source .venv/bin/activate"
        exit 1
    fi
fi

# Install/upgrade required packages
echo "📦 Installing required packages..."
pip install -r requirements.txt

# Run the build script
echo "🏗️  Starting build process..."
python build_executable.py

echo "✅ Build process completed!"
