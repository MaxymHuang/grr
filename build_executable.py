#!/usr/bin/env python3
"""
Build script for creating a standalone executable of the Gage R&R Analysis Qt6 application.
Uses PyInstaller to create a single-file executable with all dependencies included.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_pyinstaller():
    """Check if PyInstaller is installed."""
    try:
        import PyInstaller
        print(f"✓ PyInstaller {PyInstaller.__version__} found")
        return True
    except ImportError:
        print("✗ PyInstaller not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller>=5.0.0"])
            print("✓ PyInstaller installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("✗ Failed to install PyInstaller")
            return False

def clean_build_dirs():
    """Clean previous build directories."""
    dirs_to_clean = ['build', 'dist', '__pycache__']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            print(f"Cleaning {dir_name}/")
            shutil.rmtree(dir_name)

def get_hidden_imports():
    """Get list of hidden imports needed for the application."""
    return [
        # PySide6 modules
        'PySide6.QtCore',
        'PySide6.QtGui', 
        'PySide6.QtWidgets',
        
        # Matplotlib backends and dependencies
        'matplotlib.backends.backend_qtagg',
        'matplotlib.backends.backend_pdf',
        'matplotlib.figure',
        'matplotlib.pyplot',
        'matplotlib.ticker',
        
        # Scientific computing
        'numpy',
        'pandas',
        'scipy',
        'scipy.stats',
        'statsmodels',
        'statsmodels.formula.api',
        'statsmodels.stats.anova',
        
        # Plotting
        'seaborn',
        'PIL',
        'PIL.Image',
        
        # Standard library modules that might be missed
        'math',
        'pathlib',
        'datetime',
        'io',
        'typing',
        'functools',
        'itertools',
    ]

def get_data_files():
    """Get list of data files to include."""
    data_files = []
    
    # Include matplotlib data files
    try:
        import matplotlib
        mpl_data_dir = matplotlib.get_data_path()
        data_files.append((f"{mpl_data_dir}", "matplotlib/mpl-data"))
    except:
        pass
    
    return data_files

def create_spec_file():
    """Create a detailed PyInstaller spec file."""
    spec_content = f'''# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path
import matplotlib

block_cipher = None

# Hidden imports
hiddenimports = {get_hidden_imports()}

# Data files
datas = []

# Add matplotlib data
try:
    mpl_data_dir = matplotlib.get_data_path()
    datas.append((str(mpl_data_dir), "matplotlib/mpl-data"))
except:
    pass

# Analysis
a = Analysis(
    ['gage_rr_analysis_qt.py'],
    pathex=[str(Path.cwd())],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'unittest', 
        'pdb',
        'doctest',
        'difflib',
        'inspect',
        'pydoc',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Remove duplicate binaries
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='GageRR_Analysis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True for debugging
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add path to .ico file if you have one
)
'''
    
    with open('gage_rr_analysis.spec', 'w') as f:
        f.write(spec_content)
    
    print("✓ Created PyInstaller spec file")

def build_executable():
    """Build the executable using PyInstaller."""
    print("Building executable...")
    
    # Command line arguments for PyInstaller - only spec file needed
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "gage_rr_analysis.spec"
    ]
    
    try:
        # Run PyInstaller
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Executable built successfully!")
            
            # Check if executable was created
            if sys.platform.startswith('win'):
                exe_path = Path("dist/GageRR_Analysis.exe")
            else:
                exe_path = Path("dist/GageRR_Analysis")
            
            if exe_path.exists():
                print(f"✓ Executable location: {exe_path.absolute()}")
                print(f"✓ File size: {exe_path.stat().st_size / (1024*1024):.1f} MB")
                return True
            else:
                print("✗ Executable not found in expected location")
                return False
                
        else:
            print("✗ Build failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ Build error: {e}")
        return False

def main():
    """Main build process."""
    print("=" * 60)
    print("🔧 Gage R&R Analysis - Executable Builder")
    print("=" * 60)
    
    # Check current directory
    if not Path("gage_rr_analysis_qt.py").exists():
        print("✗ Error: gage_rr_analysis_qt.py not found in current directory")
        print("Please run this script from the project root directory.")
        return False
    
    # Check and install PyInstaller
    if not check_pyinstaller():
        return False
    
    # Clean previous builds
    print("\\nCleaning previous builds...")
    clean_build_dirs()
    
    # Create spec file
    print("\\nCreating PyInstaller configuration...")
    create_spec_file()
    
    # Build executable
    print("\\nBuilding executable (this may take several minutes)...")
    success = build_executable()
    
    if success:
        print("\\n" + "=" * 60)
        print("🎉 BUILD SUCCESSFUL!")
        print("=" * 60)
        print("\\nThe standalone executable has been created in the 'dist/' directory.")
        print("You can distribute this file without requiring Python installation.")
        print("\\nUsage:")
        if sys.platform.startswith('win'):
            print("  - Windows: Double-click GageRR_Analysis.exe")
        else:
            print("  - macOS/Linux: ./GageRR_Analysis")
        
        # Cleanup
        print("\\nCleaning up temporary files...")
        if Path("gage_rr_analysis.spec").exists():
            Path("gage_rr_analysis.spec").unlink()
        
        return True
    else:
        print("\\n" + "=" * 60)
        print("❌ BUILD FAILED!")
        print("=" * 60)
        print("Please check the error messages above and try again.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
