import PyInstaller.__main__
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the icon path (you can add an icon file later)
# icon_path = os.path.join(current_dir, 'icon.ico')

PyInstaller.__main__.run([
    'gage_rr_analysis.py',
    '--onefile',
    '--name=GageRRAnalysis',
    '--clean',
    '--noconsole',
    # '--icon=' + icon_path,  # Uncomment when you have an icon
    '--add-data=requirements.txt;.',
]) 