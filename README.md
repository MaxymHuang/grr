# Gage R&R Analysis Tool (Streamlit)

A simple, no-coding-required app to analyze Gage Repeatability and Reproducibility (Gage R&R) in your browser.

## Features

- Interactive, browser-based UI (Streamlit)
- Comprehensive data cleaning and validation
- Supported measurements only: `solder_thickness`, `solder_diameter` (e.g., `Solder_Diameter_Layer1`), `solder_area`
- One-operator GR&R using one-way ANOVA (Measurement ~ Part)
- Metrics: variance components, % Contribution (adds to 100%), % Study Variation (not additive), NDC
- Plots: Components view, S chart, Response-by-part, X̄ chart
- Note: PDF export and desktop executable were removed

## Requirements

- Python 3.8 or higher
- Install packages with: `pip install -r requirements.txt`

## Quick Start (no DevOps experience needed)

1) Install Python
- Windows/macOS: download and install from `https://www.python.org/downloads/`
- On Windows, check “Add Python to PATH” during install

2) Get this project
- Download ZIP from the GitHub repository and unzip, or clone via Git

3) Open a terminal and go to the folder
```bash
cd path/to/grr
```

4) (Recommended) Create and activate a virtual environment
- Windows (PowerShell):
```bash
py -m venv .venv
.\.venv\Scripts\activate
```
- macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

5) Install dependencies
```bash
pip install -r requirements.txt
```

6) Run the app
```bash
streamlit run gage_rr_analysis.py
```
The browser will open automatically. If not, visit `http://localhost:8501`.

## Using the App
1) Upload your data file (`.txt`, `.tsv`, or `.csv`)
2) Select your measurement columns (only supported names will appear)
3) Keep “Data Cleaning Report” checked to view checks and boxplots
4) Review ANOVA, key metrics, and charts for each selected measurement

## Data Format

Tab- or comma-delimited text file with a part identifier and measurement columns. Example:

```
Comp_Name    Solder_Diameter_Layer1    Solder_Thickness    Solder_Area
Part1        0.123                      0.045               3.21
Part2        0.127                      0.044               3.18
...
```

## Notes on Metrics
- % Contribution (variance-based) for Repeatability and Part adds to 100%
- % Study Variation (std-dev based) is not additive and will not sum to 100%
- NDC uses the standard formula and returns “—” if Repeatability variance is 0

## Contributing
Issues and enhancement requests are welcome.

## License
See `LICENSE`.