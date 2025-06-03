# Gage R&R Analysis Tool

A Python-based tool for performing Gage Repeatability and Reproducibility (Gage R&R) analysis on measurement data. This tool helps evaluate the quality of measurement systems by analyzing the variation in measurements.

## Features

- Interactive file selection using a graphical interface
- Comprehensive data cleaning and validation
- Analysis of multiple measurement types
- Detailed statistical analysis including:
  - ANOVA analysis
  - Variance components
  - Percent contribution
  - Study variation
  - Tolerance analysis
- Visual representation through various plots:
  - Components of variation
  - Response by part
  - X-bar chart
  - R chart
- PDF report generation with detailed results

## Requirements

- Python 3.6 or higher
- Required Python packages (install using `pip install -r requirements.txt`):
  - pandas
  - numpy
  - statsmodels
  - matplotlib
  - seaborn
  - reportlab
  - tkinter (usually comes with Python)
  - pyinstaller (for building executable)

## Installation

### Option 1: Using Python Script

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Option 2: Using Executable

1. Download the latest release from the releases page
2. Extract the ZIP file
3. Run `GageRRAnalysis.exe`

## Building the Executable

To build the executable yourself:

1. Install the requirements:
```bash
pip install -r requirements.txt
```

2. Run the setup script:
```bash
python setup.py
```

3. The executable will be created in the `dist` folder

## Usage

### Using Python Script

1. Run the script:
```bash
python gage_rr_analysis.py
```

### Using Executable

1. Double-click `GageRRAnalysis.exe`

2. Use the file selection dialog to choose your data file
   - The file should be a tab-separated text file
   - Required columns:
     - `Comp_Name`: Part identifier
     - `Solder_Diameter_Layer1` and subsequent measurement columns
   - Excluded columns:
     - `Location_X(pixel)`
     - `Location_Y(pixel)`
     - `Initial_H`
     - `Retry_H`

3. The program will:
   - Perform data cleaning and validation
   - Generate data cleaning plots
   - Perform Gage R&R analysis for each measurement
   - Create a comprehensive PDF report

4. Output files:
   - `data_cleaning_plots.png`: Visual representation of data distribution
   - `gage_rr_plots_[measurement_name].png`: Analysis plots for each measurement
   - `gage_rr_report.pdf`: Detailed analysis report

## Data Format

The input data file should be a tab-separated text file with the following structure:

```
Comp_Name    Solder_Diameter_Layer1    [Other Measurement Columns]
Part1        value1                     value2
Part2        value3                     value4
...
```

## Analysis Results

The generated PDF report includes:

1. ANOVA Results
   - Sum of squares
   - Degrees of freedom
   - Mean square
   - F-value
   - p-value

2. Variance Components
   - Repeatability
   - Part variation
   - Total variation

3. Percent Contribution
   - Repeatability
   - Part variation

4. Study Variation
   - Repeatability
   - Part variation
   - Total variation

5. Percent Tolerance
   - Repeatability
   - Total

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Add your chosen license here] 