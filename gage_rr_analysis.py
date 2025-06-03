import pandas as pd
import numpy as np
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import tkinter as tk
from tkinter import filedialog
import os

def clean_data(df):
    """Clean and prepare the data for Gage R&R analysis."""
    print("\nData Cleaning Report:")
    print("-" * 50)
    
    # Get all measurement columns (columns after Solder_Diameter_Layer1)
    # Exclude location_X, location_Y, Initial_H, and Retry_H
    excluded_cols = ['Location_X(pixel)', 'Location_Y(pixel)', 'Initial_H', 'Retry_H']
    measurement_cols = [col for col in df.columns[df.columns.get_loc('Solder_Diameter_Layer1'):] 
                       if col not in excluded_cols]
    
    print("\nMeasurement columns to analyze:", measurement_cols)
    print("\nExcluded columns:", excluded_cols)
    
    # 1. Check for missing values
    missing_values = df[measurement_cols].isnull().sum()
    print("\nMissing Values:")
    print(missing_values[missing_values > 0])
    
    # 2. Check data types
    print("\nData Types:")
    print(df[measurement_cols].dtypes)
    
    # 3. Check for negative or zero measurements
    for col in measurement_cols:
        negative_measurements = df[df[col] <= 0]
        if not negative_measurements.empty:
            print(f"\nWarning: Found negative or zero measurements in {col}:")
            print(negative_measurements[['Comp_Name', col]])
    
    # 4. Check for outliers using IQR method
    for col in measurement_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        if not outliers.empty:
            print(f"\nPotential Outliers in {col} (using IQR method):")
            print(outliers[['Comp_Name', col]])
    
    # 5. Check for consistency in categorical data
    print("\nUnique values in Comp_Name:", df['Comp_Name'].unique())
    
    # 6. Create summary statistics by Part
    for col in measurement_cols:
        print(f"\nSummary Statistics for {col} by Part:")
        print(df.groupby('Comp_Name')[col].agg(['count', 'mean', 'std', 'min', 'max']))
    
    # 7. Create box plots for visual inspection
    n_cols = len(measurement_cols)
    n_rows = (n_cols + 1) // 2  # Ceiling division
    plt.figure(figsize=(15, 5 * n_rows))
    
    for i, col in enumerate(measurement_cols, 1):
        plt.subplot(n_rows, 2, i)
        sns.boxplot(x='Comp_Name', y=col, data=df)
        plt.title(f'Distribution of {col} by Part')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('data_cleaning_plots.png')
    plt.close()
    
    # 8. Prepare data for Gage R&R
    # Create a dictionary to store data for each measurement
    data_dict = {}
    
    for col in measurement_cols:
        # Create a copy of the dataframe with only necessary columns
        data = df[['Comp_Name', col]].copy()
        
        # Rename columns for clarity
        data.columns = ['Part', 'Measurement']
        
        # Remove rows with missing values
        data = data.dropna()
        
        # Remove rows with negative or zero measurements
        data = data[data['Measurement'] > 0]
        
        # Remove outliers if needed (uncomment if you want to remove outliers)
        # Q1 = data['Measurement'].quantile(0.25)
        # Q3 = data['Measurement'].quantile(0.75)
        # IQR = Q3 - Q1
        # lower_bound = Q1 - 1.5 * IQR
        # upper_bound = Q3 + 1.5 * IQR
        # data = data[(data['Measurement'] >= lower_bound) & (data['Measurement'] <= upper_bound)]
        
        print(f"\nFinal Data Shape for {col}:", data.shape)
        print(f"Number of unique Parts: {data['Part'].nunique()}")
        print(f"Total number of measurements: {len(data)}")
        
        data_dict[col] = data
    
    return data_dict

def read_data(file_path):
    """Read the data from the text file."""
    try:
        # Handle Windows path properly
        file_path = file_path.replace('\\', '/')
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
        if df.empty:
            raise ValueError("The file is empty")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except pd.errors.EmptyDataError:
        print("Error: The file is empty")
        raise
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        raise

def perform_gage_rr(data):
    """Perform Gage R&R ANOVA analysis."""
    # Fit the ANOVA model
    model = ols('Measurement ~ Part', data=data).fit()
    anova_table = anova_lm(model, typ=2)
    
    # Calculate variance components
    MS_part = anova_table.loc['Part', 'sum_sq'] / anova_table.loc['Part', 'df']
    MS_error = anova_table.loc['Residual', 'sum_sq'] / anova_table.loc['Residual', 'df']
    
    # Calculate variance components
    n_parts = len(data['Part'].unique())
    n_replicates = len(data) / n_parts
    
    var_repeatability = MS_error
    var_part = max(0, (MS_part - MS_error) / n_replicates)  # Ensure non-negative
    
    # Calculate total variance
    var_total = var_repeatability + var_part
    
    # Calculate % contribution
    pct_repeatability = (var_repeatability / var_total) * 100
    pct_part = (var_part / var_total) * 100
    
    # Calculate study variation
    study_var_repeatability = 5.15 * np.sqrt(var_repeatability)
    study_var_part = 5.15 * np.sqrt(var_part)
    study_var_total = 5.15 * np.sqrt(var_total)
    
    # Calculate % study variation
    pct_study_var_repeatability = (study_var_repeatability / study_var_total) * 100
    pct_study_var_part = (study_var_part / study_var_total) * 100
    
    # Calculate % tolerance (assuming tolerance is 6 * standard deviation)
    tolerance = 6 * np.sqrt(var_total)
    pct_tolerance_repeatability = (study_var_repeatability / tolerance) * 100
    pct_tolerance_total = (study_var_total / tolerance) * 100
    
    # Create results dictionary
    results = {
        'Variance Components': {
            'Repeatability': var_repeatability,
            'Part': var_part,
            'Total': var_total
        },
        'Percent Contribution': {
            'Repeatability': pct_repeatability,
            'Part': pct_part
        },
        'Study Variation': {
            'Repeatability': study_var_repeatability,
            'Part': study_var_part,
            'Total': study_var_total
        },
        'Percent Study Variation': {
            'Repeatability': pct_study_var_repeatability,
            'Part': pct_study_var_part
        },
        'Percent Tolerance': {
            'Repeatability': pct_tolerance_repeatability,
            'Total': pct_tolerance_total
        }
    }
    
    return results, anova_table

def plot_results(data, results, measurement_name):
    """Create plots for Gage R&R analysis."""
    # Create a figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Components of Variation
    components = pd.Series(results['Percent Contribution'])
    bars = components.plot(kind='bar', ax=ax1)
    ax1.set_title('Components of Variation')
    ax1.set_ylabel('Percent Contribution')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on top of bars
    for i, v in enumerate(components):
        ax1.text(i, v, f'{v:.1f}%', ha='center', va='bottom')
    
    # Plot 2: Response by Part
    sns.boxplot(x='Part', y='Measurement', data=data, ax=ax2)
    ax2.set_title('Response by Part')
    ax2.set_xlabel('Part')
    ax2.set_ylabel('Measurement')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add mean values for each part
    part_means = data.groupby('Part')['Measurement'].mean()
    for i, (part, mean) in enumerate(part_means.items()):
        ax2.text(i, mean, f'{mean:.3f}', ha='center', va='bottom')
    
    # Plot 3: X-bar Chart
    part_means = data.groupby('Part')['Measurement'].mean()
    part_std = data.groupby('Part')['Measurement'].std()
    grand_mean = data['Measurement'].mean()
    ucl = grand_mean + 3 * part_std.mean()
    lcl = grand_mean - 3 * part_std.mean()
    
    part_means.plot(kind='line', marker='o', ax=ax3)
    ax3.axhline(y=grand_mean, color='r', linestyle='--', label=f'Grand Mean: {grand_mean:.3f}')
    ax3.axhline(y=ucl, color='g', linestyle='--', label=f'UCL: {ucl:.3f}')
    ax3.axhline(y=lcl, color='g', linestyle='--', label=f'LCL: {lcl:.3f}')
    ax3.set_title('X-bar Chart by Part')
    ax3.set_xlabel('Part')
    ax3.set_ylabel('Mean Measurement')
    
    # Add value labels for each point
    for i, (part, mean) in enumerate(part_means.items()):
        ax3.text(i, mean, f'{mean:.3f}', ha='center', va='bottom')
    
    ax3.legend()
    
    # Plot 4: R Chart
    part_ranges = data.groupby('Part')['Measurement'].apply(lambda x: x.max() - x.min())
    r_mean = part_ranges.mean()
    r_ucl = r_mean * 3.267  # D4 constant for n=2
    r_lcl = 0  # D3 constant for n=2
    
    part_ranges.plot(kind='line', marker='o', ax=ax4)
    ax4.axhline(y=r_mean, color='r', linestyle='--', label=f'Mean Range: {r_mean:.3f}')
    ax4.axhline(y=r_ucl, color='g', linestyle='--', label=f'UCL: {r_ucl:.3f}')
    ax4.axhline(y=r_lcl, color='g', linestyle='--', label=f'LCL: {r_lcl:.3f}')
    ax4.set_title('R Chart by Part')
    ax4.set_xlabel('Part')
    ax4.set_ylabel('Range')
    
    # Add value labels for each point
    for i, (part, range_val) in enumerate(part_ranges.items()):
        ax4.text(i, range_val, f'{range_val:.3f}', ha='center', va='bottom')
    
    ax4.legend()
    
    plt.suptitle(f'Gage R&R Analysis for {measurement_name}', y=1.02)
    plt.tight_layout()
    plt.savefig(f'gage_rr_plots_{measurement_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_pdf_report(results_dict, anova_tables_dict, data_dict):
    """Create a PDF report with ANOVA results and plots."""
    # Create the PDF document
    doc = SimpleDocTemplate("gage_rr_report.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Add title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30
    )
    story.append(Paragraph("Gage R&R Analysis Report", title_style))
    
    # Process each measurement
    for measurement_name, data in data_dict.items():
        results = results_dict[measurement_name]
        anova_table = anova_tables_dict[measurement_name]
        
        # Add measurement title
        story.append(Paragraph(f"Analysis for {measurement_name}", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # Add ANOVA Table
        story.append(Paragraph("ANOVA Results", styles['Heading3']))
        story.append(Spacer(1, 12))
        
        # Convert ANOVA table to a format suitable for reportlab
        anova_data = [['Source', 'Sum of Squares', 'df', 'Mean Square', 'F-value', 'p-value']]
        for idx, row in anova_table.iterrows():
            mean_sq = row['sum_sq'] / row['df']  # Calculate mean square manually
            anova_data.append([
                idx,
                f"{row['sum_sq']:.6f}",
                f"{row['df']:.0f}",
                f"{mean_sq:.6f}",
                f"{row['F']:.6f}",
                f"{row['PR(>F)']:.6f}"
            ])
        
        # Create ANOVA table
        anova_table = Table(anova_data)
        anova_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(anova_table)
        story.append(Spacer(1, 20))
        
        # Add Variance Components
        story.append(Paragraph("Variance Components", styles['Heading3']))
        story.append(Spacer(1, 12))
        
        var_data = [['Component', 'Value']]
        for component, value in results['Variance Components'].items():
            var_data.append([component, f"{value:.6f}"])
        
        var_table = Table(var_data)
        var_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(var_table)
        story.append(Spacer(1, 20))
        
        # Add Percent Contribution
        story.append(Paragraph("Percent Contribution", styles['Heading3']))
        story.append(Spacer(1, 12))
        
        pct_data = [['Component', 'Percent']]
        for component, value in results['Percent Contribution'].items():
            pct_data.append([component, f"{value:.2f}%"])
        
        pct_table = Table(pct_data)
        pct_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(pct_table)
        story.append(Spacer(1, 20))
        
        # Generate and add plots
        story.append(Paragraph("Gage R&R Plots", styles['Heading3']))
        story.append(Spacer(1, 12))
        
        # Create the plots
        plot_results(data, results, measurement_name)
        
        # Add the plots to the PDF
        img = Image(f'gage_rr_plots_{measurement_name}.png', width=7*inch, height=8.4*inch)
        story.append(img)
        story.append(Spacer(1, 20))
    
    # Build the PDF
    doc.build(story)

def select_file():
    """Open a file dialog to select the data file."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Get the user's home directory
    home_dir = os.path.expanduser("~")
    
    # Open file dialog
    file_path = filedialog.askopenfilename(
        initialdir=home_dir,
        title="Select Data File",
        filetypes=(
            ("Text files", "*.txt"),
            ("All files", "*.*")
        )
    )
    
    if not file_path:  # If user cancels the dialog
        print("No file selected. Exiting...")
        return None
    
    return file_path

def main():
    try:
        # Select file using tkinter dialog
        file_path = select_file()
        if file_path is None:
            return
        
        print(f"\nSelected file: {file_path}")
        
        # Read and prepare data
        df = read_data(file_path)
        
        # Clean and prepare data
        data_dict = clean_data(df)
        
        # Perform Gage R&R analysis for each measurement
        results_dict = {}
        anova_tables_dict = {}
        
        for measurement_name, data in data_dict.items():
            print(f"\nPerforming Gage R&R analysis for {measurement_name}")
            results, anova_table = perform_gage_rr(data)
            results_dict[measurement_name] = results
            anova_tables_dict[measurement_name] = anova_table
        
        # Create PDF report
        create_pdf_report(results_dict, anova_tables_dict, data_dict)
        print("\nReport has been saved as 'gage_rr_report.pdf'")
        
        # Show completion message
        root = tk.Tk()
        root.withdraw()
        tk.messagebox.showinfo("Analysis Complete", 
                             "Gage R&R analysis has been completed.\n"
                             "The report has been saved as 'gage_rr_report.pdf'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # Show error message
        root = tk.Tk()
        root.withdraw()
        tk.messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
        raise

if __name__ == "__main__":
    main() 