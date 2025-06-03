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
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

def select_columns_gui(all_columns):
    """Show a Tkinter window for the user to select columns for Gage R&R analysis."""
    import tkinter as tk
    from tkinter import messagebox

    selected_cols = []
    root = tk.Tk()
    root.title("Select Measurement Columns")
    tk.Label(root, text="Select columns to perform Gage R&R analysis:").pack(padx=10, pady=10)
    listbox = tk.Listbox(root, selectmode=tk.MULTIPLE, width=40, height=min(20, len(all_columns)))
    for col in all_columns:
        listbox.insert(tk.END, col)
    listbox.pack(padx=10, pady=10)
    def on_ok():
        selected = [all_columns[i] for i in listbox.curselection()]
        if not selected:
            messagebox.showwarning("No Selection", "Please select at least one column.")
            return
        nonlocal selected_cols
        selected_cols = selected
        root.quit()  # Use quit instead of destroy here
    ok_btn = tk.Button(root, text="OK", command=on_ok)
    ok_btn.pack(pady=(0, 10))
    root.mainloop()
    root.destroy()  # Ensure window is destroyed after mainloop
    return selected_cols

def clean_data(df, measurement_cols=None):
    cleaning_report = []
    summary_stats = {}
    print("\nData Cleaning Report:")
    print("-" * 50)
    
    # Get all measurement columns (columns after Solder_Diameter_Layer1)
    # Exclude location_X, location_Y, Initial_H, and Retry_H
    excluded_cols = ['Location_X(pixel)', 'Location_Y(pixel)', 'Initial_H', 'Retry_H']
    if measurement_cols is None:
        measurement_cols = [col for col in df.columns[df.columns.get_loc('Solder_Diameter_Layer1'):] 
                           if col not in excluded_cols]
    else:
        measurement_cols = [col for col in measurement_cols if col not in excluded_cols]
    
    cleaning_report.append(f"Measurement columns to analyze: {measurement_cols}")
    cleaning_report.append(f"Excluded columns: {excluded_cols}")
    
    # 1. Check for missing values
    missing_values = df[measurement_cols].isnull().sum()
    if missing_values[missing_values > 0].any():
        cleaning_report.append("Missing Values:")
        cleaning_report.append(str(missing_values[missing_values > 0]))
    
    # 2. Check data types
    cleaning_report.append("Data Types:")
    cleaning_report.append(str(df[measurement_cols].dtypes))
    
    # 3. Check for negative or zero measurements
    for col in measurement_cols:
        negative_measurements = df[df[col] <= 0]
        if not negative_measurements.empty:
            cleaning_report.append(f"Warning: Found negative or zero measurements in {col}:")
            cleaning_report.append(str(negative_measurements[['Comp_Name', col]]))
    
    # 4. Check for outliers using IQR method
    for col in measurement_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        if not outliers.empty:
            cleaning_report.append(f"Potential Outliers in {col} (using IQR method):")
            cleaning_report.append(str(outliers[['Comp_Name', col]]))
    
    cleaning_report.append(f"Unique values in Comp_Name: {df['Comp_Name'].unique()}")
    
    # 6. Create summary statistics by Part
    for col in measurement_cols:
        stats = df.groupby('Comp_Name')[col].agg(['count', 'mean', 'std', 'min', 'max'])
        summary_stats[col] = stats.reset_index()
    
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
        
        cleaning_report.append(f"Final Data Shape for {col}: {data.shape}")
        cleaning_report.append(f"Number of unique Parts: {data['Part'].nunique()}")
        cleaning_report.append(f"Total number of measurements: {len(data)}")
        
        data_dict[col] = data
    
    return data_dict, cleaning_report, summary_stats

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
    """
    Create a 2x2 grid of plots matching the provided example image for each measurement.
    Plots: Components of Variation, S Chart, Measurement by Part, XBar Chart.
    All plots are the same size.
    """
    # Prepare data
    part_col = 'Part' if 'Part' in data.columns else 'Comp_Name'
    measurement_col = 'Measurement'
    grouped = data.groupby(part_col)[measurement_col]
    means = grouped.mean()
    stds = grouped.std(ddof=1)
    parts = means.index.astype(str)

    # XBar Chart (means)
    xbar = means.values
    xbar_cl = np.mean(xbar)
    xbar_s = np.std(xbar, ddof=1)
    xbar_n = len(xbar)
    # S Chart (stddevs)
    sbar = np.mean(stds.values)
    s_n = grouped.count().min()  # Smallest subgroup size
    n = s_n if s_n > 1 else 2
    s_chart_constants = {2: (1.128, 0.853, 1.747), 3: (1.693, 0.888, 1.954), 4: (2.059, 0.94, 2.089), 5: (2.326, 0.97, 2.185)}
    c4, B3, B4 = s_chart_constants.get(n, (1.128, 0.853, 1.747))
    s_UCL = sbar * B4
    s_LCL = sbar * B3
    s_CL = sbar

    # XBar chart limits (using stddev of means)
    A3 = 1.023  # For n=2, adjust if needed
    xbar_UCL = xbar_cl + A3 * sbar
    xbar_LCL = xbar_cl - A3 * sbar

    # Components of Variation
    percent_contrib = results['Percent Contribution']
    percent_study_var = results['Study Variation']
    # Align keys for both dicts
    all_keys = list(set(percent_contrib.keys()) | set(percent_study_var.keys()))
    all_keys.sort()  # Optional: sort for consistent order
    contrib_vals = [percent_contrib.get(k, 0) for k in all_keys]
    study_var_vals = [percent_study_var.get(k, 0) for k in all_keys]
    labels = all_keys

    # Start plotting with equal-sized subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    plt.subplots_adjust(top=0.88)

    # Title
    fig.suptitle(f"Gage R&R (ANOVA) Report for {measurement_name}", fontsize=18, fontweight='bold', y=0.98)

    # Components of Variation (Top Left)
    ax1 = axes[0, 0]
    width = 0.35
    x = np.arange(len(labels))
    ax1.bar(x - width/2, contrib_vals, width, label='% Contribution', color='#4F81BD')
    ax1.bar(x + width/2, study_var_vals, width, label='% Study Var', color='#C0504D')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0)
    ax1.set_ylabel('Percent')
    ax1.set_ylim(0, 110)
    ax1.legend()
    ax1.set_title('Components of Variation')
    for i, v in enumerate(contrib_vals):
        ax1.text(i - width/2, v + 2, f"{v:.1f}", ha='center', fontsize=9)
    for i, v in enumerate(study_var_vals):
        ax1.text(i + width/2, v + 2, f"{v:.1f}", ha='center', fontsize=9)

    # S Chart (Top Right)
    ax2 = axes[0, 1]
    ax2.plot(parts, stds.values, marker='o', linestyle='-', color='#0070C0')
    ax2.axhline(s_UCL, color='brown', linestyle='-', label=f'UCL={s_UCL:.6f}')
    ax2.axhline(s_CL, color='green', linestyle='-', label=f'S={s_CL:.6f}')
    ax2.axhline(s_LCL, color='black', linestyle='-', label=f'LCL={s_LCL:.6f}')
    ax2.set_title('S Chart')
    ax2.set_ylabel('Sample StDev')
    ax2.set_xticks([])
    ax2.legend(loc='upper left', fontsize=8)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.5f'))

    # Measurement by Part (Bottom Left)
    ax3 = axes[1, 0]
    ax3.plot(parts, means.values, marker='*', linestyle='-', color='gray')
    ax3.set_title(f'{measurement_name} by {part_col}')
    ax3.set_xlabel(part_col)
    ax3.set_ylabel('Sample Mean')
    ax3.tick_params(axis='x', rotation=45)

    # XBar Chart (Bottom Right)
    ax4 = axes[1, 1]
    ax4.plot(parts, means.values, marker='o', linestyle='-', color='#0070C0')
    ax4.axhline(xbar_UCL, color='brown', linestyle='-', label=f'UCL={xbar_UCL:.6f}')
    ax4.axhline(xbar_cl, color='green', linestyle='-', label=f'CL={xbar_cl:.6f}')
    ax4.axhline(xbar_LCL, color='black', linestyle='-', label=f'LCL={xbar_LCL:.6f}')
    ax4.set_title('XBar Chart')
    ax4.set_xlabel('Part')
    ax4.set_ylabel('Sample Mean')
    ax4.set_xticks([])
    ax4.legend(loc='upper left', fontsize=8)
    ax4.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.5f'))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'gage_rr_plots_{measurement_name}.png', dpi=150)
    plt.close(fig)

def ask_report_options():
    import tkinter as tk
    from tkinter import BooleanVar, Checkbutton, Button
    result = {'include_cleaning': True, 'include_summary': True}
    root = tk.Tk()
    root.title("Report Options")
    tk.Label(root, text="Select which sections to include in the PDF report:").pack(padx=10, pady=10)
    var_cleaning = BooleanVar(value=True)
    var_summary = BooleanVar(value=True)
    cb1 = Checkbutton(root, text="Include Data Cleaning Report", variable=var_cleaning)
    cb2 = Checkbutton(root, text="Include Summary Statistics", variable=var_summary)
    cb1.pack(anchor='w', padx=20)
    cb2.pack(anchor='w', padx=20)
    def on_ok():
        result['include_cleaning'] = var_cleaning.get()
        result['include_summary'] = var_summary.get()
        root.quit()
    Button(root, text="OK", command=on_ok).pack(pady=10)
    root.mainloop()
    root.destroy()
    return result

def create_pdf_report(results_dict, anova_tables_dict, data_dict, cleaning_report, summary_stats, report_options):
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
    
    # Conditionally add Data Cleaning Report
    if report_options.get('include_cleaning', True):
        story.append(Paragraph("Data Cleaning Report", styles['Heading2']))
        for line in cleaning_report:
            story.append(Paragraph(str(line), styles['Normal']))
        story.append(Spacer(1, 20))
    
    # Process each measurement
    for measurement_name, data in data_dict.items():
        results = results_dict[measurement_name]
        anova_table = anova_tables_dict[measurement_name]
        
        # Add measurement title
        story.append(Paragraph(f"Analysis for {measurement_name}", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # Conditionally add Summary Statistics Table
        if report_options.get('include_summary', True):
            story.append(Paragraph("Summary Statistics by Part", styles['Heading3']))
            stats_df = summary_stats.get(measurement_name)
            if stats_df is not None:
                stats_data = [list(stats_df.columns)] + stats_df.values.tolist()
                stats_table = Table(stats_data)
                stats_table.setStyle(TableStyle([
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
                story.append(stats_table)
                story.append(Spacer(1, 20))
        
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
        
        var_data = [['Source', 'VarComp', '%Contribution (of VarComp)']]
        var_comp = results['Variance Components']
        pct_contrib = results['Percent Contribution']
        # Map keys to match image
        row_order = [
            ('Total Gage R&R', var_comp['Repeatability'], pct_contrib['Repeatability']),
            ('Repeatability', var_comp['Repeatability'], pct_contrib['Repeatability']),
            ('Part-To-Part', var_comp['Part'], pct_contrib['Part']),
            ('Total Variation', var_comp['Total'], 100.0)
        ]
        for row in row_order:
            var_data.append([row[0], f"{row[1]:.8f}", f"{row[2]:.2f}"])
        
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
        
        # Add Gage Evaluation Table and Number of Distinct Categories BEFORE the plot
        study_var = results['Study Variation']
        pct_study_var = results['Percent Study Variation']
        stddev = {k: np.sqrt(var_comp[k]) for k in var_comp}
        gage_eval_data = [['Source', 'StdDev (SD)', 'Study Var (6 x SD)', '%Study Var (%SV)']]
        row_order_eval = [
            ('Total Gage R&R', stddev['Repeatability'], study_var['Repeatability'], pct_study_var['Repeatability']),
            ('Repeatability', stddev['Repeatability'], study_var['Repeatability'], pct_study_var['Repeatability']),
            ('Part-To-Part', stddev['Part'], study_var['Part'], pct_study_var['Part']),
            ('Total Variation', stddev['Total'], study_var['Total'], 100.0)
        ]
        for row in row_order_eval:
            gage_eval_data.append([row[0], f"{row[1]:.6f}", f"{row[2]:.6f}", f"{row[3]:.2f}"])
        
        gage_eval_table = Table(gage_eval_data)
        gage_eval_table.setStyle(TableStyle([
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
        story.append(gage_eval_table)
        story.append(Spacer(1, 20))
        
        ndc = results.get('Number of Distinct Categories', None)
        if ndc is not None:
            story.append(Paragraph(f"Number of Distinct Categories = {ndc}", styles['Normal']))
            story.append(Spacer(1, 20))
        
        # Generate and add plots
        # story.append(Paragraph("Gage R&R Plots", styles['Heading3']))
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
    root.destroy()  # Ensure window is destroyed
    if not file_path:  # If user cancels the dialog
        print("No file selected. Exiting...")
        return None
    return file_path

def calculate_ndc(var_part, var_gage_rr):
    if var_gage_rr == 0:
        return np.nan
    ndc = int(np.floor(1.41 * np.sqrt(var_part) / np.sqrt(var_gage_rr)))
    return ndc

def main():
    try:
        # Select file using tkinter dialog
        file_path = select_file()
        if file_path is None:
            return
        print(f"\nSelected file: {file_path}")
        # Read and prepare data
        df = read_data(file_path)
        # Get all eligible measurement columns
        excluded_cols = ['Location_X(pixel)', 'Location_Y(pixel)', 'Initial_H', 'Retry_H']
        all_measurement_cols = [col for col in df.columns[df.columns.get_loc('Solder_Diameter_Layer1'):] if col not in excluded_cols]
        # Let user select columns
        selected_cols = select_columns_gui(all_measurement_cols)
        if not selected_cols:
            print("No columns selected. Exiting...")
            return
        # Clean and prepare data
        data_dict, cleaning_report, summary_stats = clean_data(df, measurement_cols=selected_cols)
        # Ask user which report sections to include
        report_options = ask_report_options()
        # Perform Gage R&R analysis for each measurement
        results_dict = {}
        anova_tables_dict = {}
        for measurement_name, data in data_dict.items():
            print(f"\nPerforming Gage R&R analysis for {measurement_name}")
            results, anova_table = perform_gage_rr(data)
            var_part = results['Variance Components']['Part']
            var_gage_rr = results['Variance Components']['Repeatability']
            ndc = calculate_ndc(var_part, var_gage_rr)
            results['Number of Distinct Categories'] = ndc
            results_dict[measurement_name] = results
            anova_tables_dict[measurement_name] = anova_table
        # Create PDF report
        create_pdf_report(results_dict, anova_tables_dict, data_dict, cleaning_report, summary_stats, report_options)
        print("\nReport has been saved as 'gage_rr_report.pdf'")
        # Show completion message
        root = tk.Tk()
        root.withdraw()
        import tkinter.messagebox
        tkinter.messagebox.showinfo("Analysis Complete", 
                             "Gage R&R analysis has been completed.\n"
                             "The report has been saved as 'gage_rr_report.pdf'")
        root.destroy()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # Show error message
        root = tk.Tk()
        root.withdraw()
        import tkinter.messagebox
        tkinter.messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
        root.destroy()
        raise

if __name__ == "__main__":
    main() 