import os
import io
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols

import streamlit as st
from math import sqrt, gamma

ALLOWED_METRIC_KEYWORDS = (
    'solder_thickness',
    'solder_diameter',
    'solder_area',
)

def _infer_measurement_columns(df: pd.DataFrame) -> list:
    """Infer default measurement columns from the dataframe.

    Only include columns whose names contain any of the supported
    metric keywords: solder_thickness, solder_diameter, solder_area.
    """
    excluded_cols = ['Location_X(pixel)', 'Location_Y(pixel)', 'Initial_H', 'Retry_H']
    selected = [
        col for col in df.columns
        if col not in excluded_cols and any(key in col.lower() for key in ALLOWED_METRIC_KEYWORDS)
    ]
    return selected

def clean_data(
    df: pd.DataFrame,
    measurement_cols: Optional[list] = None,
    part_col: Optional[str] = None,
):
    cleaning_report = []
    summary_stats = {}
    print("\nData Cleaning Report:")
    print("-" * 50)
    
    # Get all measurement columns (columns after Solder_Diameter_Layer1)
    # Exclude location_X, location_Y, Initial_H, and Retry_H
    excluded_cols = ['Location_X(pixel)', 'Location_Y(pixel)', 'Initial_H', 'Retry_H']
    if measurement_cols is None:
        measurement_cols = _infer_measurement_columns(df)
    else:
        # Enforce allowed metric keywords and drop excluded columns
        measurement_cols = [
            col for col in measurement_cols
            if col not in excluded_cols and any(key in col.lower() for key in ALLOWED_METRIC_KEYWORDS)
        ]
    
    cleaning_report.append(f"Measurement columns to analyze: {measurement_cols}")
    cleaning_report.append(f"Excluded columns: {excluded_cols}")

    # Determine part identifier column once
    part_col = (
        part_col
        if part_col in df.columns
        else ('Comp_Name' if 'Comp_Name' in df.columns else ('Part' if 'Part' in df.columns else None))
    )
    
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
            if part_col is not None and part_col in negative_measurements.columns:
                cleaning_report.append(str(negative_measurements[[part_col, col]]))
            else:
                cleaning_report.append(str(negative_measurements[[col]]))
    
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
            if part_col is not None and part_col in outliers.columns:
                cleaning_report.append(str(outliers[[part_col, col]]))
            else:
                cleaning_report.append(str(outliers[[col]]))
    
    if part_col is not None and part_col in df.columns:
        cleaning_report.append(f"Unique values in {part_col}: {df[part_col].unique()}")
    
    # 6. Create summary statistics by Part
    for col in measurement_cols:
        if part_col is None:
            continue
        stats = df.groupby(part_col)[col].agg(['count', 'mean', 'std', 'min', 'max'])
        summary_stats[col] = stats.reset_index()
    
    # 7. Create box plots for visual inspection
    n_cols = len(measurement_cols)
    n_rows = (n_cols + 1) // 2  # Ceiling division
    fig_cleaning = plt.figure(figsize=(15, 5 * n_rows))
    
    # Use previously determined part_col for plotting
    for i, col in enumerate(measurement_cols, 1):
        plt.subplot(n_rows, 2, i)
        if part_col is None:
            plt.text(0.5, 0.5, 'No part identifier column found', ha='center', va='center')
            plt.axis('off')
            continue
        plot_df = df[[part_col, col]].copy()
        # Coerce to numeric and drop missing
        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
        plot_df = plot_df.dropna(subset=[part_col, col])
        # Ensure each group has at least one observation
        valid_parts = plot_df.groupby(part_col)[col].count()
        valid_parts = valid_parts[valid_parts > 0].index
        plot_df = plot_df[plot_df[part_col].isin(valid_parts)]
        if plot_df.empty:
            plt.text(0.5, 0.5, f'No valid data for {col}', ha='center', va='center')
            plt.axis('off')
            continue
        sns.boxplot(x=part_col, y=col, data=plot_df)
        plt.title(f'Distribution of {col} by Part')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    # Do not save to disk; return the figure for UI display
    
    # 8. Prepare data for Gage R&R
    # Create a dictionary to store data for each measurement
    data_dict = {}
    
    for col in measurement_cols:
        if part_col is None or part_col not in df.columns:
            continue
        # Create a copy of the dataframe with only necessary columns
        data = df[[part_col, col]].copy()
        
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
    
    return data_dict, cleaning_report, summary_stats, fig_cleaning

def read_data(file_source):
    """Read the data from a file path or file-like object.

    Tries tab-delimited first, then falls back to auto-detected delimiter.
    """
    try:
        if hasattr(file_source, 'read'):
            # file-like (e.g., Streamlit uploader)
            file_source.seek(0)
            try:
                df = pd.read_csv(file_source, sep='\t', engine='python')
            except Exception:
                file_source.seek(0)
                df = pd.read_csv(file_source, sep=None, engine='python')
        else:
            # path-like
            file_path = str(file_source).replace('\\', '/')
            try:
                df = pd.read_csv(file_path, sep='\t', encoding='utf-8', engine='python')
            except Exception:
                df = pd.read_csv(file_path, sep=None, encoding='utf-8', engine='python')
        if df.empty:
            raise ValueError("The file is empty")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_source}")
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
    # Use average replicates per part to handle unbalanced data more robustly
    group_sizes = data.groupby('Part')['Measurement'].count().values
    n_replicates = float(np.mean(group_sizes)) if len(group_sizes) > 0 else 1.0
    
    var_repeatability = MS_error
    var_part = max(0, (MS_part - MS_error) / n_replicates)  # Ensure non-negative
    
    # Calculate total variance
    var_total = var_repeatability + var_part
    
    # Calculate % contribution
    pct_repeatability = (var_repeatability / var_total) * 100
    pct_part = (var_part / var_total) * 100
    
    # Calculate study variation using 6 × SD (common convention)
    study_var_repeatability = 6.0 * np.sqrt(var_repeatability)
    study_var_part = 6.0 * np.sqrt(var_part)
    study_var_total = 6.0 * np.sqrt(var_total)
    
    # Calculate % study variation
    pct_study_var_repeatability = (study_var_repeatability / study_var_total) * 100
    pct_study_var_part = (study_var_part / study_var_total) * 100
    
    # Calculate % tolerance (assuming tolerance is 6 * standard deviation)
    tolerance = 6.0 * np.sqrt(var_total)
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
    sbar = np.nanmean(stds.values)
    s_n = grouped.count().min()  # Smallest subgroup size
    n = int(s_n) if s_n and s_n >= 2 else None

    def c4_const(nn: int) -> float:
        return sqrt(2.0 / (nn - 1.0)) * (gamma(nn / 2.0) / gamma((nn - 1.0) / 2.0))

    if n is not None:
        c4n = c4_const(n)
        # Xbar-S constants
        A3 = 3.0 / (c4n * sqrt(float(n)))
        # S chart constants
        term = 3.0 * sqrt(1.0 - c4n**2) / c4n
        B3 = max(0.0, 1.0 - term)
        B4 = 1.0 + term
        s_UCL = sbar * B4
        s_LCL = sbar * B3
        s_CL = sbar
    else:
        A3 = None
        s_UCL = s_LCL = s_CL = None

    # XBar chart limits (using stddev of means)
    if A3 is not None and not np.isnan(sbar):
        xbar_UCL = xbar_cl + A3 * sbar
        xbar_LCL = xbar_cl - A3 * sbar
    else:
        xbar_UCL = xbar_LCL = None

    # Components of Variation
    percent_contrib = results['Percent Contribution']
    percent_study_var = results['Percent Study Variation']

    # Start plotting with equal-sized subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    plt.subplots_adjust(top=0.88)

    # Title
    fig.suptitle(f"Gage R&R (ANOVA) Report for {measurement_name}", fontsize=18, fontweight='bold', y=0.98)

    # Components of Variation (Top Left)
    ax1 = axes[0, 0]
    width = 0.35
    metrics = ['% Contribution', '% Study Var']
    x = np.arange(len(metrics))
    rep_vals = [
        float(percent_contrib.get('Repeatability', 0) or 0),
        float(percent_study_var.get('Repeatability', 0) or 0),
    ]
    part_vals = [
        float(percent_contrib.get('Part', 0) or 0),
        float(percent_study_var.get('Part', 0) or 0),
    ]
    bars_rep = ax1.bar(x - width/2, rep_vals, width, label='Repeatability', color='#C0504D')
    bars_part = ax1.bar(x + width/2, part_vals, width, label='Part', color='#4F81BD')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=0)
    ax1.set_ylabel('Percent')
    ax1.set_ylim(0, 110)
    ax1.legend()
    ax1.set_title('Components of Variation')
    ax1.text(0.5, -0.2, 'Note: % Study Var is not additive', transform=ax1.transAxes,
             ha='center', va='top', fontsize=9, color='gray')
    def _round_pair_to_100(a: float, b: float, decimals: int = 1) -> tuple[float, float]:
        A = round(a, decimals)
        B = round(b, decimals)
        diff = round(100.0 - (A + B), decimals)
        # Adjust the larger component by the residual to make sum exactly 100
        if abs(a) >= abs(b):
            A += diff
        else:
            B += diff
        # Clip to [0, 100]
        A = min(max(A, 0.0), 100.0)
        B = min(max(B, 0.0), 100.0)
        return A, B

    # Annotate bars with rounded labels
    # Group 0: % Contribution (ensure labels sum to 100 within rounding)
    rep_c_rounded, part_c_rounded = _round_pair_to_100(rep_vals[0], part_vals[0], decimals=1)
    ax1.text(bars_rep[0].get_x() + bars_rep[0].get_width()/2, bars_rep[0].get_height() + 2,
             f"{rep_c_rounded:.1f}", ha='center', fontsize=9)
    ax1.text(bars_part[0].get_x() + bars_part[0].get_width()/2, bars_part[0].get_height() + 2,
             f"{part_c_rounded:.1f}", ha='center', fontsize=9)
    # Group 1: % Study Var (not additive; show as-is)
    ax1.text(bars_rep[1].get_x() + bars_rep[1].get_width()/2, bars_rep[1].get_height() + 2,
             f"{rep_vals[1]:.1f}", ha='center', fontsize=9)
    ax1.text(bars_part[1].get_x() + bars_part[1].get_width()/2, bars_part[1].get_height() + 2,
             f"{part_vals[1]:.1f}", ha='center', fontsize=9)

    # S Chart (Top Right)
    ax2 = axes[0, 1]
    ax2.plot(parts, stds.values, marker='o', linestyle='-', color='#0070C0')
    if s_UCL is not None:
        ax2.axhline(s_UCL, color='brown', linestyle='-', label=f'UCL={s_UCL:.6f}')
    if s_CL is not None:
        ax2.axhline(s_CL, color='green', linestyle='-', label=f'S={s_CL:.6f}')
    if s_LCL is not None:
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
    if xbar_UCL is not None:
        ax4.axhline(xbar_UCL, color='brown', linestyle='-', label=f'UCL={xbar_UCL:.6f}')
    ax4.axhline(xbar_cl, color='green', linestyle='-', label=f'CL={xbar_cl:.6f}')
    if xbar_LCL is not None:
        ax4.axhline(xbar_LCL, color='black', linestyle='-', label=f'LCL={xbar_LCL:.6f}')
    ax4.set_title('XBar Chart')
    ax4.set_xlabel('Part')
    ax4.set_ylabel('Sample Mean')
    ax4.set_xticks([])
    ax4.legend(loc='upper left', fontsize=8)
    ax4.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.5f'))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # Return figure for UI display rather than saving to disk
    return fig

def calculate_ndc(var_part, var_gage_rr):
    if var_gage_rr == 0:
        return np.nan
    ndc = int(np.floor(1.41 * np.sqrt(var_part) / np.sqrt(var_gage_rr)))
    return ndc

# ------------------------
# Streamlit UI
# ------------------------

st.set_page_config(page_title="Gage R&R Analysis", layout="wide")
st.title("Gage R&R Analysis Tool")
st.write(
    "Upload a tab- or comma-delimited file, select measurement columns, and view the Gage R&R analysis."
)

uploaded_file = st.file_uploader("Upload data file", type=["txt", "tsv", "csv"]) 

if uploaded_file is not None:
    try:
        df = read_data(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        df = None

    if df is not None:
        st.subheader("Preview")
        st.dataframe(df.head(50))

        # Determine measurement columns (only supported metrics)
        default_measurement_cols = _infer_measurement_columns(df)
        candidate_columns = default_measurement_cols
        if not candidate_columns:
            st.warning("No supported measurement columns found. Looking for names containing: 'solder_thickness', 'solder_diameter', or 'solder_area'.")
        selected_cols = st.multiselect(
            "Select measurement columns",
            options=candidate_columns,
            default=default_measurement_cols,
        )

        show_cleaning = st.checkbox("Show data cleaning report", value=True)
        show_summary = st.checkbox("Show per-part summary statistics", value=True)

        if selected_cols:
            # Clean and prepare data
            data_dict, cleaning_report, summary_stats, fig_cleaning = clean_data(
                df, measurement_cols=selected_cols, part_col='Comp_Name' if 'Comp_Name' in df.columns else None
            )

            if show_cleaning:
                st.subheader("Data Cleaning Report")
                for line in cleaning_report:
                    st.write(str(line))
                st.pyplot(fig_cleaning, clear_figure=True)

            # Perform analysis per measurement
            for measurement_name, data in data_dict.items():
                st.header(f"Analysis for {measurement_name}")
                results, anova_table = perform_gage_rr(data)

                var_part = results['Variance Components']['Part']
                var_gage_rr = results['Variance Components']['Repeatability']
                ndc = calculate_ndc(var_part, var_gage_rr)
                results['Number of Distinct Categories'] = ndc

                # ANOVA table
                st.subheader("ANOVA Results")
                anova_display = anova_table.copy()
                # Ensure mean square is visible
                anova_display["mean_sq"] = anova_display["sum_sq"] / anova_display["df"]
                st.dataframe(anova_display)

                # Key metrics
                st.subheader("Key Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Repeatability Var", f"{var_gage_rr:.6g}")
                with col2:
                    st.metric("Part Var", f"{var_part:.6g}")
                with col3:
                    st.metric("NDC", "—" if np.isnan(ndc) else str(ndc))

                # Optional summary by part
                if show_summary:
                    st.subheader("Summary by Part")
                    stats_df = summary_stats.get(measurement_name)
                    if stats_df is not None:
                        st.dataframe(stats_df)

                # Plots
                st.subheader("Plots")
                fig = plot_results(data, results, measurement_name)
                st.pyplot(fig, clear_figure=True)

        else:
            st.info("Please select at least one measurement column.")
else:
    st.info("Upload a data file to begin.")