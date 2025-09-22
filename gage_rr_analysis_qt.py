import os
import sys
from typing import Optional
from pathlib import Path

# Set Qt API before importing matplotlib
os.environ['QT_API'] = 'PySide6'

import numpy as np
import pandas as pd
import seaborn as sns

# Configure matplotlib to use PySide6 backend
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend initially
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

# Set matplotlib to be thread-safe
plt.ioff()  # Turn off interactive mode

from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QGridLayout,
    QWidget, QPushButton, QLabel, QFileDialog, QTableView, QTabWidget,
    QTextEdit, QCheckBox, QListWidget, QListWidgetItem, QSplitter,
    QMessageBox, QProgressBar, QGroupBox, QFrame, QScrollArea
)
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, Signal, QThread, QTimer
from PySide6.QtGui import QFont, QPixmap

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
    
    # 7. Create box plots for visual inspection with improved legibility
    n_cols = len(measurement_cols)
    n_rows = (n_cols + 1) // 2  # Ceiling division
    fig_cleaning = plt.figure(figsize=(18, 7 * n_rows))
    
    # Use previously determined part_col for plotting
    for i, col in enumerate(measurement_cols, 1):
        ax = plt.subplot(n_rows, 2, i)
        if part_col is None:
            plt.text(0.5, 0.5, 'No part identifier column found', ha='center', va='center', fontsize=12)
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
            plt.text(0.5, 0.5, f'No valid data for {col}', ha='center', va='center', fontsize=12)
            plt.axis('off')
            continue
        sns.boxplot(x=part_col, y=col, data=plot_df)
        plt.title(f'Distribution of {col} by Part', fontsize=14, fontweight='bold')
        plt.xlabel(part_col, fontsize=12)
        plt.ylabel(col, fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
    
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
    All plots are the same size with improved legibility.
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

    # Start plotting with larger figure and better spacing
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(top=0.92, bottom=0.1, left=0.08, right=0.95, hspace=0.35, wspace=0.25)

    # Title with larger font
    fig.suptitle(f"Gage R&R (ANOVA) Report for {measurement_name}", 
                fontsize=22, fontweight='bold', y=0.96)

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
    ax1.set_xticklabels(metrics, fontsize=12)
    ax1.set_ylabel('Percent', fontsize=12)
    ax1.set_ylim(0, 110)
    ax1.legend(fontsize=11)
    ax1.set_title('Components of Variation', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='both', which='major', labelsize=11)
    ax1.text(0.5, -0.25, 'Note: % Study Var is not additive', transform=ax1.transAxes,
             ha='center', va='top', fontsize=10, color='gray')
    
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
    ax1.text(bars_rep[0].get_x() + bars_rep[0].get_width()/2, bars_rep[0].get_height() + 3,
             f"{rep_c_rounded:.1f}", ha='center', fontsize=11, fontweight='bold')
    ax1.text(bars_part[0].get_x() + bars_part[0].get_width()/2, bars_part[0].get_height() + 3,
             f"{part_c_rounded:.1f}", ha='center', fontsize=11, fontweight='bold')
    # Group 1: % Study Var (not additive; show as-is)
    ax1.text(bars_rep[1].get_x() + bars_rep[1].get_width()/2, bars_rep[1].get_height() + 3,
             f"{rep_vals[1]:.1f}", ha='center', fontsize=11, fontweight='bold')
    ax1.text(bars_part[1].get_x() + bars_part[1].get_width()/2, bars_part[1].get_height() + 3,
             f"{part_vals[1]:.1f}", ha='center', fontsize=11, fontweight='bold')

    # S Chart (Top Right)
    ax2 = axes[0, 1]
    ax2.plot(parts, stds.values, marker='o', markersize=6, linestyle='-', linewidth=2, color='#0070C0')
    if s_UCL is not None:
        ax2.axhline(s_UCL, color='brown', linestyle='-', linewidth=2, label=f'UCL={s_UCL:.5f}')
    if s_CL is not None:
        ax2.axhline(s_CL, color='green', linestyle='-', linewidth=2, label=f'S={s_CL:.5f}')
    if s_LCL is not None:
        ax2.axhline(s_LCL, color='black', linestyle='-', linewidth=2, label=f'LCL={s_LCL:.5f}')
    ax2.set_title('S Chart', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Sample StDev', fontsize=12)
    ax2.set_xticks([])
    ax2.legend(loc='upper left', fontsize=10)
    ax2.tick_params(axis='both', which='major', labelsize=11)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.5f'))

    # Measurement by Part (Bottom Left)
    ax3 = axes[1, 0]
    ax3.plot(parts, means.values, marker='*', markersize=8, linestyle='-', linewidth=2, color='gray')
    ax3.set_title(f'{measurement_name} by {part_col}', fontsize=14, fontweight='bold')
    ax3.set_xlabel(part_col, fontsize=12)
    ax3.set_ylabel('Sample Mean', fontsize=12)
    ax3.tick_params(axis='x', rotation=45, labelsize=10)
    ax3.tick_params(axis='y', labelsize=11)

    # XBar Chart (Bottom Right)
    ax4 = axes[1, 1]
    ax4.plot(parts, means.values, marker='o', markersize=6, linestyle='-', linewidth=2, color='#0070C0')
    if xbar_UCL is not None:
        ax4.axhline(xbar_UCL, color='brown', linestyle='-', linewidth=2, label=f'UCL={xbar_UCL:.5f}')
    ax4.axhline(xbar_cl, color='green', linestyle='-', linewidth=2, label=f'CL={xbar_cl:.5f}')
    if xbar_LCL is not None:
        ax4.axhline(xbar_LCL, color='black', linestyle='-', linewidth=2, label=f'LCL={xbar_LCL:.5f}')
    ax4.set_title('XBar Chart', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Part', fontsize=12)
    ax4.set_ylabel('Sample Mean', fontsize=12)
    ax4.set_xticks([])
    ax4.legend(loc='upper left', fontsize=10)
    ax4.tick_params(axis='both', which='major', labelsize=11)
    ax4.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.5f'))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    # Return figure for UI display rather than saving to disk
    return fig


def calculate_ndc(var_part, var_gage_rr):
    if var_gage_rr == 0:
        return np.nan
    ndc = int(np.floor(1.41 * np.sqrt(var_part) / np.sqrt(var_gage_rr)))
    return ndc


class PandasModel(QAbstractTableModel):
    """Custom model for displaying pandas DataFrames in QTableView."""
    
    def __init__(self, df: pd.DataFrame = None):
        super().__init__()
        self._df = df if df is not None else pd.DataFrame()
    
    def rowCount(self, parent=QModelIndex()):
        return len(self._df)
    
    def columnCount(self, parent=QModelIndex()):
        return len(self._df.columns)
    
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        
        if role == Qt.DisplayRole:
            value = self._df.iloc[index.row(), index.column()]
            return str(value)
        
        return None
    
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._df.columns[section])
            else:
                return str(self._df.index[section])
        return None
    
    def update_data(self, df: pd.DataFrame):
        """Update the model with new DataFrame."""
        self.beginResetModel()
        self._df = df
        self.endResetModel()


class MatplotlibWidget(QWidget):
    """Widget for embedding matplotlib plots in Qt."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(16, 12), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def plot(self, figure):
        """Display a matplotlib figure by copying its content."""
        self.figure.clear()
        
        # Copy figure content properly
        try:
            # Get the figure's layout and copy all subplots
            if hasattr(figure, '_suptitle') and figure._suptitle:
                self.figure.suptitle(figure._suptitle.get_text(), 
                                    fontsize=figure._suptitle.get_fontsize(),
                                    fontweight=figure._suptitle.get_fontweight())
            
            # Copy all axes from original figure
            for i, ax_orig in enumerate(figure.axes):
                # Determine subplot position
                rows = 2 if len(figure.axes) > 2 else 1
                cols = 2 if len(figure.axes) > 1 else 1
                ax_new = self.figure.add_subplot(rows, cols, i + 1)
                
                # Copy plot content using the original data
                for line in ax_orig.get_lines():
                    ax_new.plot(line.get_xdata(), line.get_ydata(), 
                              color=line.get_color(), marker=line.get_marker(),
                              linestyle=line.get_linestyle(), linewidth=line.get_linewidth())
                
                # Copy bar plots
                for patch in ax_orig.patches:
                    if hasattr(patch, 'get_height'):  # Bar patch
                        ax_new.bar(patch.get_x() + patch.get_width()/2, patch.get_height(),
                                 width=patch.get_width(), color=patch.get_facecolor(),
                                 alpha=patch.get_alpha())
                
                # Copy horizontal lines (axhlines)
                for line in ax_orig.lines:
                    if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1]:
                        continue  # Skip vertical lines for now
                    if len(line.get_ydata()) == 2 and line.get_ydata()[0] == line.get_ydata()[1]:
                        # Horizontal line
                        ax_new.axhline(line.get_ydata()[0], color=line.get_color(),
                                     linestyle=line.get_linestyle(), 
                                     label=line.get_label() if line.get_label() and not line.get_label().startswith('_') else None)
                
                # Copy titles and labels
                ax_new.set_title(ax_orig.get_title())
                ax_new.set_xlabel(ax_orig.get_xlabel())
                ax_new.set_ylabel(ax_orig.get_ylabel())
                
                # Copy axis limits
                ax_new.set_xlim(ax_orig.get_xlim())
                ax_new.set_ylim(ax_orig.get_ylim())
                
                # Copy ticks
                ax_new.set_xticks(ax_orig.get_xticks())
                ax_new.set_xticklabels([t.get_text() for t in ax_orig.get_xticklabels()],
                                     rotation=ax_orig.get_xticklabels()[0].get_rotation() if ax_orig.get_xticklabels() else 0)
                
                # Copy legend if present
                legend = ax_orig.get_legend()
                if legend:
                    ax_new.legend(loc=legend._loc if hasattr(legend, '_loc') else 'best')
                
                # Copy text annotations
                for text in ax_orig.texts:
                    ax_new.text(text.get_position()[0], text.get_position()[1], 
                              text.get_text(), ha=text.get_ha(), va=text.get_va(),
                              fontsize=text.get_fontsize(), color=text.get_color())
            
            self.figure.tight_layout()
            
        except Exception as e:
            # Fallback to image-based copying if direct copying fails
            print(f"Direct copying failed, using image fallback: {e}")
            self._plot_as_image(figure)
        
        self.canvas.draw()
    
    def _plot_as_image(self, figure):
        """Fallback method to display figure as image."""
        import io
        from PIL import Image
        
        buf = io.BytesIO()
        figure.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.axis('off')
        
        img = Image.open(buf)
        ax.imshow(img)
        buf.close()
    
    def clear(self):
        """Clear the plot."""
        self.figure.clear()
        self.canvas.draw()


class AnalysisWorker(QThread):
    """Worker thread for running analysis to avoid blocking UI."""
    
    finished = Signal(dict, object, object)  # results, anova_table, figure
    error = Signal(str)
    
    def __init__(self, data, measurement_name):
        super().__init__()
        self.data = data
        self.measurement_name = measurement_name
    
    def run(self):
        try:
            results, anova_table = perform_gage_rr(self.data)
            var_part = results['Variance Components']['Part']
            var_gage_rr = results['Variance Components']['Repeatability']
            ndc = calculate_ndc(var_part, var_gage_rr)
            results['Number of Distinct Categories'] = ndc
            
            figure = plot_results(self.data, results, self.measurement_name)
            
            self.finished.emit(results, anova_table, figure)
            
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.df = None
        self.data_dict = {}
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("Gage R&R Analysis Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # File loading section
        file_group = QGroupBox("Data File")
        file_layout = QHBoxLayout(file_group)
        
        self.load_button = QPushButton("Load Data File")
        self.load_button.clicked.connect(self.load_file)
        self.file_label = QLabel("No file loaded")
        
        file_layout.addWidget(self.load_button)
        file_layout.addWidget(self.file_label)
        file_layout.addStretch()
        
        main_layout.addWidget(file_group)
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Controls
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Results
        self.results_tabs = QTabWidget()
        splitter.addWidget(self.results_tabs)
        
        # Set splitter proportions
        splitter.setSizes([400, 1000])
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def create_control_panel(self):
        """Create the left control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Data preview
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.data_table = QTableView()
        self.data_model = PandasModel()
        self.data_table.setModel(self.data_model)
        self.data_table.setMaximumHeight(200)
        
        preview_layout.addWidget(self.data_table)
        layout.addWidget(preview_group)
        
        # Column selection
        column_group = QGroupBox("Measurement Columns")
        column_layout = QVBoxLayout(column_group)
        
        self.column_list = QListWidget()
        self.column_list.setSelectionMode(QListWidget.MultiSelection)
        column_layout.addWidget(self.column_list)
        
        layout.addWidget(column_group)
        
        # Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)
        
        self.show_cleaning_cb = QCheckBox("Show data cleaning report")
        self.show_cleaning_cb.setChecked(True)
        
        self.show_summary_cb = QCheckBox("Show per-part summary statistics")
        self.show_summary_cb.setChecked(True)
        
        options_layout.addWidget(self.show_cleaning_cb)
        options_layout.addWidget(self.show_summary_cb)
        
        layout.addWidget(options_group)
        
        # Analyze button
        self.analyze_button = QPushButton("Run Analysis")
        self.analyze_button.clicked.connect(self.run_analysis)
        self.analyze_button.setEnabled(False)
        layout.addWidget(self.analyze_button)
        
        # Export PDF button
        self.export_pdf_button = QPushButton("Export Report to PDF")
        self.export_pdf_button.clicked.connect(self.export_to_pdf)
        self.export_pdf_button.setEnabled(False)
        layout.addWidget(self.export_pdf_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        layout.addStretch()
        return panel
    
    def load_file(self):
        """Load data file using file dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Load Data File", 
            "", 
            "Data Files (*.csv *.tsv *.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                self.df = read_data(file_path)
                self.current_file_name = Path(file_path).name
                self.file_label.setText(f"Loaded: {self.current_file_name}")
                self.statusBar().showMessage(f"Loaded {len(self.df)} rows")
                
                # Update data preview
                self.data_model.update_data(self.df.head(50))
                self.data_table.resizeColumnsToContents()
                
                # Update column selection
                self.update_column_list()
                
                self.analyze_button.setEnabled(True)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")
    
    def update_column_list(self):
        """Update the measurement column selection list."""
        if self.df is None:
            return
        
        self.column_list.clear()
        
        default_cols = _infer_measurement_columns(self.df)
        
        for col in default_cols:
            item = QListWidgetItem(col)
            item.setSelected(True)
            self.column_list.addItem(item)
    
    def get_selected_columns(self):
        """Get the selected measurement columns."""
        selected = []
        for i in range(self.column_list.count()):
            item = self.column_list.item(i)
            if item.isSelected():
                selected.append(item.text())
        return selected
    
    def run_analysis(self):
        """Run the Gage R&R analysis."""
        if self.df is None:
            return
        
        selected_cols = self.get_selected_columns()
        if not selected_cols:
            QMessageBox.warning(self, "Warning", "Please select at least one measurement column.")
            return
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.analyze_button.setEnabled(False)
        self.statusBar().showMessage("Running analysis...")
        
        try:
            # Clean and prepare data
            part_col = 'Comp_Name' if 'Comp_Name' in self.df.columns else None
            self.data_dict, cleaning_report, summary_stats, fig_cleaning = clean_data(
                self.df, measurement_cols=selected_cols, part_col=part_col
            )
            
            # Store for PDF export
            self.cleaning_report = cleaning_report
            self.fig_cleaning = fig_cleaning
            self.summary_stats = summary_stats
            
            # Clear existing results
            self.results_tabs.clear()
            
            # Add cleaning report if requested
            if self.show_cleaning_cb.isChecked():
                self.add_cleaning_report_tab(cleaning_report, fig_cleaning)
            
            # Run analysis for each measurement
            self.current_analysis = 0
            self.total_analyses = len(self.data_dict)
            self.analysis_results = {}
            
            if self.data_dict:
                self.run_next_analysis()
            else:
                self.finish_analysis()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")
            self.finish_analysis()
    
    def run_next_analysis(self):
        """Run analysis for the next measurement."""
        if self.current_analysis >= len(self.data_dict):
            self.finish_analysis()
            return
        
        measurement_name = list(self.data_dict.keys())[self.current_analysis]
        data = self.data_dict[measurement_name]
        
        # Create worker thread
        self.worker = AnalysisWorker(data, measurement_name)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.start()
    
    def on_analysis_finished(self, results, anova_table, figure):
        """Handle completed analysis."""
        measurement_name = list(self.data_dict.keys())[self.current_analysis]
        
        # Store results
        self.analysis_results[measurement_name] = {
            'results': results,
            'anova_table': anova_table,
            'figure': figure
        }
        
        # Add results tab
        self.add_analysis_tab(measurement_name, results, anova_table, figure)
        
        # Move to next analysis
        self.current_analysis += 1
        self.run_next_analysis()
    
    def on_analysis_error(self, error_msg):
        """Handle analysis error."""
        QMessageBox.critical(self, "Analysis Error", f"Analysis failed: {error_msg}")
        self.current_analysis += 1
        self.run_next_analysis()
    
    def finish_analysis(self):
        """Finish the analysis process."""
        self.progress_bar.setVisible(False)
        self.analyze_button.setEnabled(True)
        self.export_pdf_button.setEnabled(bool(self.analysis_results))
        self.statusBar().showMessage("Analysis complete")
    
    def add_cleaning_report_tab(self, cleaning_report, fig_cleaning):
        """Add data cleaning report tab."""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        
        # Text report
        report_text = QTextEdit()
        report_text.setPlainText('\n'.join(str(line) for line in cleaning_report))
        report_text.setMaximumHeight(200)
        layout.addWidget(report_text)
        
        # Cleaning plots
        plot_widget = MatplotlibWidget()
        plot_widget.plot(fig_cleaning)
        layout.addWidget(plot_widget)
        
        self.results_tabs.addTab(tab_widget, "Data Cleaning")
    
    def add_analysis_tab(self, measurement_name, results, anova_table, figure):
        """Add analysis results tab."""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        
        # Create scroll area for content
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # ANOVA results
        anova_group = QGroupBox("ANOVA Results")
        anova_layout = QVBoxLayout(anova_group)
        
        anova_table_display = anova_table.copy()
        anova_table_display["mean_sq"] = anova_table_display["sum_sq"] / anova_table_display["df"]
        
        anova_model = PandasModel(anova_table_display)
        anova_view = QTableView()
        anova_view.setModel(anova_model)
        anova_view.setMaximumHeight(150)
        anova_view.resizeColumnsToContents()
        
        anova_layout.addWidget(anova_view)
        scroll_layout.addWidget(anova_group)
        
        # Key metrics
        metrics_group = QGroupBox("Key Metrics")
        metrics_layout = QGridLayout(metrics_group)
        
        var_gage_rr = results['Variance Components']['Repeatability']
        var_part = results['Variance Components']['Part']
        ndc = results.get('Number of Distinct Categories', 'N/A')
        
        metrics_layout.addWidget(QLabel("Repeatability Variance:"), 0, 0)
        metrics_layout.addWidget(QLabel(f"{var_gage_rr:.6g}"), 0, 1)
        metrics_layout.addWidget(QLabel("Part Variance:"), 1, 0)
        metrics_layout.addWidget(QLabel(f"{var_part:.6g}"), 1, 1)
        metrics_layout.addWidget(QLabel("Number of Distinct Categories:"), 2, 0)
        metrics_layout.addWidget(QLabel(str(ndc) if not np.isnan(ndc) else "—"), 2, 1)
        
        scroll_layout.addWidget(metrics_group)
        
        # Plots
        plots_group = QGroupBox("Analysis Plots")
        plots_layout = QVBoxLayout(plots_group)
        
        plot_widget = MatplotlibWidget()
        plot_widget.plot(figure)
        plots_layout.addWidget(plot_widget)
        
        scroll_layout.addWidget(plots_group)
        
        # Setup scroll area
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        self.results_tabs.addTab(tab_widget, measurement_name)
    
    def export_to_pdf(self):
        """Export the complete analysis report to PDF."""
        if not self.analysis_results:
            QMessageBox.warning(self, "Warning", "No analysis results to export.")
            return
        
        # Get save location from user
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Report to PDF",
            "gage_rr_analysis_report.pdf",
            "PDF Files (*.pdf);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.statusBar().showMessage("Generating PDF report...")
            
            # Create PDF report
            self.create_pdf_report(file_path)
            
            QMessageBox.information(self, "Success", f"Report successfully exported to:\n{file_path}")
            self.statusBar().showMessage("PDF export complete")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export PDF: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
    
    def create_pdf_report(self, file_path):
        """Create a comprehensive PDF report with all analysis results."""
        from datetime import datetime
        
        with PdfPages(file_path) as pdf:
            # Cover page
            self.create_cover_page(pdf)
            
            # Data cleaning report if available
            if hasattr(self, 'cleaning_report') and hasattr(self, 'fig_cleaning'):
                self.create_cleaning_page(pdf, self.cleaning_report, self.fig_cleaning)
            
            # Analysis results for each measurement
            for measurement_name, analysis_data in self.analysis_results.items():
                self.create_analysis_pages(pdf, measurement_name, analysis_data)
            
            # Metadata
            d = pdf.infodict()
            d['Title'] = 'Gage R&R Analysis Report'
            d['Author'] = 'Gage R&R Analysis Tool'
            d['Subject'] = 'Statistical Analysis Report'
            d['Keywords'] = 'Gage R&R, ANOVA, Quality Control'
            d['CreationDate'] = datetime.now()
    
    def create_cover_page(self, pdf):
        """Create the cover page for the PDF report."""
        fig = Figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.8, 'Gage R&R Analysis Report', 
                ha='center', va='center', fontsize=28, fontweight='bold')
        
        # Subtitle
        ax.text(0.5, 0.7, 'ANOVA-based Measurement System Analysis',
                ha='center', va='center', fontsize=16, style='italic')
        
        # Generated info
        from datetime import datetime
        ax.text(0.5, 0.6, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                ha='center', va='center', fontsize=12)
        
        # Summary info
        if self.df is not None:
            ax.text(0.5, 0.5, f'Data file: {getattr(self, "current_file_name", "Unknown")}',
                    ha='center', va='center', fontsize=12)
            ax.text(0.5, 0.45, f'Total measurements analyzed: {len(self.analysis_results)}',
                    ha='center', va='center', fontsize=12)
            ax.text(0.5, 0.4, f'Data points: {len(self.df)} rows',
                    ha='center', va='center', fontsize=12)
        
        # Add logo area (placeholder)
        ax.text(0.5, 0.2, 'Quality Analysis Tools',
                ha='center', va='center', fontsize=14, color='gray')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def create_cleaning_page(self, pdf, cleaning_report, fig_cleaning):
        """Create data cleaning report page."""
        # Text report page
        fig = Figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        ax.text(0.5, 0.95, 'Data Cleaning Report', 
                ha='center', va='top', fontsize=20, fontweight='bold')
        
        # Format cleaning report text
        report_text = '\n'.join(str(line) for line in cleaning_report)
        ax.text(0.05, 0.85, report_text, ha='left', va='top', fontsize=10,
                fontfamily='monospace', wrap=True)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Add cleaning plots
        if fig_cleaning:
            pdf.savefig(fig_cleaning, bbox_inches='tight')
    
    def create_analysis_pages(self, pdf, measurement_name, analysis_data):
        """Create analysis pages for a specific measurement."""
        results = analysis_data['results']
        anova_table = analysis_data['anova_table']
        figure = analysis_data['figure']
        
        # Summary page
        fig = Figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, f'Analysis Results: {measurement_name}', 
                ha='center', va='top', fontsize=18, fontweight='bold')
        
        # Key metrics table
        y_pos = 0.85
        ax.text(0.1, y_pos, 'Key Metrics:', fontsize=14, fontweight='bold')
        y_pos -= 0.05
        
        var_part = results['Variance Components']['Part']
        var_gage_rr = results['Variance Components']['Repeatability']
        var_total = results['Variance Components']['Total']
        ndc = results.get('Number of Distinct Categories', 'N/A')
        
        metrics_text = f"""
Variance Components:
  • Part Variance: {var_part:.6g}
  • Repeatability Variance: {var_gage_rr:.6g}
  • Total Variance: {var_total:.6g}

Percent Contributions:
  • Part: {results['Percent Contribution']['Part']:.1f}%
  • Repeatability: {results['Percent Contribution']['Repeatability']:.1f}%

Study Variation:
  • Part: {results['Study Variation']['Part']:.6g}
  • Repeatability: {results['Study Variation']['Repeatability']:.6g}
  • Total: {results['Study Variation']['Total']:.6g}

Number of Distinct Categories (NDC): {ndc if not np.isnan(ndc) else "N/A"}
        """
        
        ax.text(0.1, y_pos, metrics_text, ha='left', va='top', fontsize=11,
                fontfamily='monospace')
        
        # ANOVA table
        y_pos = 0.4
        ax.text(0.1, y_pos, 'ANOVA Table:', fontsize=14, fontweight='bold')
        y_pos -= 0.05
        
        # Format ANOVA table
        anova_display = anova_table.copy()
        anova_display["mean_sq"] = anova_display["sum_sq"] / anova_display["df"]
        anova_text = anova_display.to_string()
        
        ax.text(0.1, y_pos, anova_text, ha='left', va='top', fontsize=9,
                fontfamily='monospace')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Add the analysis plots
        pdf.savefig(figure, bbox_inches='tight')


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Gage R&R Analysis")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Quality Analysis Tools")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
