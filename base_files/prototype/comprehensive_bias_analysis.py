# Set global plotting parameters for publication quality with consistent design
#!/usr/bin/env python3
"""
Modified Bias Tables Visualization Generator
Excludes comorbidity status and symptom presentation from all visualizations

Creates visualizations for remaining bias categories: Age, Drug Use, Gender, Smoking, and Occupation
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set global plotting parameters for publication quality with consistent design
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'figure.constrained_layout.use': True
})

# Unified color palette for consistency across all visualizations
# Custom color palette using specified colors
# Custom color palette using specified colors
CUSTOM_COLORS = ['#8f0707', '#be6c65', '#d7c2c1', '#de99a1', '#de6e8c']

# Unified color palette for consistency across all visualizations
COLORS = {
    'primary': '#8f0707',      # Dark red (main categories)
    'secondary': '#be6c65',    # Medium red (secondary categories)
    'accent': '#de99a1',       # Light pink (highlights/accents)
    'warning': '#de6e8c',      # Pink (high bias/warnings)
    'success': '#d7c2c1',      # Light beige (positive outcomes)
    'neutral': '#be6c65',      # Medium red-brown (neutral/unknown)
    'background': '#F8F9FA',   # Light background for readability
}

# Color schemes for different category types using custom palette
AGE_COLORS = ['#8f0707', '#be6c65', '#d7c2c1', '#de99a1', '#de6e8c', '#8f0707', '#be6c65']
DRUG_COLORS = ['#8f0707', '#be6c65', '#d7c2c1']
GENDER_COLORS = ['#de99a1', '#8f0707', '#be6c65']
SMOKING_COLORS = ['#d7c2c1', '#8f0707', '#be6c65']
OCCUPATION_COLORS = ['#8f0707', '#be6c65', '#d7c2c1', '#de99a1', '#de6e8c', '#8f0707']

def load_filtered_table_data():
    """Load bias table data excluding comorbidity status and symptom presentation"""
    
    # High-Level Bias Categories (0.2-0.3) - EXCLUDING SYMPTOM PRESENTATION
    high_data = {
        'Category': ['Age', 'Age', 'Age', 'Age', 'Age', 'Age', 'Age',
                    'Drug Use', 'Drug Use', 'Drug Use'],
        'Group': ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60+',
                 'Drug User', 'Non-drug User', 'Unknown'],
        'N': [36, 21, 38, 29, 21, 33, 36,
              7, 38, 169],
        'Accuracy': [0.639, 0.524, 0.500, 0.552, 0.619, 0.727, 0.611,
                    0.714, 0.368, 0.645],
        'CI_Lower': [0.489, 0.328, 0.348, 0.382, 0.426, 0.579, 0.459,
                    0.428, 0.221, 0.574],
        'CI_Upper': [0.789, 0.720, 0.652, 0.722, 0.812, 0.875, 0.763,
                    1.000, 0.515, 0.716],
        'Confidence': [81.7, 82.1, 77.2, 82.2, 82.1, 84.1, 81.0,
                      80.7, 77.2, 82.2],
        'Parity_Gap': [0.139, 0.024, 0.000, 0.052, 0.119, 0.227, 0.111,
                      0.069, -0.277, 0.000]
    }
    
    # Moderate-Level Bias Categories (0.1-0.2)
    moderate_data = {
        'Category': ['Gender', 'Gender', 'Gender',
                    'Smoking', 'Smoking', 'Smoking',
                    'Occupation Type', 'Occupation Type', 'Occupation Type', 'Occupation Type', 'Occupation Type', 'Occupation Type'],
        'Group': ['Female', 'Male', 'Other',
                 'Non-smoker', 'Smoker', 'Unknown',
                 'Knowledge Worker', 'Manual Labor', 'Retired', 'Student', 'Unemployed', 'Unknown'],
        'N': [92, 115, 7,
              94, 30, 90,
              28, 11, 18, 38, 2, 117],
        'Accuracy': [0.500, 0.661, 0.857,
                    0.585, 0.733, 0.567,
                    0.607, 0.727, 0.556, 0.605, 0.500, 0.590],
        'CI_Lower': [0.400, 0.576, 0.593,
                    0.487, 0.580, 0.467,
                    0.436, 0.487, 0.347, 0.456, 0.095, 0.502],
        'CI_Upper': [0.600, 0.746, 1.121,
                    0.683, 0.886, 0.667,
                    0.778, 0.967, 0.765, 0.754, 0.905, 0.678],
        'Confidence': [80.6, 81.7, 85.0,
                      79.9, 84.2, 81.8,
                      81.8, 83.2, 77.2, 81.1, 90.0, 81.6],
        'Parity_Gap': [-0.161, 0.000, 0.196,
                      0.000, 0.148, -0.018,
                      0.017, 0.137, -0.034, 0.015, -0.090, 0.000]
    }
    
    return high_data, moderate_data

def create_age_analysis_visualization():
    """Create detailed age group analysis visualization"""
    
    high_data, _ = load_filtered_table_data()
    df = pd.DataFrame(high_data)
    age_df = df[df['Category'] == 'Age']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Age Group Bias Analysis\nDetailed Performance and Bias Assessment', 
                fontsize=16, fontweight='bold')
    
    # Color scheme for age groups - using consistent palette
    age_colors = AGE_COLORS
    age_midpoints = [5, 15, 25, 35, 45, 55, 70]
    
    # 1. Age Group Performance with Trend Analysis
    ax1 = axes[0, 0]
    
    # Main trend line with confidence bands
    line = ax1.plot(age_midpoints, age_df['Accuracy'], 'o-', color=COLORS['primary'], 
                   linewidth=3, markersize=10, markerfacecolor=COLORS['secondary'], 
                   markeredgecolor='white', markeredgewidth=2, label='Observed Accuracy')
    
    # Confidence bands
    ci_lower = age_df['CI_Lower'].values
    ci_upper = age_df['CI_Upper'].values
    ax1.fill_between(age_midpoints, ci_lower, ci_upper, alpha=0.2, color=COLORS['secondary'], label='95% CI')
    
    # Polynomial trend fitting
    z = np.polyfit(age_midpoints, age_df['Accuracy'], 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(5, 70, 100)
    ax1.plot(x_smooth, p(x_smooth), '--', color=COLORS['warning'], alpha=0.8, linewidth=2, label='Quadratic Trend')
    
    # Statistical annotations with consistent styling
    from scipy.stats import pearsonr
    corr, p_value = pearsonr(age_midpoints, age_df['Accuracy'])
    r_squared = corr**2
    
    ax1.text(0.05, 0.95, f'R² = {r_squared:.3f}\nCorr = {corr:.3f}\np = {p_value:.3f}', 
             transform=ax1.transAxes, va='top', ha='left',
             bbox=dict(boxstyle="round,pad=0.4", facecolor=COLORS['background'], 
                      edgecolor=COLORS['primary'], alpha=0.9), 
             fontweight='bold', fontsize=11)
    
    # Sample size annotations with consistent font
    for x, acc, n in zip(age_midpoints, age_df['Accuracy'], age_df['N']):
        ax1.annotate(f'n={n}', (x, acc), textcoords="offset points", xytext=(0,-25), 
                    ha='center', fontsize=9, alpha=0.7, color=COLORS['neutral'])
    
    ax1.set_xlabel('Age Group (Years)', fontweight='bold')
    ax1.set_ylabel('Top-1 Accuracy', fontweight='bold')
    ax1.set_title('Age Group Performance Trend', fontweight='bold', color=COLORS['primary'])
    ax1.set_xticks(age_midpoints)
    ax1.set_xticklabels(age_df['Group'])
    ax1.grid(True, alpha=0.3, color=COLORS['neutral'])
    ax1.legend(loc='lower right')
    ax1.set_ylim(0.4, 0.9)
    
    # 2. Sample Size Distribution with consistent styling
    ax2 = axes[0, 1]
    
    bars = ax2.bar(age_df['Group'], age_df['N'], color=age_colors, alpha=0.8)
    
    for bar, n in zip(bars, age_df['N']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{n}', ha='center', va='bottom', fontweight='bold', fontsize=11, color=COLORS['primary'])
    
    ax2.set_ylabel('Sample Size', fontweight='bold')
    ax2.set_title('Age Group Sample Distribution', fontweight='bold', color=COLORS['primary'])
    ax2.set_xticklabels(age_df['Group'], rotation=45, ha='right')
    
    # 3. Parity Gap Analysis with consistent color scheme
    ax3 = axes[1, 0]
    
    # Color bars by parity gap direction and magnitude using consistent palette
    bar_colors = []
    for gap in age_df['Parity_Gap']:
        if gap == 0:
            bar_colors.append(COLORS['neutral'])  # Neutral for baseline
        elif gap > 0.1:
            bar_colors.append(COLORS['warning'])  # Red for high positive bias
        elif gap > 0:
            bar_colors.append(COLORS['accent'])  # Orange for moderate positive bias
        else:
            bar_colors.append(COLORS['success'])  # Blue for negative bias
    
    bars = ax3.bar(age_df['Group'], np.abs(age_df['Parity_Gap']), color=bar_colors, alpha=0.8)
    
    # Add value labels with direction indicators
    for i, (bar, gap) in enumerate(zip(bars, age_df['Parity_Gap'])):
        height = bar.get_height()
        direction = '→' if gap == 0 else '↑' if gap > 0 else '↓'
        color = COLORS['neutral'] if gap == 0 else COLORS['success'] if gap > 0 else COLORS['warning']
        
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{abs(gap):.3f} {direction}', ha='center', va='bottom', 
                fontweight='bold', color=color, fontsize=10)
    
    # Add threshold lines with consistent colors
    ax3.axhline(y=0.2, color=COLORS['accent'], linestyle='--', alpha=0.7, label='High (0.2)', linewidth=2)
    ax3.axhline(y=0.3, color=COLORS['warning'], linestyle='--', alpha=0.7, label='Critical (0.3)', linewidth=2)
    
    ax3.set_ylabel('Absolute Parity Gap', fontweight='bold')
    ax3.set_title('Age Group Bias Magnitude', fontweight='bold', color=COLORS['primary'])
    ax3.set_xticklabels(age_df['Group'], rotation=45, ha='right')
    ax3.legend()
    
    # 4. Performance Statistics Table with consistent styling
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    table_data = []
    for i, row in age_df.iterrows():
        ci_width = row['CI_Upper'] - row['CI_Lower']
        table_data.append([
            row['Group'],
            f"{row['N']}",
            f"{row['Accuracy']:.3f}",
            f"±{ci_width/2:.3f}",
            f"{row['Confidence']:.1f}%",
            f"{row['Parity_Gap']:+.3f}"
        ])
    
    headers = ['Age Group', 'N', 'Accuracy', 'CI ±', 'Confidence', 'Parity Gap']
    
    table = ax4.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.0)
    
    # Color code header with consistent colors
    for i in range(len(headers)):
        table[(0, i)].set_facecolor(COLORS['background'])
        table[(0, i)].set_text_props(weight='bold', color=COLORS['primary'])
    
    # Highlight highest bias with consistent color
    highest_bias_idx = age_df['Parity_Gap'].abs().idxmax() + 1
    for j in range(len(headers)):
        table[(highest_bias_idx, j)].set_facecolor('#FFE6E6')
    
    ax4.set_title('Age Group Summary Statistics', fontweight='bold', pad=20, color=COLORS['primary'])
    
    plt.tight_layout()
    plt.savefig('age_group_analysis_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_drug_use_analysis_visualization():
    """Create detailed drug use analysis visualization with fixed label positioning"""
    
    high_data, _ = load_filtered_table_data()
    df = pd.DataFrame(high_data)
    drug_df = df[df['Category'] == 'Drug Use']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Drug Use Status Bias Analysis\nPerformance Impact and Bias Assessment', 
                fontsize=16, fontweight='bold', color=COLORS['primary'])
    
    # Color scheme for drug use groups - using consistent palette
    drug_colors = DRUG_COLORS
    
    # 1. Drug Use Performance Comparison - FIXED LABEL POSITIONING
    ax1 = axes[0, 0]
    
    bars = ax1.bar(drug_df['Group'], drug_df['Accuracy'], color=drug_colors, alpha=0.8)
    
    # Add confidence intervals
    error_lower = drug_df['Accuracy'] - drug_df['CI_Lower']
    error_upper = drug_df['CI_Upper'] - drug_df['Accuracy']
    ax1.errorbar(drug_df['Group'], drug_df['Accuracy'], yerr=[error_lower, error_upper], 
                fmt='none', color=COLORS['primary'], capsize=5, linewidth=2)
    
    # Value labels with effect size calculation - MOVED HIGHER
    baseline_acc = drug_df[drug_df['Group'] == 'Unknown']['Accuracy'].iloc[0]
    for bar, acc, n, group, ci_upper_val in zip(bars, drug_df['Accuracy'], drug_df['N'], drug_df['Group'], drug_df['CI_Upper']):
        height = bar.get_height()
        effect_size = acc - baseline_acc
        # Calculate the top of the error bar and add extra spacing
        error_bar_top = ci_upper_val
        # Special case for Drug User to prevent escaping from graph
        if group == 'Drug User':
            label_y_position = error_bar_top + 0.08  # Reduced spacing for Drug User
        else:
            label_y_position = error_bar_top + 0.12  # Standard spacing for others
        
        ax1.text(bar.get_x() + bar.get_width()/2., label_y_position,
                f'{acc:.3f}\n(n={n})\nΔ={effect_size:+.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11, color=COLORS['primary'])
    
    ax1.set_ylabel('Top-1 Accuracy', fontweight='bold')
    ax1.set_title('Drug Use Status Performance', fontweight='bold', color=COLORS['primary'])
    ax1.set_xticklabels(drug_df['Group'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, color=COLORS['neutral'])
    ax1.set_ylim(0.2, 1.2)  # Increased upper limit to accommodate higher labels
    
    # 2. Sample Size Distribution with Bar Chart
    ax2 = axes[0, 1]
    
    # Use bar chart instead of pie chart to avoid matplotlib version issues
    bars = ax2.bar(drug_df['Group'], drug_df['N'], color=drug_colors)
    
    # Add percentage labels - positioned above bars
    total_samples = drug_df['N'].sum()
    for bar, n, group in zip(bars, drug_df['N'], drug_df['Group']):
        height = bar.get_height()
        percentage = (n / total_samples) * 100
        ax2.text(bar.get_x() + bar.get_width()/2., height + 5,  # Increased spacing
                f'{n}\n({percentage:.1f}%)', ha='center', va='bottom', 
                fontweight='bold', fontsize=11, color=COLORS['primary'])
    
    ax2.set_ylabel('Sample Size', fontweight='bold')
    ax2.set_title('Sample Distribution by Drug Use Status', fontweight='bold', color=COLORS['primary'])
    ax2.set_xticklabels(drug_df['Group'], rotation=45, ha='right')
    ax2.set_ylim(0, max(drug_df['N']) * 1.3)  # Add more space at top
    
    # 3. Bias Magnitude and Direction with consistent styling
    ax3 = axes[1, 0]
    
    # Create horizontal bar chart for better visibility
    y_positions = np.arange(len(drug_df))
    
    # Color by bias direction using consistent palette
    bar_colors_bias = []
    for gap in drug_df['Parity_Gap']:
        if gap > 0.05:
            bar_colors_bias.append(COLORS['warning'])  # Red for positive bias
        elif gap < -0.05:
            bar_colors_bias.append(COLORS['success'])  # Blue for negative bias
        else:
            bar_colors_bias.append(COLORS['neutral'])  # Gray for neutral
    
    bars = ax3.barh(y_positions, drug_df['Parity_Gap'], color=bar_colors_bias, alpha=0.8)
    
    # Add value labels
    for i, (bar, gap) in enumerate(zip(bars, drug_df['Parity_Gap'])):
        width = bar.get_width()
        x_pos = width + 0.02 if width >= 0 else width - 0.02  # Increased spacing
        ha = 'left' if width >= 0 else 'right'
        ax3.text(x_pos, bar.get_y() + bar.get_height()/2.,
                f'{gap:+.3f}', ha=ha, va='center', fontweight='bold', fontsize=11, color=COLORS['primary'])
    
    # Add reference lines with consistent colors
    ax3.axvline(x=0, color=COLORS['primary'], linestyle='-', alpha=0.5, linewidth=2)
    ax3.axvline(x=0.2, color=COLORS['accent'], linestyle='--', alpha=0.7, label='High (±0.2)', linewidth=2)
    ax3.axvline(x=-0.2, color=COLORS['accent'], linestyle='--', alpha=0.7, linewidth=2)
    
    ax3.set_xlabel('Parity Gap', fontweight='bold')
    ax3.set_title('Bias Direction and Magnitude', fontweight='bold', color=COLORS['primary'])
    ax3.set_yticks(y_positions)
    ax3.set_yticklabels(drug_df['Group'])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='x', color=COLORS['neutral'])
    
    # 4. Confidence Interval Analysis with consistent styling
    ax4 = axes[1, 1]
    
    # Plot confidence intervals as error bars
    for i, (group, acc, ci_low, ci_high) in enumerate(zip(drug_df['Group'], drug_df['Accuracy'], 
                                                         drug_df['CI_Lower'], drug_df['CI_Upper'])):
        # Draw confidence interval
        ax4.errorbar(i, acc, yerr=[[acc - ci_low], [ci_high - acc]], 
                    fmt='o', color=drug_colors[i], capsize=10, capthick=3, 
                    markersize=12, markerfacecolor=drug_colors[i])
        
        # Add accuracy value - positioned higher above the confidence interval
        ci_upper_error = ci_high - acc
        label_y_position = acc + ci_upper_error + 0.08  # Increased spacing
        ax4.text(i, label_y_position, f'{acc:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11, color=COLORS['primary'])
        
        # Add CI width - positioned lower below the confidence interval
        ci_width = ci_high - ci_low
        ax4.text(i, ci_low - 0.08, f'CI: ±{ci_width/2:.3f}', ha='center', va='top', 
                fontsize=9, alpha=0.7, color=COLORS['neutral'])
    
    ax4.set_ylabel('Top-1 Accuracy', fontweight='bold')
    ax4.set_title('Confidence Interval Comparison', fontweight='bold', color=COLORS['primary'])
    ax4.set_xticks(range(len(drug_df)))
    ax4.set_xticklabels(drug_df['Group'], rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, color=COLORS['neutral'])
    ax4.set_ylim(0.0, 1.3)  # Increased upper limit for better label visibility
    
    plt.tight_layout()
    plt.savefig('drug_use_analysis_detailed_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig
def create_demographic_analysis_visualization():
    """Create visualization for gender, smoking, and occupation"""
    
    _, moderate_data = load_filtered_table_data()
    df = pd.DataFrame(moderate_data)
    
    # Add reference line at zero
    ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax3.axvline(x=0.2, color='orange', linestyle='--', alpha=0.7, label='High (±0.2)')
    ax3.axvline(x=-0.2, color='orange', linestyle='--', alpha=0.7)
    
    ax3.set_xlabel('Parity Gap')
    ax3.set_title('Bias Direction and Magnitude', fontweight='bold')
    ax3.set_yticks(y_positions)
    ax3.set_yticklabels(drug_df['Group'])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Confidence Interval Analysis
    ax4 = axes[1, 1]
    
    # Plot confidence intervals as error bars
    for i, (group, acc, ci_low, ci_high) in enumerate(zip(drug_df['Group'], drug_df['Accuracy'], 
                                                         drug_df['CI_Lower'], drug_df['CI_Upper'])):
        # Draw confidence interval
        ax4.errorbar(i, acc, yerr=[[acc - ci_low], [ci_high - acc]], 
                    fmt='o', color=drug_colors[i], capsize=10, capthick=3, 
                    markersize=10, markerfacecolor=drug_colors[i], alpha=0.8)
        
        # Add accuracy value
        ax4.text(i, acc + 0.05, f'{acc:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
        
        # Add CI width
        ci_width = ci_high - ci_low
        ax4.text(i, ci_low - 0.05, f'CI: ±{ci_width/2:.3f}', ha='center', va='top', 
                fontsize=8, alpha=0.7)
    
    ax4.set_ylabel('Top-1 Accuracy')
    ax4.set_title('Confidence Interval Comparison', fontweight='bold')
    ax4.set_xticks(range(len(drug_df)))
    ax4.set_xticklabels(drug_df['Group'], rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig('drug_use_analysis_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_demographic_analysis_visualization():
    """Create visualization for gender, smoking, and occupation"""
    
    _, moderate_data = load_filtered_table_data()
    df = pd.DataFrame(moderate_data)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Demographic Factors Bias Analysis\nGender, Smoking Status, and Occupation Performance', 
                fontsize=16, fontweight='bold', color='black')
    
    # Color schemes using consistent palette
    gender_colors = GENDER_COLORS
    smoking_colors = SMOKING_COLORS
    occupation_colors = OCCUPATION_COLORS
    
    # 1. Gender Analysis with consistent styling
    ax1 = axes[0, 0]
    gender_df = df[df['Category'] == 'Gender']
    
    bars = ax1.bar(gender_df['Group'], gender_df['Accuracy'], color=gender_colors, alpha=0.8)
    
    # Add confidence intervals
    error_lower = gender_df['Accuracy'] - gender_df['CI_Lower']
    error_upper = gender_df['CI_Upper'] - gender_df['Accuracy']
    ax1.errorbar(gender_df['Group'], gender_df['Accuracy'], 
                yerr=[error_lower, error_upper], fmt='none', color=COLORS['primary'], capsize=5, linewidth=2)
    
    for i, (bar, acc, n) in enumerate(zip(bars, gender_df['Accuracy'], gender_df['N'])):
        height = bar.get_height()
        ci_upper = gender_df['CI_Upper'].iloc[i] - acc
        y_pos = height + ci_upper + 0.05
        ax1.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{acc:.3f}\n(n={n})', ha='center', va='bottom', fontweight='bold', 
                fontsize=11, color=COLORS['primary'])
    
    ax1.set_ylabel('Top-1 Accuracy', fontweight='bold')
    ax1.set_title('Gender Performance', fontweight='bold', color='black')
    ax1.set_ylim(0, 1.3)
    ax1.grid(True, alpha=0.3, color=COLORS['neutral'])
    
    # 2. Smoking Analysis with consistent styling
    ax2 = axes[0, 1]
    smoking_df = df[df['Category'] == 'Smoking']
    
    bars = ax2.bar(smoking_df['Group'], smoking_df['Accuracy'], color=smoking_colors, alpha=0.8)
    
    for bar, acc, n in zip(bars, smoking_df['Accuracy'], smoking_df['N']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.3f}\n(n={n})', ha='center', va='bottom', fontweight='bold', 
                fontsize=11, color=COLORS['primary'])
    
    ax2.set_ylabel('Top-1 Accuracy', fontweight='bold')
    ax2.set_title('Smoking Status Performance', fontweight='bold', color='black')
    ax2.set_xticklabels(smoking_df['Group'], rotation=45, ha='right')
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3, color=COLORS['neutral'])
    
    # 3. Occupation Analysis - Horizontal bars for readability with consistent styling
    ax3 = axes[0, 2]
    occupation_df = df[df['Category'] == 'Occupation Type']
    
    y_positions = np.arange(len(occupation_df))
    bars = ax3.barh(y_positions, occupation_df['Accuracy'], color=occupation_colors, alpha=0.8)
    
    for i, (acc, n) in enumerate(zip(occupation_df['Accuracy'], occupation_df['N'])):
        ax3.text(acc + 0.01, i, f'{acc:.3f} (n={n})', va='center', ha='left', 
                fontweight='bold', fontsize=10, color=COLORS['primary'])
    
    ax3.set_xlabel('Top-1 Accuracy', fontweight='bold')
    ax3.set_title('Occupation Type Performance', fontweight='bold', color='black')
    ax3.set_yticks(y_positions)
    ax3.set_yticklabels(occupation_df['Group'], fontsize=10)
    ax3.grid(True, alpha=0.3, axis='x', color=COLORS['neutral'])
    ax3.set_xlim(0, 0.9)
    
    # 4. Parity Gap Comparison with consistent colors
    ax4 = axes[1, 0]
    
    categories = ['Gender', 'Smoking', 'Occupation Type']
    max_biases = []
    category_colors = [COLORS['accent'], COLORS['primary'], COLORS['secondary']]
    
    for cat in categories:
        cat_df = df[df['Category'] == cat]
        max_bias = cat_df['Parity_Gap'].abs().max()
        max_biases.append(max_bias)
    
    bars = ax4.bar(categories, max_biases, color=category_colors, alpha=0.8)
    
    for bar, bias in zip(bars, max_biases):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{bias:.3f}', ha='center', va='bottom', fontweight='bold', 
                fontsize=11, color=COLORS['primary'])
    
    # Add threshold lines with consistent colors
    ax4.axhline(y=0.1, color=COLORS['accent'], linestyle='--', alpha=0.7, label='Moderate (0.1)', linewidth=2)
    ax4.axhline(y=0.2, color=COLORS['warning'], linestyle='--', alpha=0.7, label='High (0.2)', linewidth=2)
    
    ax4.set_ylabel('Maximum Bias Magnitude', fontweight='bold')
    ax4.set_title('Category Bias Comparison', fontweight='bold', color='black')
    ax4.set_xticklabels(categories, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, color=COLORS['neutral'])
    
    # 5. Sample Size vs Accuracy with consistent styling
    ax5 = axes[1, 1]
    
    for i, cat in enumerate(categories):
        cat_df = df[df['Category'] == cat]
        ax5.scatter(cat_df['N'], cat_df['Accuracy'], 
                   c=category_colors[i], label=cat, alpha=0.8, s=80, edgecolors=COLORS['primary'])
    
    ax5.set_xlabel('Sample Size', fontweight='bold')
    ax5.set_ylabel('Top-1 Accuracy', fontweight='bold')
    ax5.set_title('Sample Size vs Accuracy', fontweight='bold', color='black')
    ax5.legend()
    ax5.grid(True, alpha=0.3, color=COLORS['neutral'])
    
    # 6. Performance Range Analysis with consistent styling
    ax6 = axes[1, 2]
    
    perf_ranges = []
    for cat in categories:
        cat_df = df[df['Category'] == cat]
        perf_range = cat_df['Accuracy'].max() - cat_df['Accuracy'].min()
        perf_ranges.append(perf_range)
    
    bars = ax6.bar(categories, perf_ranges, color=category_colors, alpha=0.8)
    
    for bar, range_val in zip(bars, perf_ranges):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{range_val:.3f}', ha='center', va='bottom', fontweight='bold', 
                fontsize=11, color=COLORS['primary'])
    
    ax6.set_ylabel('Performance Range (Max - Min)', fontweight='bold')
    ax6.set_title('Performance Disparity', fontweight='bold', color='black')
    ax6.set_xticklabels(categories, rotation=45, ha='right')
    ax6.grid(True, alpha=0.3, color=COLORS['neutral'])
    
    plt.tight_layout()
    plt.savefig('demographic_analysis_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_overall_summary_visualization():
    """Create overall summary excluding comorbidity and symptom presentation"""
    
    high_data, moderate_data = load_filtered_table_data()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Overall Bias Analysis Summary\nExcluding Comorbidity Status and Symptom Presentation', 
                fontsize=16, fontweight='bold', color='black')
    
    # Combine all data
    all_data = []
    
    # Add high-level bias categories
    high_df = pd.DataFrame(high_data)
    for cat in ['Age', 'Drug Use']:
        cat_df = high_df[high_df['Category'] == cat]
        max_bias = cat_df['Parity_Gap'].abs().max()
        sample_size = cat_df['N'].sum()
        perf_range = cat_df['Accuracy'].max() - cat_df['Accuracy'].min()
        all_data.append([cat, max_bias, sample_size, perf_range, 'High'])
    
    # Add moderate-level bias categories
    moderate_df = pd.DataFrame(moderate_data)
    for cat in ['Gender', 'Smoking', 'Occupation Type']:
        cat_df = moderate_df[moderate_df['Category'] == cat]
        max_bias = cat_df['Parity_Gap'].abs().max()
        sample_size = cat_df['N'].sum()
        perf_range = cat_df['Accuracy'].max() - cat_df['Accuracy'].min()
        all_data.append([cat, max_bias, sample_size, perf_range, 'Moderate'])
    
    summary_df = pd.DataFrame(all_data, columns=['Category', 'Max_Bias', 'Sample_Size', 'Perf_Range', 'Level'])
    
    # Color scheme using consistent palette
    level_colors = {'High': COLORS['warning'], 'Moderate': COLORS['accent']}
    colors = [level_colors[level] for level in summary_df['Level']]
    
    # 1. Bias Magnitude Comparison with consistent styling
    ax1 = axes[0, 0]
    
    bars = ax1.bar(summary_df['Category'], summary_df['Max_Bias'], color=colors, alpha=0.8)
    
    for bar, bias in zip(bars, summary_df['Max_Bias']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{bias:.3f}', ha='center', va='bottom', fontweight='bold', 
                fontsize=11, color=COLORS['primary'])
    
    # Add threshold lines with consistent colors
    ax1.axhline(y=0.1, color=COLORS['accent'], linestyle='--', alpha=0.7, label='Moderate (0.1)', linewidth=2)
    ax1.axhline(y=0.2, color=COLORS['warning'], linestyle='--', alpha=0.7, label='High (0.2)', linewidth=2)
    ax1.axhline(y=0.3, color=COLORS['warning'], linestyle='-', alpha=0.7, label='Critical (0.3)', linewidth=2)
    
    ax1.set_ylabel('Maximum Bias Magnitude', fontweight='bold')
    ax1.set_title('Bias Magnitude by Category', fontweight='bold', color='black')
    ax1.set_xticklabels(summary_df['Category'], rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 0.35)
    ax1.grid(True, alpha=0.3, color=COLORS['neutral'])
    
    # 2. Sample Size Distribution with consistent styling
    ax2 = axes[0, 1]
    
    bars = ax2.bar(summary_df['Category'], summary_df['Sample_Size'], color=colors, alpha=0.8)
    
    for bar, size in zip(bars, summary_df['Sample_Size']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{size}', ha='center', va='bottom', fontweight='bold', 
                fontsize=11, color=COLORS['primary'])
    
    ax2.set_ylabel('Total Sample Size', fontweight='bold')
    ax2.set_title('Sample Distribution by Category', fontweight='bold', color='black')
    ax2.set_xticklabels(summary_df['Category'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, color=COLORS['neutral'])
    
    # 3. Performance Range Analysis with consistent styling
    ax3 = axes[1, 0]
    
    bars = ax3.bar(summary_df['Category'], summary_df['Perf_Range'], color=colors, alpha=0.8)
    
    for bar, range_val in zip(bars, summary_df['Perf_Range']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{range_val:.3f}', ha='center', va='bottom', fontweight='bold', 
                fontsize=11, color=COLORS['primary'])
    
    ax3.set_ylabel('Performance Range (Max - Min)', fontweight='bold')
    ax3.set_title('Performance Disparity by Category', fontweight='bold', color='black')
    ax3.set_xticklabels(summary_df['Category'], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, color=COLORS['neutral'])
    
    # 4. Key Insights Summary with consistent styling
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    insights_text = """
    KEY FINDINGS (FILTERED):
    
    • HIGH BIAS CATEGORIES:
      - Age: 0.227 max bias (50-60 vs 20-30)
      - Drug Use: 0.277 max bias (Non-user vs Drug User)
      
    • MODERATE BIAS CATEGORIES:
      - Gender: 0.196 max bias (Other vs Female)
      - Smoking: 0.148 max bias (Smoker vs baseline)
      - Occupation: 0.137 max bias (Manual vs baseline)
      
    • SAMPLE SIZE CONCERNS:
      - Drug Users: n=7 (very small sample)
      - Gender Other: n=7 (very small sample)
      - Some occupation types: n=2-11
      
    • PATTERN OBSERVATIONS:
      - Age shows U-shaped performance curve
      - Drug use status has largest bias magnitude
      - Unknown status often performs differently
      - Small sample sizes affect CI reliability
    """
    
    ax4.text(0.05, 0.95, insights_text, transform=ax4.transAxes, 
             fontsize=11, va='top', ha='left',
             bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['background'], 
                      edgecolor=COLORS['primary'], alpha=0.9),
             fontweight='normal', color=COLORS['primary'])
    
    ax4.set_title('Summary Insights (Filtered Data)', fontweight='bold', pad=20, color='black')
    
    plt.tight_layout()
    plt.savefig('overall_bias_summary_filtered.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig#!/usr/bin/env python3

def print_filtered_accuracy_differences():
    """Print accuracy differences for filtered bias categories"""
    
    high_data, moderate_data = load_filtered_table_data()
    
    print("\n" + "="*80)
    print("ACCURACY DIFFERENCES BETWEEN SUBCATEGORY GROUPS")
    print("FILTERED DATA: Excluding Comorbidity Status and Symptom Presentation")
    print("="*80)
    
    # High Level (excluding symptom presentation)
    print("\nHIGH LEVEL BIAS (0.2-0.3):")
    df_high = pd.DataFrame(high_data)
    for cat in df_high['Category'].unique():
        cat_df = df_high[df_high['Category'] == cat]
        max_acc = cat_df['Accuracy'].max()
        min_acc = cat_df['Accuracy'].min()
        acc_diff = max_acc - min_acc
        
        best_group = cat_df.loc[cat_df['Accuracy'].idxmax(), 'Group']
        worst_group = cat_df.loc[cat_df['Accuracy'].idxmin(), 'Group']
        
        print(f"  {cat}:")
        print(f"    • Accuracy Difference: {acc_diff:.3f}")
        print(f"    • Best Group: {best_group} ({max_acc:.3f})")
        print(f"    • Worst Group: {worst_group} ({min_acc:.3f})")
    
    # Moderate Level
    print("\nMODERATE LEVEL BIAS (0.1-0.2):")
    df_moderate = pd.DataFrame(moderate_data)
    for cat in df_moderate['Category'].unique():
        cat_df = df_moderate[df_moderate['Category'] == cat]
        max_acc = cat_df['Accuracy'].max()
        min_acc = cat_df['Accuracy'].min()
        acc_diff = max_acc - min_acc
        
        best_group = cat_df.loc[cat_df['Accuracy'].idxmax(), 'Group']
        worst_group = cat_df.loc[cat_df['Accuracy'].idxmin(), 'Group']
        
        print(f"  {cat}:")
        print(f"    • Accuracy Difference: {acc_diff:.3f}")
        print(f"    • Best Group: {best_group} ({max_acc:.3f})")
        print(f"    • Worst Group: {worst_group} ({min_acc:.3f})")

def main():
    """Main function to generate filtered bias visualizations"""
    
    print("="*80)
    print("FILTERED BIAS TABLES VISUALIZATION GENERATOR")
    print("Excluding Comorbidity Status and Symptom Presentation")
    print("="*80)
    
    print("\n1. Creating detailed age group analysis...")
    create_age_analysis_visualization()
    
    print("2. Creating detailed drug use analysis...")
    create_drug_use_analysis_visualization()
    
    print("3. Creating demographic factors analysis...")
    create_demographic_analysis_visualization()
    
    print("4. Creating overall summary visualization...")
    create_overall_summary_visualization()
    
    print_filtered_accuracy_differences()
    
    print(f"\n" + "="*80)
    print("FILTERED VISUALIZATION GENERATION COMPLETE")
    print("Generated files:")
    print("• age_group_analysis_detailed.png")
    print("• drug_use_analysis_detailed.png")
    print("• demographic_analysis_detailed.png")
    print("• overall_bias_summary_filtered.png")
    print("="*80)
    print("\nRemaining Categories Analyzed:")
    print("• Age Groups (High Bias: 0.227 max)")
    print("• Drug Use Status (High Bias: 0.277 max)")
    print("• Gender (Moderate Bias: 0.196 max)")
    print("• Smoking Status (Moderate Bias: 0.148 max)")
    print("• Occupation Type (Moderate Bias: 0.137 max)")
    print("="*80)

if __name__ == "__main__":
    main()