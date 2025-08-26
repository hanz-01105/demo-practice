#!/usr/bin/env python3
"""
Improved Bias Tables Visualization Generator
Fixes visualization issues and improves readability

Creates enhanced visualizations for Tables 2, 3, and 4 from bias analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set global plotting parameters for publication quality
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.constrained_layout.use': True
})

def load_table_data():
    """Load all bias table data with improved labels"""
    
    # Table 2: Critical-Level Bias Categories (≥0.3) - FIXED LABEL
    critical_data = {
        'Category': ['Comorbidity Status', 'Comorbidity Status', 'Comorbidity Status'],
        'Group': ['Chronic Condition Present', 'Immunocompromised', 'Unknown'],  # FIXED: More professional term
        'N': [70, 4, 1],
        'Accuracy': [0.614, 0.750, 1.000],
        'CI_Lower': [0.503, 0.412, 0.564],
        'CI_Upper': [0.725, 1.088, 1.436],
        'Confidence': [82.8, 85.0, 85.0],
        'Parity_Gap': [0.000, 0.136, 0.386]
    }
    
    # Table 3: High-Level Bias Categories (0.2-0.3)
    high_data = {
        'Category': ['Age', 'Age', 'Age', 'Age', 'Age', 'Age', 'Age',
                    'Symptom Presentation', 'Symptom Presentation', 'Symptom Presentation', 'Symptom Presentation', 'Symptom Presentation',
                    'Drug Use', 'Drug Use', 'Drug Use'],
        'Group': ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60+',
                 'Atypical/Vague', 'Classic Textbook', 'Multi-System Complex', 'Single Symptom', 'Unknown',
                 'Drug User', 'Non-drug User', 'Unknown'],
        'N': [36, 21, 38, 29, 21, 33, 36,
              73, 23, 66, 47, 5,
              7, 38, 169],
        'Accuracy': [0.639, 0.524, 0.500, 0.552, 0.619, 0.727, 0.611,
                    0.521, 0.522, 0.591, 0.745, 0.800,
                    0.714, 0.368, 0.645],
        'CI_Lower': [0.489, 0.328, 0.348, 0.382, 0.426, 0.579, 0.459,
                    0.409, 0.333, 0.475, 0.622, 0.490,
                    0.428, 0.221, 0.574],
        'CI_Upper': [0.789, 0.720, 0.652, 0.722, 0.812, 0.875, 0.763,
                    0.633, 0.711, 0.707, 0.868, 1.110,
                    1.000, 0.515, 0.716],
        'Confidence': [81.7, 82.1, 77.2, 82.2, 82.1, 84.1, 81.0,
                      81.4, 80.0, 80.8, 82.0, 85.0,
                      80.7, 77.2, 82.2],
        'Parity_Gap': [0.139, 0.024, 0.000, 0.052, 0.119, 0.227, 0.111,
                      0.000, 0.001, 0.070, 0.224, 0.279,
                      0.069, -0.277, 0.000]
    }
    
    # Table 4: Moderate-Level Bias Categories (0.1-0.2)
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
    
    return critical_data, high_data, moderate_data

def create_table_2_visualization():
    """Create improved visualization for Table 2: Critical-Level Bias Categories"""
    
    critical_data, _, _ = load_table_data()
    df = pd.DataFrame(critical_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Table 2: Critical-Level Bias Categories (≥0.3)\nComorbidity Status Performance Analysis', 
                fontsize=16, fontweight='bold')
    
    # Colors for comorbidity groups
    colors = ['#E74C3C', '#C0392B', '#A93226']
    
    # 1. Accuracy with Confidence Intervals - IMPROVED
    ax1 = axes[0, 0]
    
    # Calculate error bars
    error_lower = df['Accuracy'] - df['CI_Lower']
    error_upper = df['CI_Upper'] - df['Accuracy']
    
    bars = ax1.bar(df['Group'], df['Accuracy'], color=colors, alpha=0.8, capsize=5)
    ax1.errorbar(df['Group'], df['Accuracy'], yerr=[error_lower, error_upper], 
                fmt='none', color='black', capsize=5, capthick=2)
    
    # Add value labels - IMPROVED positioning
    for i, (bar, acc, n) in enumerate(zip(bars, df['Accuracy'], df['N'])):
        height = bar.get_height()
        ci_upper = df['CI_Upper'].iloc[i] - acc
        y_pos = height + ci_upper + 0.08  # More space above error bars
        ax1.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{acc:.3f}\n(n={n})', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax1.set_ylabel('Top-1 Accuracy')
    ax1.set_title('Accuracy with 95% Confidence Intervals', fontweight='bold')
    ax1.set_xticklabels(df['Group'], rotation=45, ha='right')
    ax1.set_ylim(0, 1.8)  # More space for Unknown group CI
    
    # 2. Sample Size Distribution
    ax2 = axes[0, 1]
    
    bars = ax2.bar(df['Group'], df['N'], color=colors, alpha=0.8)
    
    for bar, n in zip(bars, df['N']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{n}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax2.set_ylabel('Sample Size')
    ax2.set_title('Sample Distribution', fontweight='bold')
    ax2.set_xticklabels(df['Group'], rotation=45, ha='right')
    
    # 3. Parity Gap Analysis
    ax3 = axes[1, 0]
    
    # Color bars by parity gap direction
    bar_colors = ['gray' if gap == 0 else '#E74C3C' if gap > 0 else '#3498DB' for gap in df['Parity_Gap']]
    bars = ax3.bar(df['Group'], np.abs(df['Parity_Gap']), color=bar_colors, alpha=0.8)
    
    # Add direction indicators
    for i, (bar, gap) in enumerate(zip(bars, df['Parity_Gap'])):
        height = bar.get_height()
        direction = '→' if gap == 0 else '↑' if gap > 0 else '↓'
        color = 'black' if gap == 0 else 'green' if gap > 0 else 'red'
        
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{abs(gap):.3f} {direction}', ha='center', va='bottom', 
                fontweight='bold', color=color, fontsize=10)
    
    # Add threshold lines
    ax3.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Critical (0.3)')
    ax3.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='High (0.2)')
    
    ax3.set_ylabel('Absolute Parity Gap')
    ax3.set_title('Bias Magnitude Analysis', fontweight='bold')
    ax3.set_xticklabels(df['Group'], rotation=45, ha='right')
    ax3.legend()
    
    # 4. Summary Statistics Table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary data
    table_data = []
    for i, row in df.iterrows():
        ci_width = row['CI_Upper'] - row['CI_Lower']
        # Better truncation for group names
        group_name = row['Group'][:12] + '...' if len(row['Group']) > 12 else row['Group']
        table_data.append([
            group_name,
            f"{row['N']}",
            f"{row['Accuracy']:.3f}",
            f"±{ci_width/2:.3f}",
            f"{row['Confidence']:.1f}%",
            f"{row['Parity_Gap']:+.3f}"
        ])
    
    headers = ['Group', 'N', 'Accuracy', 'CI ±', 'Confidence', 'Parity Gap']
    
    table = ax4.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.0)
    
    # Color code header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#ffcccc')  # Light red for critical
        table[(0, i)].set_text_props(weight='bold')
    
    # Highlight highest bias
    highest_bias_idx = df['Parity_Gap'].abs().idxmax() + 1
    for j in range(len(headers)):
        table[(highest_bias_idx, j)].set_facecolor('#ffe6e6')
    
    ax4.set_title('Critical Bias Summary Statistics', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('table_2_critical_bias_visualization_improved.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_table_3_visualization():
    """Create improved visualization for Table 3: High-Level Bias Categories"""
    
    _, high_data, _ = load_table_data()
    df = pd.DataFrame(high_data)
    
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))  # Slightly wider
    fig.suptitle('Table 3: High-Level Bias Categories (0.2-0.3)\nAge, Symptom Presentation, and Drug Use Analysis', 
                fontsize=16, fontweight='bold')
    
    # Color schemes for each category
    age_colors = ['#3498DB', '#2980B9', '#1F618D', '#1A5490', '#154360', '#0F2E4C', '#0A1B2E']
    symptom_colors = ['#F39C12', '#E67E22', '#D35400', '#BA4A00', '#A04000']
    drug_colors = ['#1ABC9C', '#16A085', '#138D75']
    
    # 1. Age Group Analysis - REDESIGNED AS LINE PLOT WITH CONFIDENCE BANDS
    ax1 = axes[0, 0]
    age_df = df[df['Category'] == 'Age']
    
    # Convert age groups to numeric midpoints for better visualization
    age_midpoints = [5, 15, 25, 35, 45, 55, 70]
    
    # Create line plot with markers
    line = ax1.plot(age_midpoints, age_df['Accuracy'], 'o-', color='#2980B9', 
                   linewidth=3, markersize=8, markerfacecolor='#3498DB', 
                   markeredgecolor='white', markeredgewidth=2, label='Accuracy')
    
    # Add confidence bands using CI data
    ci_lower = age_df['CI_Lower'].values
    ci_upper = age_df['CI_Upper'].values
    ax1.fill_between(age_midpoints, ci_lower, ci_upper, alpha=0.2, color='#3498DB', label='95% CI')
    
    # Add data points with sample size labels
    for i, (x, acc, n) in enumerate(zip(age_midpoints, age_df['Accuracy'], age_df['N'])):
        ax1.annotate(f'{acc:.3f}\n(n={n})', (x, acc), 
                    textcoords="offset points", xytext=(0,15), ha='center',
                    fontweight='bold', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Fit and show trend line
    z = np.polyfit(age_midpoints, age_df['Accuracy'], 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(5, 70, 100)
    ax1.plot(x_smooth, p(x_smooth), '--', color='red', alpha=0.7, linewidth=2, label='Trend')
    
    # Calculate and display correlation
    from scipy.stats import pearsonr
    corr, p_value = pearsonr(age_midpoints, age_df['Accuracy'])
    r_squared = corr**2
    
    ax1.text(0.05, 0.95, f'R² = {r_squared:.3f}\nCorr = {corr:.3f}', 
             transform=ax1.transAxes, va='top', ha='left',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7), 
             fontweight='bold', fontsize=9)
    
    ax1.set_xlabel('Age Group (Midpoint)')
    ax1.set_ylabel('Top-1 Accuracy')
    ax1.set_title('Age Group Performance with Trend Analysis', fontweight='bold')
    ax1.set_xticks(age_midpoints)
    ax1.set_xticklabels(age_df['Group'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0.3, 0.9)  # Focus on the data range
    
    # 2. Symptom Presentation Analysis - IMPROVED LABELS
    ax2 = axes[0, 1]
    symptom_df = df[df['Category'] == 'Symptom Presentation']
    
    x_positions = np.arange(len(symptom_df))
    bars = ax2.bar(x_positions, symptom_df['Accuracy'], color=symptom_colors, alpha=0.8)
    
    for i, (bar, acc, n) in enumerate(zip(bars, symptom_df['Accuracy'], symptom_df['N'])):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                f'{acc:.3f}\n(n={n})', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    ax2.set_ylabel('Top-1 Accuracy')
    ax2.set_title('Symptom Presentation Performance', fontweight='bold')
    ax2.set_xticks(x_positions)
    # FIXED: Full symptom presentation labels with proper rotation
    ax2.set_xticklabels(symptom_df['Group'], rotation=45, ha='right', fontsize=8)
    ax2.set_ylim(0, 1.0)
    
    # 3. Drug Use Analysis
    ax3 = axes[0, 2]
    drug_df = df[df['Category'] == 'Drug Use']
    
    bars = ax3.bar(drug_df['Group'], drug_df['Accuracy'], color=drug_colors, alpha=0.8)
    
    for bar, acc, n in zip(bars, drug_df['Accuracy'], drug_df['N']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                f'{acc:.3f}\n(n={n})', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax3.set_ylabel('Top-1 Accuracy')
    ax3.set_title('Drug Use Performance', fontweight='bold')
    ax3.set_xticklabels(drug_df['Group'], rotation=45, ha='right')
    ax3.set_ylim(0, 1.0)
    
    # 4. Combined Bias Magnitude Analysis - IMPROVED THRESHOLD POSITIONING
    ax4 = axes[1, 0]
    
    categories = ['Age', 'Symptom Presentation', 'Drug Use']
    max_biases = []
    category_colors = ['#3498DB', '#F39C12', '#1ABC9C']
    
    for cat in categories:
        cat_df = df[df['Category'] == cat]
        max_bias = cat_df['Parity_Gap'].abs().max()
        max_biases.append(max_bias)
    
    bars = ax4.bar(categories, max_biases, color=category_colors, alpha=0.8)
    
    for bar, bias in zip(bars, max_biases):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{bias:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # FIXED: Better positioned threshold lines
    ax4.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='High (0.2)')
    ax4.axhline(y=0.275, color='red', linestyle='--', alpha=0.7, label='Near Critical (0.275)')
    ax4.axhline(y=0.3, color='darkred', linestyle='--', alpha=0.7, label='Critical (0.3)')
    
    ax4.set_ylabel('Maximum Bias Magnitude')
    ax4.set_title('Category Bias Comparison', fontweight='bold')
    ax4.set_xticklabels(categories, rotation=45, ha='right')
    ax4.legend(loc='upper right')
    ax4.set_ylim(0, 0.32)  # Better range to show threshold lines
    
    # 5. Sample Size vs Accuracy Correlation
    ax5 = axes[1, 1]
    
    for i, cat in enumerate(categories):
        cat_df = df[df['Category'] == cat]
        ax5.scatter(cat_df['N'], cat_df['Accuracy'], 
                   c=category_colors[i], label=cat, alpha=0.7, s=60)
    
    ax5.set_xlabel('Sample Size')
    ax5.set_ylabel('Top-1 Accuracy')
    ax5.set_title('Sample Size vs Accuracy', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Performance Range Analysis
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
                f'{range_val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax6.set_ylabel('Performance Range (Max - Min)')
    ax6.set_title('Performance Disparity', fontweight='bold')
    ax6.set_xticklabels(categories, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('table_3_high_bias_visualization_improved.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_table_4_visualization():
    """Create improved visualization for Table 4: Moderate-Level Bias Categories"""
    
    _, _, moderate_data = load_table_data()
    df = pd.DataFrame(moderate_data)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Table 4: Moderate-Level Bias Categories (0.1-0.2)\nGender, Smoking, and Occupation Analysis', 
                fontsize=16, fontweight='bold')
    
    # Color schemes for each category
    gender_colors = ['#FF6B9D', '#4ECDC4', '#45B7D1']
    smoking_colors = ['#2ECC71', '#E74C3C', '#95A5A6']
    occupation_colors = ['#F39C12', '#E67E22', '#D35400', '#27AE60', '#8E44AD', '#34495E']
    
    # 1. Gender Analysis - FIXED TEXT POSITIONING
    ax1 = axes[0, 0]
    gender_df = df[df['Category'] == 'Gender']
    
    bars = ax1.bar(gender_df['Group'], gender_df['Accuracy'], color=gender_colors, alpha=0.8)
    
    # Add confidence intervals
    error_lower = gender_df['Accuracy'] - gender_df['CI_Lower']
    error_upper = gender_df['CI_Upper'] - gender_df['Accuracy']
    ax1.errorbar(gender_df['Group'], gender_df['Accuracy'], 
                yerr=[error_lower, error_upper], fmt='none', color='black', capsize=5)
    
    # FIXED: Better positioning of text to avoid overlap with error bars
    for i, (bar, acc, n) in enumerate(zip(bars, gender_df['Accuracy'], gender_df['N'])):
        height = bar.get_height()
        ci_upper = gender_df['CI_Upper'].iloc[i] - acc
        y_pos = height + ci_upper + 0.08  # More clearance above error bars
        ax1.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{acc:.3f}\n(n={n})', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax1.set_ylabel('Top-1 Accuracy')
    ax1.set_title('Gender Performance', fontweight='bold')
    ax1.set_ylim(0, 1.4)  # More space for error bars and labels
    
    # Calculate and display accuracy difference
    gender_diff = gender_df['Accuracy'].max() - gender_df['Accuracy'].min()
    ax1.text(0.02, 0.95, f'Accuracy Difference: {gender_diff:.3f}', 
             transform=ax1.transAxes, va='top', ha='left',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7), fontweight='bold')
    
    # 2. Smoking Analysis
    ax2 = axes[0, 1]
    smoking_df = df[df['Category'] == 'Smoking']
    
    bars = ax2.bar(smoking_df['Group'], smoking_df['Accuracy'], color=smoking_colors, alpha=0.8)
    
    for bar, acc, n in zip(bars, smoking_df['Accuracy'], smoking_df['N']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                f'{acc:.3f}\n(n={n})', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax2.set_ylabel('Top-1 Accuracy')
    ax2.set_title('Smoking Status Performance', fontweight='bold')
    ax2.set_xticklabels(smoking_df['Group'], rotation=45, ha='right')
    
    # Display accuracy difference
    smoking_diff = smoking_df['Accuracy'].max() - smoking_df['Accuracy'].min()
    ax2.text(0.02, 0.98, f'Accuracy Difference: {smoking_diff:.3f}', 
             transform=ax2.transAxes, va='top', ha='left',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7), fontweight='bold')
    
    # 3. Occupation Analysis
    ax3 = axes[0, 2]
    occupation_df = df[df['Category'] == 'Occupation Type']
    
    bars = ax3.bar(range(len(occupation_df)), occupation_df['Accuracy'], 
                   color=occupation_colors, alpha=0.8)
    
    for i, (bar, acc, n) in enumerate(zip(bars, occupation_df['Accuracy'], occupation_df['N'])):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                f'{acc:.3f}\n(n={n})', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    ax3.set_ylabel('Top-1 Accuracy')
    ax3.set_title('Occupation Type Performance', fontweight='bold')
    ax3.set_xticks(range(len(occupation_df)))
    ax3.set_xticklabels([g[:8] + '...' if len(g) > 8 else g for g in occupation_df['Group']], 
                       rotation=45, ha='right')
    
    # Display accuracy difference
    occupation_diff = occupation_df['Accuracy'].max() - occupation_df['Accuracy'].min()
    ax3.text(0.02, 0.98, f'Accuracy Difference: {occupation_diff:.3f}', 
             transform=ax3.transAxes, va='top', ha='left',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7), fontweight='bold')
    
    # 4. Parity Gap Comparison
    ax4 = axes[1, 0]
    
    # Show parity gaps for all groups
    all_gaps = df['Parity_Gap']
    all_groups = [f"{row['Category'][:6]}\n{row['Group'][:8]}" for _, row in df.iterrows()]
    
    # Color by gap direction
    bar_colors = ['#E74C3C' if gap > 0 else '#3498DB' if gap < 0 else '#95A5A6' for gap in all_gaps]
    bars = ax4.bar(range(len(all_gaps)), np.abs(all_gaps), color=bar_colors, alpha=0.8)
    
    ax4.set_ylabel('Absolute Parity Gap')
    ax4.set_title('Parity Gap Analysis', fontweight='bold')
    ax4.set_xticks(range(len(all_gaps)))
    ax4.set_xticklabels(all_groups, rotation=45, ha='right', fontsize=7)
    
    # Add threshold lines
    ax4.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Moderate (0.1)')
    ax4.axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='High (0.2)')
    ax4.legend()
    
    # 5. Category Bias Summary
    ax5 = axes[1, 1]
    
    categories = ['Gender', 'Smoking', 'Occupation Type']
    max_biases = []
    category_colors = ['#FF6B9D', '#2ECC71', '#F39C12']
    
    for cat in categories:
        cat_df = df[df['Category'] == cat]
        max_bias = cat_df['Parity_Gap'].abs().max()
        max_biases.append(max_bias)
    
    bars = ax5.bar(categories, max_biases, color=category_colors, alpha=0.8)
    
    for bar, bias in zip(bars, max_biases):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                f'{bias:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax5.set_ylabel('Maximum Bias Magnitude')
    ax5.set_title('Category Bias Comparison', fontweight='bold')
    ax5.set_xticklabels(categories, rotation=45, ha='right')
    
    # 6. Accuracy Differences Summary
    ax6 = axes[1, 2]
    
    # Calculate accuracy differences for each category
    acc_differences = []
    for cat in categories:
        cat_df = df[df['Category'] == cat]
        acc_diff = cat_df['Accuracy'].max() - cat_df['Accuracy'].min()
        acc_differences.append(acc_diff)
    
    bars = ax6.bar(categories, acc_differences, color=category_colors, alpha=0.8)
    
    for bar, diff in zip(bars, acc_differences):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{diff:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax6.set_ylabel('Accuracy Difference (Max - Min)')
    ax6.set_title('Performance Disparities', fontweight='bold')
    ax6.set_xticklabels(categories, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('table_4_moderate_bias_visualization_improved.png', dpi=300, bbox_inches='tight')
    plt.show()

    return fig

def create_combined_summary_visualization():
    """Create the recommended Figure 3 for publication - High bias categories performance"""
    
    _, high_data, _ = load_table_data()
    df = pd.DataFrame(high_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Figure 3: High-Level Bias Categories Performance Analysis\nAge Trends, Symptom Presentation, and Drug Use Patterns', 
                fontsize=16, fontweight='bold')
    
    # Color schemes
    age_color = '#2980B9'
    symptom_colors = ['#F39C12', '#E67E22', '#D35400', '#BA4A00', '#A04000']
    drug_colors = ['#1ABC9C', '#16A085', '#138D75']
    
    # 1. Age Group Trend Analysis (TOP LEFT)
    ax1 = axes[0, 0]
    age_df = df[df['Category'] == 'Age']
    age_midpoints = [5, 15, 25, 35, 45, 55, 70]
    
    # Main trend line with confidence bands
    line = ax1.plot(age_midpoints, age_df['Accuracy'], 'o-', color=age_color, 
                   linewidth=3, markersize=10, markerfacecolor='#3498DB', 
                   markeredgecolor='white', markeredgewidth=2, label='Observed Accuracy')
    
    # Confidence bands
    ci_lower = age_df['CI_Lower'].values
    ci_upper = age_df['CI_Upper'].values
    ax1.fill_between(age_midpoints, ci_lower, ci_upper, alpha=0.2, color='#3498DB', label='95% CI')
    
    # Polynomial trend fitting
    z = np.polyfit(age_midpoints, age_df['Accuracy'], 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(5, 70, 100)
    ax1.plot(x_smooth, p(x_smooth), '--', color='red', alpha=0.8, linewidth=2, label='Quadratic Trend')
    
    # Statistical annotations
    from scipy.stats import pearsonr
    corr, p_value = pearsonr(age_midpoints, age_df['Accuracy'])
    r_squared = corr**2
    slope = (age_df['Accuracy'].iloc[-1] - age_df['Accuracy'].iloc[0]) / (age_midpoints[-1] - age_midpoints[0])
    
    ax1.text(0.05, 0.95, f'R² = {r_squared:.3f}\nSlope = {slope:.4f}\np = {p_value:.3f}', 
             transform=ax1.transAxes, va='top', ha='left',
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8), 
             fontweight='bold', fontsize=10)
    
    # Sample size annotations
    for x, acc, n in zip(age_midpoints, age_df['Accuracy'], age_df['N']):
        ax1.annotate(f'n={n}', (x, acc), textcoords="offset points", xytext=(0,-20), 
                    ha='center', fontsize=8, alpha=0.7)
    
    ax1.set_xlabel('Age Group (Years)')
    ax1.set_ylabel('Top-1 Accuracy')
    ax1.set_title('A. Age Group Performance Trend', fontweight='bold', loc='left')
    ax1.set_xticks(age_midpoints)
    ax1.set_xticklabels(age_df['Group'])
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')
    ax1.set_ylim(0.4, 0.8)
    
    # 2. Symptom Presentation Comparison (TOP RIGHT)
    ax2 = axes[0, 1]
    symptom_df = df[df['Category'] == 'Symptom Presentation']
    
    # Horizontal bar chart for better label readability
    y_positions = np.arange(len(symptom_df))
    bars = ax2.barh(y_positions, symptom_df['Accuracy'], color=symptom_colors, alpha=0.8)
    
    # Add confidence intervals as error bars
    error_lower = symptom_df['Accuracy'] - symptom_df['CI_Lower']
    error_upper = symptom_df['CI_Upper'] - symptom_df['Accuracy']
    ax2.errorbar(symptom_df['Accuracy'], y_positions, xerr=[error_lower, error_upper], 
                fmt='none', color='black', capsize=4)
    
    # Add value and sample size labels
    for i, (acc, n) in enumerate(zip(symptom_df['Accuracy'], symptom_df['N'])):
        ax2.text(acc + 0.02, i, f'{acc:.3f} (n={n})', va='center', fontweight='bold', fontsize=9)
    
    ax2.set_xlabel('Top-1 Accuracy')
    ax2.set_title('B. Symptom Presentation Types', fontweight='bold', loc='left')
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(symptom_df['Group'])
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim(0.3, 1.0)
    
    # 3. Drug Use Impact Analysis (BOTTOM LEFT)
    ax3 = axes[1, 0]
    drug_df = df[df['Category'] == 'Drug Use']
    
    bars = ax3.bar(drug_df['Group'], drug_df['Accuracy'], color=drug_colors, alpha=0.8)
    
    # Add confidence intervals
    error_lower = drug_df['Accuracy'] - drug_df['CI_Lower']
    error_upper = drug_df['CI_Upper'] - drug_df['Accuracy']
    ax3.errorbar(drug_df['Group'], drug_df['Accuracy'], yerr=[error_lower, error_upper], 
                fmt='none', color='black', capsize=5)
    
    # Value labels with effect size calculation
    baseline_acc = drug_df[drug_df['Group'] == 'Unknown']['Accuracy'].iloc[0]
    for bar, acc, n, group in zip(bars, drug_df['Accuracy'], drug_df['N'], drug_df['Group']):
        height = bar.get_height()
        effect_size = acc - baseline_acc
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{acc:.3f}\n(n={n})\nΔ={effect_size:+.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    
    ax3.set_ylabel('Top-1 Accuracy')
    ax3.set_title('C. Drug Use Status Impact', fontweight='bold', loc='left')
    ax3.set_xticklabels(drug_df['Group'], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.2, 0.9)
    
    # 4. Bias Magnitude Summary (BOTTOM RIGHT)
    ax4 = axes[1, 1]
    
    categories = ['Age Groups', 'Symptom Types', 'Drug Use']
    max_biases = []
    perf_ranges = []
    sample_ranges = []
    
    for cat_name, cat_key in zip(categories, ['Age', 'Symptom Presentation', 'Drug Use']):
        cat_df = df[df['Category'] == cat_key]
        max_bias = cat_df['Parity_Gap'].abs().max()
        perf_range = cat_df['Accuracy'].max() - cat_df['Accuracy'].min()
        sample_range = cat_df['N'].max() - cat_df['N'].min()
        
        max_biases.append(max_bias)
        perf_ranges.append(perf_range)
        sample_ranges.append(sample_range)
    
    x = np.arange(len(categories))
    width = 0.25
    
    bars1 = ax4.bar(x - width, max_biases, width, label='Max Bias Magnitude', color='#E74C3C', alpha=0.8)
    bars2 = ax4.bar(x, perf_ranges, width, label='Performance Range', color='#F39C12', alpha=0.8)
    
    # Add value labels
    for bars, values in [(bars1, max_biases), (bars2, perf_ranges)]:
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Add threshold lines
    ax4.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='High Bias (0.2)')
    ax4.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Critical Bias (0.3)')
    
    ax4.set_ylabel('Magnitude')
    ax4.set_title('D. Bias and Performance Summary', fontweight='bold', loc='left')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure_3_high_bias_categories_recommended.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig
    """Create an improved combined summary visualization of all three tables"""
    
    critical_data, high_data, moderate_data = load_table_data()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bias Severity Analysis: Complete Summary\nTables 2, 3, and 4 Combined Overview', 
                fontsize=16, fontweight='bold')
    
    # Calculate max bias for each category
    categories = ['Comorbidity', 'Age', 'Symptom', 'Drug Use', 'Gender', 'Smoking', 'Occupation']
    max_biases = [0.386, 0.227, 0.279, 0.277, 0.196, 0.148, 0.137]
    severity_colors = ['#8B0000', '#E67E22', '#E67E22', '#E67E22', '#F39C12', '#F39C12', '#F39C12']
    
    # 1. Bias Magnitude Comparison - IMPROVED
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(categories)), max_biases, color=severity_colors, alpha=0.8)
    
    for i, (bar, bias) in enumerate(zip(bars, max_biases)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{bias:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # IMPROVED threshold lines positioning
    ax1.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Moderate (0.1)', linewidth=2)
    ax1.axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='High (0.2)', linewidth=2)
    ax1.axhline(y=0.3, color='darkred', linestyle='--', alpha=0.7, label='Critical (0.3)', linewidth=2)
    
    ax1.set_ylabel('Maximum Bias Magnitude')
    ax1.set_title('Bias Magnitude by Category', fontweight='bold')
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 0.42)
    
    # 2. Sample Size Distribution
    ax2 = axes[0, 1]
    
    # Calculate total sample sizes per category
    critical_df = pd.DataFrame(critical_data)
    high_df = pd.DataFrame(high_data)
    moderate_df = pd.DataFrame(moderate_data)
    
    sample_sizes = []
    sample_sizes.append(critical_df['N'].sum())  # Comorbidity
    sample_sizes.append(high_df[high_df['Category'] == 'Age']['N'].sum())  # Age
    sample_sizes.append(high_df[high_df['Category'] == 'Symptom Presentation']['N'].sum())  # Symptom
    sample_sizes.append(high_df[high_df['Category'] == 'Drug Use']['N'].sum())  # Drug Use
    sample_sizes.append(moderate_df[moderate_df['Category'] == 'Gender']['N'].sum())  # Gender
    sample_sizes.append(moderate_df[moderate_df['Category'] == 'Smoking']['N'].sum())  # Smoking
    sample_sizes.append(moderate_df[moderate_df['Category'] == 'Occupation Type']['N'].sum())  # Occupation
    
    bars = ax2.bar(range(len(categories)), sample_sizes, color=severity_colors, alpha=0.8)
    
    for bar, size in zip(bars, sample_sizes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{size}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax2.set_ylabel('Total Sample Size')
    ax2.set_title('Sample Distribution by Category', fontweight='bold')
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    
    # 3. Bias Severity Classification
    ax3 = axes[1, 0]
    
    severity_levels = ['Critical (≥0.3)', 'High (0.2-0.3)', 'Moderate (0.1-0.2)']
    severity_counts = [1, 3, 3]  # Number of categories in each level
    severity_level_colors = ['#8B0000', '#E67E22', '#F39C12']
    
    bars = ax3.bar(severity_levels, severity_counts, color=severity_level_colors, alpha=0.8)
    
    for bar, count in zip(bars, severity_counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax3.set_ylabel('Number of Categories')
    ax3.set_title('Bias Severity Distribution', fontweight='bold')
    ax3.set_xticklabels(severity_levels, rotation=45, ha='right')
    
    # 4. Key Insights Text Box
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    insights_text = """
    KEY FINDINGS:
    
    • CRITICAL BIAS: Comorbidity status shows
      the highest bias (0.386), particularly for 
      unknown status patients
      
    • HIGH BIAS: Age groups, symptom 
      presentation, and drug use all exceed 
      0.2 bias threshold
      
    • MODERATE BIAS: Gender, smoking, and 
      occupation show concerning but 
      manageable bias levels
      
    • SAMPLE IMBALANCE: Several categories 
      have very small sample sizes (n<10) 
      which may affect reliability
      
    • PATTERN: Unknown/missing status 
      consistently shows higher bias across 
      multiple categories
    """
    
    ax4.text(0.05, 0.95, insights_text, transform=ax4.transAxes, 
             fontsize=10, va='top', ha='left',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
             fontweight='normal')
    
    ax4.set_title('Summary Insights', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('combined_bias_tables_summary_improved.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def print_accuracy_differences():
    """Print accuracy differences for all bias categories"""
    
    critical_data, high_data, moderate_data = load_table_data()
    
    print("\n" + "="*80)
    print("ACCURACY DIFFERENCES BETWEEN SUBCATEGORY GROUPS")
    print("All Tables: Critical, High, and Moderate-Level Bias Categories")
    print("="*80)
    
    # Critical Level
    print("\nCRITICAL LEVEL BIAS (≥0.3):")
    df_critical = pd.DataFrame(critical_data)
    for cat in df_critical['Category'].unique():
        cat_df = df_critical[df_critical['Category'] == cat]
        max_acc = cat_df['Accuracy'].max()
        min_acc = cat_df['Accuracy'].min()
        acc_diff = max_acc - min_acc
        
        best_group = cat_df.loc[cat_df['Accuracy'].idxmax(), 'Group']
        worst_group = cat_df.loc[cat_df['Accuracy'].idxmin(), 'Group']
        
        print(f"  {cat}:")
        print(f"    • Accuracy Difference: {acc_diff:.3f}")
        print(f"    • Best Group: {best_group} ({max_acc:.3f})")
        print(f"    • Worst Group: {worst_group} ({min_acc:.3f})")
    
    # High Level
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
    """Main function to generate all improved bias table visualizations"""
    
    print("="*80)
    print("IMPROVED BIAS TABLES VISUALIZATION GENERATOR")
    print("Converting Tables 2, 3, and 4 to Enhanced Publication-Quality Images")
    print("="*80)
    
    print("\n1. Creating improved Table 2 visualization (Critical-Level Bias)...")
    create_table_2_visualization()
    
    print("2. Creating improved Table 3 visualization (High-Level Bias)...")
    create_table_3_visualization()
    
    print("3. Creating improved Table 4 visualization (Moderate-Level Bias)...")
    create_table_4_visualization()
    
    print_accuracy_differences()
    
    print(f"\n" + "="*80)
    print("IMPROVED VISUALIZATION GENERATION COMPLETE")
    print("Generated improved files:")
    print("• table_2_critical_bias_visualization_improved.png")
    print("• table_3_high_bias_visualization_improved.png")  
    print("• table_4_moderate_bias_visualization_improved.png")
    print("="*80)
    print("\nKey Improvements Made:")
    print("• Fixed 'Immunosuppressed/Special' → 'Immunocompromised'")
    print("• Improved age group spacing and visibility")
    print("• Full symptom presentation labels on x-axis")
    print("• Better positioned threshold lines in bias comparison")
    print("• Fixed overlapping text in gender performance chart")
    print("• Enhanced overall readability and professional appearance")
    print("="*80)

if __name__ == "__main__":
    main()
