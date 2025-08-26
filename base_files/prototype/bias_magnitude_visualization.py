#!/usr/bin/env python3
"""
Bias Magnitude by Category Visualization - Custom Color Palette

This script creates publication-quality visualizations of bias magnitude across
all 8 demographic categories using a custom color scheme.

Usage: python bias_magnitude_visualization.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Custom color palette
CUSTOM_COLORS = ['#8f0707', '#be6c65', '#d7c2c1', '#de99a1', '#de6e8c']

# Set publication-quality parameters
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3
})

def load_all_demographic_data():
    """Load all demographic data from the provided CSV content"""
    
    # All 8 demographic categories with their data
    demographic_data = {
        'Age Group': {
            'groups': ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60+'],
            'max_bias_magnitude': 0.227,  # From Parity Gap analysis
            'worst_group_disadvantage': 0.227,
            'performance_range': 0.727 - 0.500,  # 0.227
            'groups_count': 7,
            'total_sample': 214
        },
        'Gender': {
            'groups': ['Female', 'Male', 'Other'],
            'max_bias_magnitude': 0.357,  # Highest bias magnitude
            'worst_group_disadvantage': 0.357,
            'performance_range': 0.857 - 0.500,  # 0.357
            'groups_count': 3,
            'total_sample': 214
        },
        'Smoking Status': {
            'groups': ['Non-smoker', 'Smoker', 'Unknown'],
            'max_bias_magnitude': 0.166,
            'worst_group_disadvantage': 0.166,
            'performance_range': 0.733 - 0.567,  # 0.166
            'groups_count': 3,
            'total_sample': 214
        },
        'Alcohol Use': {
            'groups': ['Drinker', 'Non-drinker', 'Unknown'],
            'max_bias_magnitude': 0.144,  # From worst-group disadvantage
            'worst_group_disadvantage': 0.144,
            'performance_range': 0.656 - 0.512,  # 0.144
            'groups_count': 3,
            'total_sample': 214
        },
        'Drug Use': {
            'groups': ['Drug User', 'Non-drug User', 'Unknown'],
            'max_bias_magnitude': 0.346,  # High bias magnitude
            'worst_group_disadvantage': 0.346,
            'performance_range': 0.714 - 0.368,  # 0.346
            'groups_count': 3,
            'total_sample': 214
        },
        'Occupation Type': {
            'groups': ['Knowledge Worker', 'Manual Labor', 'Retired', 'Student', 'Unemployed', 'Unknown'],
            'max_bias_magnitude': 0.227,
            'worst_group_disadvantage': 0.227,
            'performance_range': 0.727 - 0.500,  # 0.227
            'groups_count': 6,
            'total_sample': 214
        },
        'Comorbidity Status': {
            'groups': ['Chronic Condition Present', 'Immunosuppressed/Special Treatment', 'Unknown'],
            'max_bias_magnitude': 0.386,  # Very high bias
            'worst_group_disadvantage': 0.386,
            'performance_range': 1.000 - 0.614,  # 0.386
            'groups_count': 3,
            'total_sample': 214
        },
        'Symptom Presentation': {
            'groups': ['Atypical/Vague', 'Classic Textbook', 'Multi-System', 'Single Symptom', 'Unknown'],
            'max_bias_magnitude': 0.279,
            'worst_group_disadvantage': 0.279,
            'performance_range': 0.800 - 0.521,  # 0.279
            'groups_count': 5,
            'total_sample': 214
        }
    }
    
    return demographic_data

def get_custom_color_by_severity(magnitude):
    """Assign custom colors based on bias severity levels"""
    if magnitude >= 0.35:
        return CUSTOM_COLORS[0]  # #8f0707 - Darkest red for highest bias
    elif magnitude >= 0.3:
        return CUSTOM_COLORS[1]  # #be6c65 - Dark red
    elif magnitude >= 0.2:
        return CUSTOM_COLORS[2]  # #d7c2c1 - Medium red
    elif magnitude >= 0.15:
        return CUSTOM_COLORS[3]  # #de99a1 - Light red
    else:
        return CUSTOM_COLORS[4]  # #de6e8c - Pink for lowest bias

def create_custom_color_mapping(values):
    """Create color mapping for a list of values using the custom palette"""
    # Sort values to assign colors from darkest to lightest
    sorted_indices = np.argsort(values)[::-1]  # Descending order
    colors = [''] * len(values)
    
    # Distribute custom colors evenly across the range
    for i, idx in enumerate(sorted_indices):
        color_idx = min(i * len(CUSTOM_COLORS) // len(values), len(CUSTOM_COLORS) - 1)
        colors[idx] = CUSTOM_COLORS[color_idx]
    
    return colors

def create_bias_magnitude_main_chart():
    """Create the main bias magnitude comparison chart with custom colors"""
    
    data = load_all_demographic_data()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bias Magnitude Analysis Across All Demographic Categories', 
                fontsize=16, fontweight='bold')
    
    # 1. Main bias magnitude comparison (Top Left)
    ax1 = axes[0, 0]
    
    categories = list(data.keys())
    bias_magnitudes = [data[cat]['max_bias_magnitude'] for cat in categories]
    
    # Use custom color mapping
    colors = [get_custom_color_by_severity(mag) for mag in bias_magnitudes]
    
    bars = ax1.bar(range(len(categories)), bias_magnitudes, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for i, (bar, magnitude) in enumerate(zip(bars, bias_magnitudes)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{magnitude:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add severity threshold lines with custom colors
    ax1.axhline(y=0.15, color=CUSTOM_COLORS[4], linestyle='--', alpha=0.7, label='Low (0.15)')
    ax1.axhline(y=0.2, color=CUSTOM_COLORS[3], linestyle='--', alpha=0.7, label='Moderate (0.2)')
    ax1.axhline(y=0.3, color=CUSTOM_COLORS[1], linestyle='--', alpha=0.7, label='High (0.3)')
    ax1.axhline(y=0.35, color=CUSTOM_COLORS[0], linestyle='--', alpha=0.7, label='Critical (0.35)')
    
    ax1.set_ylabel('Maximum Bias Magnitude')
    ax1.set_title('Bias Magnitude by Demographic Category', fontweight='bold')
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend(loc='upper right', bbox_to_anchor=(0.50, 0.95), framealpha=0.9, fancybox=True, shadow=True)
    ax1.set_ylim(0, max(bias_magnitudes) * 1.15)
    
    # 2. Performance range comparison (Top Right)
    ax2 = axes[0, 1]
    
    performance_ranges = [data[cat]['performance_range'] for cat in categories]
    range_colors = create_custom_color_mapping(performance_ranges)
    
    bars2 = ax2.bar(range(len(categories)), performance_ranges, color=range_colors, alpha=0.8, 
                    edgecolor='black', linewidth=1)
    
    for i, (bar, range_val) in enumerate(zip(bars2, performance_ranges)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{range_val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax2.set_ylabel('Performance Range (Max - Min)')
    ax2.set_title('Performance Disparity Range', fontweight='bold')
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    
    # 3. Bias severity distribution (Bottom Left)
    ax3 = axes[1, 0]
    
    # Count categories by severity level
    severity_counts = {
        'Low (<0.15)': sum(1 for mag in bias_magnitudes if mag < 0.15),
        'Moderate (0.15-0.2)': sum(1 for mag in bias_magnitudes if 0.15 <= mag < 0.2),
        'High (0.2-0.3)': sum(1 for mag in bias_magnitudes if 0.2 <= mag < 0.3),
        'Very High (0.3-0.35)': sum(1 for mag in bias_magnitudes if 0.3 <= mag < 0.35),
        'Critical (≥0.35)': sum(1 for mag in bias_magnitudes if mag >= 0.35)
    }
    
    severity_labels = list(severity_counts.keys())
    severity_values = list(severity_counts.values())
    
    bars3 = ax3.bar(severity_labels, severity_values, color=CUSTOM_COLORS, alpha=0.8, 
                    edgecolor='black', linewidth=1)
    
    for bar, count in zip(bars3, severity_values):
        if count > 0:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax3.set_ylabel('Number of Categories')
    ax3.set_title('Bias Severity Distribution', fontweight='bold')
    ax3.set_xticklabels(severity_labels, rotation=45, ha='right')
    
    # 4. Category complexity vs bias (Bottom Right)
    ax4 = axes[1, 1]
    
    group_counts = [data[cat]['groups_count'] for cat in categories]
    
    # Scatter plot with custom colors
    scatter = ax4.scatter(group_counts, bias_magnitudes, s=120, c=colors, alpha=0.8, 
                         edgecolors='black', linewidth=1)
    
    # Add category labels
    for i, cat in enumerate(categories):
        ax4.annotate(cat.replace(' ', '\n'), (group_counts[i], bias_magnitudes[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax4.set_xlabel('Number of Groups in Category')
    ax4.set_ylabel('Maximum Bias Magnitude')
    ax4.set_title('Category Complexity vs Bias Magnitude', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('bias_magnitude_by_category_custom.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_detailed_bias_heatmap():
    """Create a detailed heatmap with custom color scheme"""
    
    data = load_all_demographic_data()
    
    # Create matrix for heatmap
    categories = list(data.keys())
    metrics = ['Max Bias', 'Performance Range', 'Groups Count']
    
    # Normalize values for better comparison
    bias_values = [data[cat]['max_bias_magnitude'] for cat in categories]
    range_values = [data[cat]['performance_range'] for cat in categories]
    group_values = [data[cat]['groups_count'] for cat in categories]
    
    # Normalize to 0-1 scale for heatmap
    bias_norm = [(val - min(bias_values)) / (max(bias_values) - min(bias_values)) for val in bias_values]
    range_norm = [(val - min(range_values)) / (max(range_values) - min(range_values)) for val in range_values]
    group_norm = [(val - min(group_values)) / (max(group_values) - min(group_values)) for val in group_values]
    
    heatmap_data = np.array([bias_norm, range_norm, group_norm]).T
    
    # Create custom colormap from our palette
    from matplotlib.colors import LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap.from_list('custom', CUSTOM_COLORS)
    
    # Create figure with explicit layout
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    # Create heatmap with custom colormap
    im = ax.imshow(heatmap_data, cmap=custom_cmap, aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    
    # Add text annotations with actual values
    for i in range(len(categories)):
        ax.text(0, i, f'{bias_values[i]:.3f}', ha='center', va='center', fontweight='bold')
        ax.text(1, i, f'{range_values[i]:.3f}', ha='center', va='center', fontweight='bold')
        ax.text(2, i, f'{group_values[i]}', ha='center', va='center', fontweight='bold')
    
    # Colorbar with explicit positioning
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Normalized Intensity', rotation=270, labelpad=15)
    
    ax.set_title('Bias Characteristics Heatmap\n(Actual values shown)', fontweight='bold', fontsize=14)
    
    plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1)
    plt.savefig('bias_characteristics_heatmap_custom.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_bias_ranking_table():
    """Create a ranking table instead of a chart"""
    
    data = load_all_demographic_data()
    
    # Exclude specific categories
    excluded_categories = ['Comorbidity Status', 'Symptom Presentation']
    
    # Filter and sort categories by bias magnitude
    categories = [cat for cat in data.keys() if cat not in excluded_categories]
    bias_data = [(cat, data[cat]['max_bias_magnitude'], data[cat]['performance_range'], 
                  data[cat]['groups_count'], data[cat]['total_sample']) 
                 for cat in categories]
    bias_data.sort(key=lambda x: x[1], reverse=True)  # Sort by bias magnitude
    
    # Create ranking DataFrame
    ranking_data = []
    for i, (category, bias_mag, perf_range, groups, sample_size) in enumerate(bias_data):
        # Determine severity level
        if bias_mag >= 0.35:
            severity = 'Critical'
            color_code = CUSTOM_COLORS[0]
        elif bias_mag >= 0.3:
            severity = 'Very High'
            color_code = CUSTOM_COLORS[1]
        elif bias_mag >= 0.2:
            severity = 'High'
            color_code = CUSTOM_COLORS[2]
        elif bias_mag >= 0.15:
            severity = 'Moderate'
            color_code = CUSTOM_COLORS[3]
        else:
            severity = 'Low'
            color_code = CUSTOM_COLORS[4]
        
        ranking_data.append({
            'Rank': i + 1,
            'Category': category,
            'Bias Magnitude': f"{bias_mag:.3f}",
            'Severity Level': severity,
            'Performance Range': f"{perf_range:.3f}",
            'Groups Count': groups,
            'Sample Size': sample_size,
            'Color Code': color_code
        })
    
    df_ranking = pd.DataFrame(ranking_data)
    
    # Create a visual table using matplotlib
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data for matplotlib (without color code column for display)
    table_data = df_ranking.drop('Color Code', axis=1).values
    col_labels = df_ranking.drop('Color Code', axis=1).columns
    
    # Create the table
    table = ax.table(cellText=table_data,
                    colLabels=col_labels,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color code only the severity level column
    for i in range(len(ranking_data)):
        color = ranking_data[i]['Color Code']
        
        # Only color the Severity Level column (index 3)
        table[(i+1, 3)].set_facecolor(color)  # Severity Level column only
        
        # Make text white for darker colors in severity column
        if color in [CUSTOM_COLORS[0], CUSTOM_COLORS[1]]:
            table[(i+1, 3)].set_text_props(weight='bold', color='white')
        else:
            table[(i+1, 3)].set_text_props(weight='bold')
    
    # Style header row
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#333333')
        table[(0, j)].set_text_props(weight='bold', color='white')
        table[(0, j)].set_height(0.08)
    
    # Add title
    plt.title('Demographic Categories Ranked by Bias Magnitude', 
              fontweight='bold', fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.savefig('bias_magnitude_ranking_table.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Also save as CSV
    df_ranking.drop('Color Code', axis=1).to_csv('bias_magnitude_ranking_table.csv', index=False)
    
    # Print the table to console
    print("\n" + "="*100)
    print("BIAS MAGNITUDE RANKING TABLE")
    print("="*100)
    print(df_ranking.drop('Color Code', axis=1).to_string(index=False))
    print("="*100)
    print(f"Table saved as: bias_magnitude_ranking_table.png")
    print(f"Data saved as: bias_magnitude_ranking_table.csv")
    print(f"Note: Excluded categories: {', '.join(excluded_categories)}")
    
    return fig, df_ranking

def create_color_palette_reference():
    """Create a reference chart showing the custom color palette"""
    
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # Create color swatches
    for i, color in enumerate(CUSTOM_COLORS):
        ax.bar(i, 1, color=color, alpha=0.8, edgecolor='black', linewidth=1, width=0.8)
        ax.text(i, 0.5, color, ha='center', va='center', fontweight='bold', 
                color='white', fontsize=12)
        
        # Add severity labels
        severity_labels = ['Critical/Highest', 'Very High', 'High/Medium', 'Moderate', 'Low/Lowest']
        ax.text(i, -0.1, severity_labels[i], ha='center', va='top', fontweight='bold', 
                fontsize=10, rotation=45)
    
    ax.set_xlim(-0.5, len(CUSTOM_COLORS) - 0.5)
    ax.set_ylim(-0.3, 1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Custom Color Palette Reference\nBias Severity Mapping', 
                fontweight='bold', fontsize=14)
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.savefig('color_palette_reference.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def generate_summary_table():
    """Generate a summary table of all bias magnitudes"""
    
    data = load_all_demographic_data()
    
    # Create summary DataFrame
    summary_data = []
    for category, cat_data in data.items():
        magnitude = cat_data['max_bias_magnitude']
        if magnitude >= 0.35:
            severity = 'Critical'
        elif magnitude >= 0.3:
            severity = 'Very High'
        elif magnitude >= 0.2:
            severity = 'High'
        elif magnitude >= 0.15:
            severity = 'Moderate'
        else:
            severity = 'Low'
            
        summary_data.append({
            'Category': category,
            'Groups': cat_data['groups_count'],
            'Sample Size': cat_data['total_sample'],
            'Max Bias Magnitude': f"{cat_data['max_bias_magnitude']:.3f}",
            'Performance Range': f"{cat_data['performance_range']:.3f}",
            'Severity Level': severity,
            'Color Code': get_custom_color_by_severity(magnitude)
        })
    
    df = pd.DataFrame(summary_data)
    df = df.sort_values('Max Bias Magnitude', ascending=False)
    
    print("\n" + "="*90)
    print("BIAS MAGNITUDE SUMMARY TABLE - CUSTOM COLOR SCHEME")
    print("="*90)
    print(df.to_string(index=False))
    print("="*90)
    
    # Save to CSV
    df.to_csv('bias_magnitude_summary_custom.csv', index=False)
    print(f"Summary table saved to: bias_magnitude_summary_custom.csv")
    
    return df

def main():
    """Main function to generate all bias magnitude visualizations with custom colors"""
    
    print("="*80)
    print("BIAS MAGNITUDE VISUALIZATION GENERATOR - CUSTOM COLOR PALETTE")
    print("Colors: #8f0707, #be6c65, #d7c2c1, #de99a1, #de6e8c")
    print("Analyzing Emergent Biases in Multi-Agent Medical Systems")
    print("="*80)
    
    print("\n1. Creating color palette reference...")
    color_ref = create_color_palette_reference()
    
    print("2. Creating main bias magnitude comparison chart...")
    fig1 = create_bias_magnitude_main_chart()
    
    print("3. Creating detailed bias characteristics heatmap...")
    fig2 = create_detailed_bias_heatmap()
    
    print("4. Creating bias magnitude ranking table...")
    fig3, ranking_df = create_bias_ranking_table()
    
    print("5. Generating summary table...")
    summary_df = generate_summary_table()
    
    print(f"\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)
    
    # Find highest and lowest bias categories
    data = load_all_demographic_data()
    bias_values = [(cat, data[cat]['max_bias_magnitude']) for cat in data.keys()]
    bias_values.sort(key=lambda x: x[1], reverse=True)
    
    print(f"HIGHEST BIAS: {bias_values[0][0]} ({bias_values[0][1]:.3f}) - Color: {get_custom_color_by_severity(bias_values[0][1])}")
    print(f"LOWEST BIAS: {bias_values[-1][0]} ({bias_values[-1][1]:.3f}) - Color: {get_custom_color_by_severity(bias_values[-1][1])}")
    
    # Count severity levels
    critical_count = sum(1 for _, mag in bias_values if mag >= 0.35)
    very_high_count = sum(1 for _, mag in bias_values if 0.3 <= mag < 0.35)
    high_count = sum(1 for _, mag in bias_values if 0.2 <= mag < 0.3)
    moderate_count = sum(1 for _, mag in bias_values if 0.15 <= mag < 0.2)
    low_count = sum(1 for _, mag in bias_values if mag < 0.15)
    
    print(f"\nSEVERITY DISTRIBUTION (Custom Color Scheme):")
    print(f"Critical (≥0.35): {critical_count} categories - {CUSTOM_COLORS[0]}")
    print(f"Very High (0.3-0.35): {very_high_count} categories - {CUSTOM_COLORS[1]}")
    print(f"High (0.2-0.3): {high_count} categories - {CUSTOM_COLORS[2]}")
    print(f"Moderate (0.15-0.2): {moderate_count} categories - {CUSTOM_COLORS[3]}")
    print(f"Low (<0.15): {low_count} categories - {CUSTOM_COLORS[4]}")
    
    print(f"\nFILES GENERATED:")
    print(f"- color_palette_reference.png")
    print(f"- bias_magnitude_by_category_custom.png")
    print(f"- bias_characteristics_heatmap_custom.png") 
    print(f"- bias_magnitude_ranking_table.png")
    print(f"- bias_magnitude_ranking_table.csv")
    print(f"- bias_magnitude_summary_custom.csv")
    
    print("="*80)
    print("CUSTOM COLOR ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()