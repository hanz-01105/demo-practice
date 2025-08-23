#!/usr/bin/env python3
"""
Age-Based Performance Trend Analysis
Multi-Agent Medical Diagnostic Systems

Focused analysis on age-related performance patterns and trends
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Set global plotting parameters for publication quality
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.constrained_layout.use': True
})

def load_age_data():
    """Load age group performance data from CSV"""
    
    age_df = pd.read_csv('demographic_metrics/age_group_metrics.csv')
    
    # Map age groups to midpoint values for trend analysis
    age_midpoints = {
        '0-10': 5,
        '10-20': 15,
        '20-30': 25,
        '30-40': 35,
        '40-50': 45,
        '50-60': 55,
        '60+': 70
    }
    
    age_data = {
        'groups': age_df['Group'].tolist(),
        'counts': age_df['Count'].tolist(),
        'top1_acc': age_df['Top-1 Accuracy'].tolist(),
        'top3_acc': age_df['Top-3 Accuracy'].tolist(),
        'top5_acc': age_df['Top-5 Accuracy'].tolist(),
        'confidence': age_df['Avg Confidence (All Cases)'].tolist(),
        'parity_gap': age_df['Parity Gap (vs Baseline)'].tolist(),
        'midpoints': [age_midpoints[group] for group in age_df['Group'].tolist()]
    }
    
    return age_data

def calculate_wilson_confidence_intervals(accuracies, counts, z=1.96):
    """Calculate Wilson score confidence intervals"""
    conf_intervals = []
    for acc, n in zip(accuracies, counts):
        p = acc
        n_eff = n + z**2
        p_eff = (n * p + 0.5 * z**2) / n_eff
        ci_half = z * np.sqrt(p_eff * (1 - p_eff) / n_eff)
        conf_intervals.append(ci_half)
    return conf_intervals

def create_age_trend_analysis():
    """Create comprehensive age trend analysis"""
    
    age_data = load_age_data()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Age-Based Performance Patterns in Medical Diagnosis\nMulti-Agent System Analysis', 
                fontsize=16, fontweight='bold')
    
    # Color scheme
    age_color = '#3498DB'
    accent_color = '#E74C3C'
    
    # 1. Top-1 Accuracy with Confidence Intervals (Top Left)
    ax1 = axes[0, 0]
    
    groups = age_data['groups']
    accuracies = age_data['top1_acc']
    counts = age_data['counts']
    
    # Calculate confidence intervals
    conf_intervals = calculate_wilson_confidence_intervals(accuracies, counts)
    
    x_pos = np.arange(len(groups))
    bars = ax1.bar(x_pos, accuracies, color=age_color, alpha=0.8, 
                   capsize=5, width=0.7)
    ax1.errorbar(x_pos, accuracies, yerr=conf_intervals, fmt='none', 
                color='black', capsize=5, capthick=2)
    
    # Add value labels with confidence intervals
    for i, (bar, acc, ci, count) in enumerate(zip(bars, accuracies, conf_intervals, counts)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + ci + 0.015,
                f'{acc:.3f}\n±{ci:.3f}\n(n={count})', 
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax1.set_ylabel('Top-1 Accuracy')
    ax1.set_xlabel('Age Group (years)')
    ax1.set_title('Diagnostic Accuracy by Age Group\nwith 95% Confidence Intervals', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(groups)
    ax1.set_ylim(0, max(accuracies) + max(conf_intervals) + 0.1)
    
    # Add overall mean line
    overall_mean = np.average(accuracies, weights=counts)
    ax1.axhline(y=overall_mean, color=accent_color, linestyle='--', alpha=0.7, 
               label=f'Overall Mean: {overall_mean:.3f}')
    ax1.legend()
    
    # 2. Age Trend Analysis with Regression (Top Center)
    ax2 = axes[0, 1]
    
    midpoints = age_data['midpoints']
    
    # Scatter plot with varying sizes based on sample size
    sizes = [count * 3 for count in counts]  # Scale for visibility
    scatter = ax2.scatter(midpoints, accuracies, s=sizes, color=age_color, alpha=0.7, 
                         edgecolors='black', linewidth=1)
    
    # Add polynomial trend line (quadratic)
    poly_coeffs = np.polyfit(midpoints, accuracies, 2)
    x_smooth = np.linspace(min(midpoints), max(midpoints), 100)
    y_smooth = np.polyval(poly_coeffs, x_smooth)
    ax2.plot(x_smooth, y_smooth, '--', color=accent_color, linewidth=3, 
             label='Quadratic Trend')
    
    # Add linear trend for comparison
    linear_coeffs = np.polyfit(midpoints, accuracies, 1)
    y_linear = np.polyval(linear_coeffs, x_smooth)
    ax2.plot(x_smooth, y_linear, ':', color='gray', linewidth=2, 
             label='Linear Trend')
    
    # Calculate correlations
    correlation_linear, p_value_linear = stats.pearsonr(midpoints, accuracies)
    
    # Calculate R² for polynomial fit
    y_pred_poly = np.polyval(poly_coeffs, midpoints)
    r2_poly = r2_score(accuracies, y_pred_poly)
    
    ax2.set_xlabel('Age (years)')
    ax2.set_ylabel('Top-1 Accuracy')
    ax2.set_title(f'Age Trend Analysis\nLinear r={correlation_linear:.3f}, Poly R²={r2_poly:.3f}', fontweight='bold')
    ax2.legend()
    
    # Add age group labels
    for i, (x, y, group) in enumerate(zip(midpoints, accuracies, groups)):
        ax2.annotate(group, (x, y), xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=8, alpha=0.8)
    
    # 3. Top-K Performance Comparison (Top Right)
    ax3 = axes[0, 2]
    
    x_pos = np.arange(len(groups))
    width = 0.25
    
    bars1 = ax3.bar(x_pos - width, age_data['top1_acc'], width, 
                    label='Top-1', color='#E74C3C', alpha=0.8)
    bars2 = ax3.bar(x_pos, age_data['top3_acc'], width, 
                    label='Top-3', color='#F39C12', alpha=0.8)
    bars3 = ax3.bar(x_pos + width, age_data['top5_acc'], width, 
                    label='Top-5', color='#27AE60', alpha=0.8)
    
    ax3.set_ylabel('Accuracy')
    ax3.set_xlabel('Age Group')
    ax3.set_title('Top-K Performance by Age', fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(groups, rotation=45)
    ax3.legend()
    
    # 4. Sample Size and Bias Analysis (Bottom Left)
    ax4 = axes[1, 0]
    
    # Create dual y-axis plot
    ax4_twin = ax4.twinx()
    
    # Sample sizes as bars
    bars = ax4.bar(x_pos, counts, color=age_color, alpha=0.6, 
                   label='Sample Size')
    ax4.set_ylabel('Sample Size', color=age_color)
    ax4.tick_params(axis='y', labelcolor=age_color)
    
    # Bias magnitude as line
    bias_magnitudes = [abs(gap) for gap in age_data['parity_gap']]
    line = ax4_twin.plot(x_pos, bias_magnitudes, 'o-', color=accent_color, 
                        linewidth=3, markersize=8, label='Bias Magnitude')
    ax4_twin.set_ylabel('Absolute Parity Gap', color=accent_color)
    ax4_twin.tick_params(axis='y', labelcolor=accent_color)
    
    # Add threshold lines
    ax4_twin.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Moderate')
    ax4_twin.axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='High')
    
    ax4.set_xlabel('Age Group')
    ax4.set_title('Sample Size vs Bias Magnitude', fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(groups, rotation=45)
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 5. Confidence vs Accuracy Calibration (Bottom Center)
    ax5 = axes[1, 1]
    
    confidence_scores = age_data['confidence']
    
    # Scatter plot with trend line
    scatter = ax5.scatter(accuracies, confidence_scores, s=sizes, 
                         color=age_color, alpha=0.7, edgecolors='black', linewidth=1)
    
    # Perfect calibration line
    min_val = min(min(accuracies), min(confidence_scores)/100)
    max_val = max(max(accuracies), max(confidence_scores)/100)
    ax5.plot([min_val, max_val], [min_val*100, max_val*100], 'r--', 
             alpha=0.7, linewidth=2, label='Perfect Calibration')
    
    # Calculate calibration correlation
    calib_corr, calib_p = stats.pearsonr(accuracies, confidence_scores)
    
    ax5.set_xlabel('Top-1 Accuracy')
    ax5.set_ylabel('Average Confidence (%)')
    ax5.set_title(f'Confidence-Accuracy Calibration\nby Age (r={calib_corr:.3f})', fontweight='bold')
    ax5.legend()
    
    # Add age group labels
    for i, (x, y, group) in enumerate(zip(accuracies, confidence_scores, groups)):
        ax5.annotate(group, (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8)
    
    # 6. Performance Statistics Summary (Bottom Right)
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Create detailed statistics table
    table_data = []
    for i, group in enumerate(groups):
        ci = conf_intervals[i]
        bias_dir = '↑' if age_data['parity_gap'][i] > 0 else ('↓' if age_data['parity_gap'][i] < 0 else '→')
        
        table_data.append([
            group,
            f"{counts[i]}",
            f"{accuracies[i]:.3f}",
            f"±{ci:.3f}",
            f"{confidence_scores[i]:.1f}%",
            f"{abs(age_data['parity_gap'][i]):.3f} {bias_dir}"
        ])
    
    headers = ['Age Group', 'N', 'Accuracy', '95% CI', 'Confidence', 'Bias']
    
    table = ax6.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.0)
    
    # Color code header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#E8E8E8')
        table[(0, i)].set_text_props(weight='bold')
    
    # Color code best and worst performing groups
    best_idx = accuracies.index(max(accuracies)) + 1
    worst_idx = accuracies.index(min(accuracies)) + 1
    
    for j in range(len(headers)):
        table[(best_idx, j)].set_facecolor('#d4f4dd')  # Light green
        table[(worst_idx, j)].set_facecolor('#f4d4d4')  # Light red
    
    ax6.set_title('Age Group Performance Summary\n(Green=Best, Red=Worst)', 
                 fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('age_trend_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def print_age_statistics():
    """Print detailed age-based statistics"""
    
    age_data = load_age_data()
    
    print("=" * 80)
    print("AGE-BASED PERFORMANCE ANALYSIS")
    print("Multi-Agent Medical Diagnostic Systems")
    print("=" * 80)
    
    groups = age_data['groups']
    accuracies = age_data['top1_acc']
    counts = age_data['counts']
    confidence_scores = age_data['confidence']
    midpoints = age_data['midpoints']
    
    # Calculate confidence intervals
    conf_intervals = calculate_wilson_confidence_intervals(accuracies, counts)
    
    print(f"\nOVERALL STATISTICS:")
    print(f"• Total Samples: {sum(counts)}")
    print(f"• Age Groups: {len(groups)}")
    print(f"• Overall Accuracy: {np.average(accuracies, weights=counts):.3f}")
    print(f"• Performance Range: {max(accuracies) - min(accuracies):.3f}")
    print(f"• Maximum Bias Magnitude: {max([abs(gap) for gap in age_data['parity_gap']]):.3f}")
    
    # Best and worst performing groups
    best_idx = accuracies.index(max(accuracies))
    worst_idx = accuracies.index(min(accuracies))
    
    print(f"\nPERFORMANCE EXTREMES:")
    print(f"• Best Performing: {groups[best_idx]} ({accuracies[best_idx]:.3f} accuracy, n={counts[best_idx]})")
    print(f"• Worst Performing: {groups[worst_idx]} ({accuracies[worst_idx]:.3f} accuracy, n={counts[worst_idx]})")
    
    # Trend analysis
    correlation, p_value = stats.pearsonr(midpoints, accuracies)
    print(f"\nTREND ANALYSIS:")
    print(f"• Linear Correlation (r): {correlation:.3f}")
    print(f"• P-value: {p_value:.3f}")
    print(f"• Trend Significance: {'Significant' if p_value < 0.05 else 'Not Significant'}")
    
    # Detailed group statistics
    print(f"\nDETAILED GROUP STATISTICS:")
    print(f"{'Group':<8} {'N':<4} {'Accuracy':<9} {'95% CI':<12} {'Confidence':<11} {'Bias Gap':<10}")
    print("-" * 60)
    
    for i, group in enumerate(groups):
        ci = conf_intervals[i]
        bias = age_data['parity_gap'][i]
        bias_str = f"{bias:+.3f}"
        
        print(f"{group:<8} {counts[i]:<4} {accuracies[i]:<9.3f} ±{ci:<11.3f} {confidence_scores[i]:<11.1f}% {bias_str:<10}")
    
    print("\n" + "=" * 80)

def main():
    """Main function to run age trend analysis"""
    
    print("Generating age-based performance trend analysis...")
    
    # Create the visualization
    fig = create_age_trend_analysis()
    
    # Print detailed statistics
    print_age_statistics()
    
    print(f"\nAge trend analysis visualization saved as: age_trend_analysis.png")
    print("Analysis complete!")

if __name__ == "__main__":
    main()