#!/usr/bin/env python3
"""
Gender-Based Performance Pattern Analysis
Multi-Agent Medical Diagnostic Systems

Focused analysis on gender-related performance patterns and bias
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
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

def load_gender_data():
    """Load gender performance data from CSV"""
    
    gender_df = pd.read_csv('demographic_metrics/gender_metrics.csv')
    
    gender_data = {
        'groups': gender_df['Group'].tolist(),
        'counts': gender_df['Count'].tolist(),
        'top1_acc': gender_df['Top-1 Accuracy'].tolist(),
        'top3_acc': gender_df['Top-3 Accuracy'].tolist(),
        'top5_acc': gender_df['Top-5 Accuracy'].tolist(),
        'top7_acc': gender_df['Top-7 Accuracy'].tolist(),
        'confidence': gender_df['Avg Confidence (All Cases)'].tolist(),
        'parity_gap': gender_df['Parity Gap (vs Baseline)'].tolist(),
        'volatility': gender_df['Performance Volatility (Std Dev)'].iloc[0]
    }
    
    return gender_data

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

def perform_statistical_tests(gender_data):
    """Perform statistical significance tests"""
    
    accuracies = gender_data['top1_acc']
    counts = gender_data['counts']
    
    # Chi-square test for independence
    correct = [int(acc * count) for acc, count in zip(accuracies, counts)]
    incorrect = [count - corr for count, corr in zip(counts, correct)]
    contingency_table = np.array([correct, incorrect])
    
    chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency_table)
    
    # Pairwise Fisher's exact tests
    fisher_results = {}
    groups = gender_data['groups']
    
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            table = [[correct[i], incorrect[i]], [correct[j], incorrect[j]]]
            odds_ratio, fisher_p = fisher_exact(table)
            fisher_results[f"{groups[i]} vs {groups[j]}"] = {
                'odds_ratio': odds_ratio,
                'p_value': fisher_p
            }
    
    return {
        'chi2_stat': chi2_stat,
        'chi2_p': chi2_p,
        'fisher_results': fisher_results
    }

def create_gender_analysis():
    """Create comprehensive gender performance analysis"""
    
    gender_data = load_gender_data()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Gender-Based Performance Patterns in Medical Diagnosis\nHighest Magnitude Emergent Bias Analysis', 
                fontsize=16, fontweight='bold')
    
    # Color scheme for gender groups
    gender_colors = ['#FF6B9D', '#4ECDC4', '#45B7D1']  # Female, Male, Other
    
    groups = gender_data['groups']
    accuracies = gender_data['top1_acc']
    counts = gender_data['counts']
    
    # Calculate confidence intervals
    conf_intervals = calculate_wilson_confidence_intervals(accuracies, counts)
    
    # 1. Top-1 Accuracy with Confidence Intervals (Top Left)
    ax1 = axes[0, 0]
    
    bars = ax1.bar(groups, accuracies, color=gender_colors, alpha=0.8, capsize=5)
    ax1.errorbar(groups, accuracies, yerr=conf_intervals, fmt='none', 
                color='black', capsize=5, capthick=2)
    
    # Add value labels with exact statistics
    for i, (bar, acc, ci, count) in enumerate(zip(bars, accuracies, conf_intervals, counts)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + ci + 0.02,
                f'{acc:.3f}\n±{ci:.3f}\n(n={count})', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax1.set_ylabel('Top-1 Accuracy')
    ax1.set_title('Diagnostic Accuracy by Gender\nwith 95% Confidence Intervals', fontweight='bold')
    ax1.set_ylim(0, max(accuracies) + max(conf_intervals) + 0.15)
    
    # Add baseline reference
    baseline = gender_data['top1_acc'][1]  # Male as baseline (parity gap = 0.000)
    ax1.axhline(y=baseline, color='red', linestyle='--', alpha=0.7, 
               label=f'Baseline (Male): {baseline:.3f}')
    ax1.legend()
    
    # 2. Top-K Performance Comparison (Top Center)
    ax2 = axes[0, 1]
    
    x_pos = np.arange(len(groups))
    width = 0.2
    
    bars1 = ax2.bar(x_pos - 1.5*width, gender_data['top1_acc'], width, 
                    label='Top-1', color='#E74C3C', alpha=0.8)
    bars2 = ax2.bar(x_pos - 0.5*width, gender_data['top3_acc'], width, 
                    label='Top-3', color='#F39C12', alpha=0.8)
    bars3 = ax2.bar(x_pos + 0.5*width, gender_data['top5_acc'], width, 
                    label='Top-5', color='#27AE60', alpha=0.8)
    bars4 = ax2.bar(x_pos + 1.5*width, gender_data['top7_acc'], width, 
                    label='Top-7', color='#9B59B6', alpha=0.8)
    
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Gender Groups')
    ax2.set_title('Top-K Performance by Gender', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(groups)
    ax2.legend()
    
    # 3. Sample Size Weighted Analysis (Top Right)
    ax3 = axes[0, 2]
    
    # Create bars with width proportional to sample size
    max_count = max(counts)
    widths = [count / max_count * 0.8 for count in counts]
    
    x_pos = np.arange(len(groups))
    bars = ax3.bar(x_pos, accuracies, width=widths, alpha=0.7, 
                  color=gender_colors)
    
    # Add sample size labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'n={count}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax3.set_ylabel('Top-1 Accuracy')
    ax3.set_xlabel('Gender Groups')
    ax3.set_title('Sample Size Weighted Performance\n(Bar width ∝ sample size)', fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(groups)
    
    # 4. Bias Magnitude and Direction Analysis (Bottom Left)
    ax4 = axes[1, 0]
    
    parity_gaps = gender_data['parity_gap']
    abs_gaps = [abs(gap) for gap in parity_gaps]
    
    bars = ax4.bar(groups, abs_gaps, color=gender_colors, alpha=0.8)
    
    # Add bias direction indicators
    for i, (bar, gap) in enumerate(zip(bars, parity_gaps)):
        height = bar.get_height()
        direction = '↑' if gap > 0 else ('↓' if gap < 0 else '→')
        color = 'green' if gap > 0 else ('red' if gap < 0 else 'gray')
        
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{abs(gap):.3f} {direction}', ha='center', va='bottom', 
                fontweight='bold', color=color, fontsize=10)
    
    # Add threshold lines
    ax4.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Moderate (0.1)')
    ax4.axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='High (0.2)')
    
    ax4.set_ylabel('Absolute Parity Gap')
    ax4.set_xlabel('Gender Groups')
    ax4.set_title('Bias Magnitude by Gender\n(↑ positive bias, ↓ negative bias)', fontweight='bold')
    ax4.legend()
    
    # 5. Confidence vs Accuracy Calibration (Bottom Center)
    ax5 = axes[1, 1]
    
    confidence_scores = gender_data['confidence']
    
    # Scatter plot with varying sizes
    sizes = [count * 0.5 for count in counts]  # Scale for visibility
    for i, (acc, conf, size, group, color) in enumerate(zip(accuracies, confidence_scores, sizes, groups, gender_colors)):
        ax5.scatter(acc, conf, s=size, color=color, alpha=0.7, 
                   edgecolors='black', linewidth=1, label=group)
    
    # Perfect calibration line
    min_val = min(min(accuracies), min(confidence_scores)/100)
    max_val = max(max(accuracies), max(confidence_scores)/100)
    ax5.plot([min_val, max_val], [min_val*100, max_val*100], 'r--', 
             alpha=0.7, linewidth=2, label='Perfect Calibration')
    
    # Calculate calibration error for each group
    calib_errors = [abs(acc*100 - conf) for acc, conf in zip(accuracies, confidence_scores)]
    
    ax5.set_xlabel('Top-1 Accuracy')
    ax5.set_ylabel('Average Confidence (%)')
    ax5.set_title('Confidence-Accuracy Calibration\nby Gender', fontweight='bold')
    ax5.legend()
    
    # 6. Statistical Significance and Summary (Bottom Right)
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Perform statistical tests
    stats_results = perform_statistical_tests(gender_data)
    
    # Create summary table
    table_data = []
    for i, group in enumerate(groups):
        ci = conf_intervals[i]
        bias_dir = '↑' if parity_gaps[i] > 0 else ('↓' if parity_gaps[i] < 0 else '→')
        calib_error = abs(accuracies[i]*100 - confidence_scores[i])
        
        table_data.append([
            group,
            f"{counts[i]}",
            f"{accuracies[i]:.3f}",
            f"±{ci:.3f}",
            f"{confidence_scores[i]:.1f}%",
            f"{abs(parity_gaps[i]):.3f} {bias_dir}",
            f"{calib_error:.1f}%"
        ])
    
    headers = ['Gender', 'N', 'Accuracy', '95% CI', 'Confidence', 'Bias Gap', 'Calib Error']
    
    table = ax6.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='upper center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)
    
    # Color code header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#E8E8E8')
        table[(0, i)].set_text_props(weight='bold')
    
    # Color code performance levels
    best_idx = accuracies.index(max(accuracies)) + 1
    worst_idx = accuracies.index(min(accuracies)) + 1
    
    for j in range(len(headers)):
        table[(best_idx, j)].set_facecolor('#d4f4dd')  # Light green
        table[(worst_idx, j)].set_facecolor('#f4d4d4')  # Light red
    
    # Add statistical test results
    chi2_text = f"Chi-square test: χ² = {stats_results['chi2_stat']:.3f}, p = {stats_results['chi2_p']:.3f}"
    significance = "Significant" if stats_results['chi2_p'] < 0.05 else "Not Significant"
    
    ax6.text(0.5, 0.3, f"Statistical Tests:\n{chi2_text}\n({significance})", 
             transform=ax6.transAxes, ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
             fontsize=10, fontweight='bold')
    
    ax6.set_title('Gender Performance Summary\n(Green=Best, Red=Worst)', 
                 fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('gender_bias_detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig, stats_results

def print_gender_statistics():
    """Print detailed gender-based statistics for Table 2"""
    
    gender_data = load_gender_data()
    
    print("=" * 80)
    print("GENDER-BASED PERFORMANCE ANALYSIS")
    print("Section 2.2 - Table 2: Gender Performance Metrics")
    print("=" * 80)
    
    groups = gender_data['groups']
    accuracies = gender_data['top1_acc']
    counts = gender_data['counts']
    confidence_scores = gender_data['confidence']
    parity_gaps = gender_data['parity_gap']
    
    # Calculate confidence intervals
    conf_intervals = calculate_wilson_confidence_intervals(accuracies, counts)
    
    print(f"\nOVERALL GENDER STATISTICS:")
    print(f"• Groups: {len(groups)}")
    print(f"• Total Samples: {sum(counts)}")
    print(f"• Performance Range: {max(accuracies) - min(accuracies):.3f}")
    print(f"• Maximum Bias Magnitude: {max([abs(gap) for gap in parity_gaps]):.3f}")
    
    # Best and worst performing groups
    best_idx = accuracies.index(max(accuracies))
    worst_idx = accuracies.index(min(accuracies))
    
    print(f"• Best Performing: {groups[best_idx]} ({accuracies[best_idx]:.3f})")
    print(f"• Worst Performing: {groups[worst_idx]} ({accuracies[worst_idx]:.3f})")
    
    print(f"\nTABLE 2: GENDER PERFORMANCE METRICS WITH CONFIDENCE INTERVALS")
    print("=" * 90)
    print(f"{'Gender':<8} {'N':<4} {'Accuracy':<9} {'95% CI Lower':<12} {'95% CI Upper':<12} {'Confidence':<11} {'Parity Gap':<11}")
    print("-" * 90)
    
    for i, group in enumerate(groups):
        ci = conf_intervals[i]
        ci_lower = accuracies[i] - ci
        ci_upper = accuracies[i] + ci
        
        print(f"{group:<8} {counts[i]:<4} {accuracies[i]:<9.3f} {ci_lower:<12.3f} {ci_upper:<12.3f} {confidence_scores[i]:<11.1f}% {parity_gaps[i]:<+11.3f}")
    
    # Statistical significance
    stats_results = perform_statistical_tests(gender_data)
    
    print(f"\nSTATISTICAL SIGNIFICANCE TESTS:")
    print(f"• Chi-square statistic: {stats_results['chi2_stat']:.3f}")
    print(f"• Chi-square p-value: {stats_results['chi2_p']:.3f}")
    print(f"• Statistical significance: {'Significant (p < 0.05)' if stats_results['chi2_p'] < 0.05 else 'Not significant (p ≥ 0.05)'}")
    
    print(f"\nPAIRWISE COMPARISONS (Fisher's Exact Test):")
    for comparison, result in stats_results['fisher_results'].items():
        print(f"• {comparison}: OR = {result['odds_ratio']:.3f}, p = {result['p_value']:.3f}")
    
    print("\n" + "=" * 80)

def main():
    """Main function to run gender analysis"""
    
    print("Generating gender-based performance pattern analysis...")
    
    # Create the visualization
    fig, stats_results = create_gender_analysis()
    
    # Print detailed statistics for Table 2
    print_gender_statistics()
    
    print(f"\nGender bias detailed analysis saved as: gender_bias_detailed_analysis.png")
    print("Ready for Section 2.2 and Table 2!")

if __name__ == "__main__":
    main()