import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Data from CSV files
age_data = {
    'Group': ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60+'],
    'Count': [36, 21, 38, 29, 21, 33, 36],
    'Top-1 Accuracy': [0.639, 0.524, 0.500, 0.552, 0.619, 0.727, 0.611],
    'Top-3 Accuracy': [0.694, 0.810, 0.658, 0.690, 0.762, 0.788, 0.722],
    'Top-5 Accuracy': [0.806, 0.857, 0.711, 0.724, 0.762, 0.818, 0.833],
    'Avg Confidence': [81.667, 82.143, 77.237, 82.241, 82.143, 84.091, 80.972],
    'Performance Volatility': [0.078, 0.078, 0.078, 0.078, 0.078, 0.078, 0.078],
    'Parity Gap': [0.139, 0.024, 0.000, 0.052, 0.119, 0.227, 0.111]
}

gender_data = {
    'Group': ['Female', 'Male', 'Other'],
    'Count': [92, 115, 7],
    'Top-1 Accuracy': [0.500, 0.661, 0.857],
    'Top-3 Accuracy': [0.652, 0.774, 0.857],
    'Top-5 Accuracy': [0.685, 0.843, 1.000],
    'Avg Confidence': [80.598, 81.652, 85.000],
    'Performance Volatility': [0.179, 0.179, 0.179],
    'Parity Gap': [-0.161, 0.000, 0.196]
}

smoking_data = {
    'Group': ['Non-smoker', 'Smoker', 'Unknown'],
    'Count': [94, 30, 90],
    'Top-1 Accuracy': [0.585, 0.733, 0.567],
    'Top-3 Accuracy': [0.713, 0.833, 0.689],
    'Top-5 Accuracy': [0.798, 0.833, 0.744],
    'Avg Confidence': [79.947, 84.167, 81.778],
    'Performance Volatility': [0.091, 0.091, 0.091],
    'Parity Gap': [0.000, 0.148, -0.018]
}

symptom_data = {
    'Group': ['Atypical/Vague', 'Classic Textbook', 'Multi-System', 'Single Symptom', 'Unknown'],
    'Count': [73, 23, 66, 47, 5],
    'Top-1 Accuracy': [0.521, 0.522, 0.591, 0.745, 0.800],
    'Top-3 Accuracy': [0.630, 0.652, 0.758, 0.872, 0.800],
    'Top-5 Accuracy': [0.726, 0.652, 0.788, 0.894, 1.000],
    'Avg Confidence': [81.438, 80.000, 80.833, 82.021, 85.000],
    'Performance Volatility': [0.129, 0.129, 0.129, 0.129, 0.129],
    'Parity Gap': [0.000, 0.001, 0.070, 0.224, 0.279]
}

# Convert to DataFrames
age_df = pd.DataFrame(age_data)
gender_df = pd.DataFrame(gender_data)
smoking_df = pd.DataFrame(smoking_data)
symptom_df = pd.DataFrame(symptom_data)

def create_accuracy_comparison_chart():
    """Create Top-K Accuracy comparison across demographics"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Top-K Diagnostic Accuracy Across Demographic Groups', fontsize=16, fontweight='bold')
    
    # Age Group
    ax1 = axes[0, 0]
    x_pos = np.arange(len(age_df))
    width = 0.25
    
    ax1.bar(x_pos - width, age_df['Top-1 Accuracy'], width, label='Top-1', color='#2E86AB', alpha=0.8)
    ax1.bar(x_pos, age_df['Top-3 Accuracy'], width, label='Top-3', color='#A23B72', alpha=0.8)
    ax1.bar(x_pos + width, age_df['Top-5 Accuracy'], width, label='Top-5', color='#F18F01', alpha=0.8)
    
    ax1.set_title('Age Groups', fontweight='bold')
    ax1.set_xlabel('Age Group')
    ax1.set_ylabel('Accuracy')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(age_df['Group'], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gender
    ax2 = axes[0, 1]
    x_pos = np.arange(len(gender_df))
    
    ax2.bar(x_pos - width, gender_df['Top-1 Accuracy'], width, label='Top-1', color='#2E86AB', alpha=0.8)
    ax2.bar(x_pos, gender_df['Top-3 Accuracy'], width, label='Top-3', color='#A23B72', alpha=0.8)
    ax2.bar(x_pos + width, gender_df['Top-5 Accuracy'], width, label='Top-5', color='#F18F01', alpha=0.8)
    
    ax2.set_title('Gender', fontweight='bold')
    ax2.set_xlabel('Gender')
    ax2.set_ylabel('Accuracy')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(gender_df['Group'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Smoking Status
    ax3 = axes[1, 0]
    x_pos = np.arange(len(smoking_df))
    
    ax3.bar(x_pos - width, smoking_df['Top-1 Accuracy'], width, label='Top-1', color='#2E86AB', alpha=0.8)
    ax3.bar(x_pos, smoking_df['Top-3 Accuracy'], width, label='Top-3', color='#A23B72', alpha=0.8)
    ax3.bar(x_pos + width, smoking_df['Top-5 Accuracy'], width, label='Top-5', color='#F18F01', alpha=0.8)
    
    ax3.set_title('Smoking Status', fontweight='bold')
    ax3.set_xlabel('Smoking Status')
    ax3.set_ylabel('Accuracy')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(smoking_df['Group'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Symptom Presentation
    ax4 = axes[1, 1]
    x_pos = np.arange(len(symptom_df))
    
    ax4.bar(x_pos - width, symptom_df['Top-1 Accuracy'], width, label='Top-1', color='#2E86AB', alpha=0.8)
    ax4.bar(x_pos, symptom_df['Top-3 Accuracy'], width, label='Top-3', color='#A23B72', alpha=0.8)
    ax4.bar(x_pos + width, symptom_df['Top-5 Accuracy'], width, label='Top-5', color='#F18F01', alpha=0.8)
    
    ax4.set_title('Symptom Presentation', fontweight='bold')
    ax4.set_xlabel('Presentation Type')
    ax4.set_ylabel('Accuracy')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(symptom_df['Group'], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('top_k_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_bias_magnitude_chart():
    """Create bias magnitude comparison using parity gaps"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Bias Magnitude Analysis: Parity Gaps Across Demographics', fontsize=16, fontweight='bold')
    
    # Color mapping for bias direction
    def get_color(value):
        if value > 0.1:
            return '#27AE60'  # Green for positive bias
        elif value < -0.1:
            return '#E74C3C'  # Red for negative bias
        else:
            return '#95A5A6'  # Gray for neutral
    
    # Age Group
    ax1 = axes[0, 0]
    colors_age = [get_color(val) for val in age_df['Parity Gap']]
    bars1 = ax1.bar(age_df['Group'], age_df['Parity Gap'], color=colors_age, alpha=0.8)
    ax1.set_title('Age Groups - Parity Gap', fontweight='bold')
    ax1.set_xlabel('Age Group')
    ax1.set_ylabel('Parity Gap (vs Baseline)')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xticklabels(age_df['Group'], rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, age_df['Parity Gap']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.02,
                f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    # Gender
    ax2 = axes[0, 1]
    colors_gender = [get_color(val) for val in gender_df['Parity Gap']]
    bars2 = ax2.bar(gender_df['Group'], gender_df['Parity Gap'], color=colors_gender, alpha=0.8)
    ax2.set_title('Gender - Parity Gap', fontweight='bold')
    ax2.set_xlabel('Gender')
    ax2.set_ylabel('Parity Gap (vs Baseline)')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    for bar, value in zip(bars2, gender_df['Parity Gap']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.02,
                f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    # Smoking Status
    ax3 = axes[1, 0]
    colors_smoking = [get_color(val) for val in smoking_df['Parity Gap']]
    bars3 = ax3.bar(smoking_df['Group'], smoking_df['Parity Gap'], color=colors_smoking, alpha=0.8)
    ax3.set_title('Smoking Status - Parity Gap', fontweight='bold')
    ax3.set_xlabel('Smoking Status')
    ax3.set_ylabel('Parity Gap (vs Baseline)')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    
    for bar, value in zip(bars3, smoking_df['Parity Gap']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.02,
                f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    # Symptom Presentation
    ax4 = axes[1, 1]
    colors_symptom = [get_color(val) for val in symptom_df['Parity Gap']]
    bars4 = ax4.bar(symptom_df['Group'], symptom_df['Parity Gap'], color=colors_symptom, alpha=0.8)
    ax4.set_title('Symptom Presentation - Parity Gap', fontweight='bold')
    ax4.set_xlabel('Presentation Type')
    ax4.set_ylabel('Parity Gap (vs Baseline)')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_xticklabels(symptom_df['Group'], rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    for bar, value in zip(bars4, symptom_df['Parity Gap']):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.02,
                f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    # Add legend
    legend_elements = [
        Patch(facecolor='#27AE60', label='Positive Bias (>0.1)'),
        Patch(facecolor='#95A5A6', label='Neutral (-0.1 to 0.1)'),
        Patch(facecolor='#E74C3C', label='Negative Bias (<-0.1)')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3)
    
    plt.tight_layout()
    plt.savefig('bias_magnitude_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_confidence_accuracy_chart():
    """Create confidence vs accuracy analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Confidence vs Accuracy Analysis Across Demographics', fontsize=16, fontweight='bold')
    
    datasets = [
        (age_df, 'Age Groups', axes[0, 0]),
        (gender_df, 'Gender', axes[0, 1]),
        (smoking_df, 'Smoking Status', axes[1, 0]),
        (symptom_df, 'Symptom Presentation', axes[1, 1])
    ]
    
    for df, title, ax in datasets:
        # Scatter plot with group labels
        scatter = ax.scatter(df['Top-1 Accuracy'], df['Avg Confidence'], 
                           s=df['Count']*3, alpha=0.7, c=range(len(df)), cmap='viridis')
        
        # Add group labels
        for i, group in enumerate(df['Group']):
            ax.annotate(group, (df['Top-1 Accuracy'].iloc[i], df['Avg Confidence'].iloc[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Add diagonal line for perfect calibration
        min_val = min(df['Top-1 Accuracy'].min()*100, df['Avg Confidence'].min())
        max_val = max(df['Top-1 Accuracy'].max()*100, df['Avg Confidence'].max())
        ax.plot([min_val/100, max_val/100], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Calibration')
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Top-1 Accuracy')
        ax.set_ylabel('Average Confidence (%)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('confidence_accuracy_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_sample_size_weighted_chart():
    """Create sample size weighted performance chart"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Sample Size Weighted Performance Analysis', fontsize=16, fontweight='bold')
    
    datasets = [
        (age_df, 'Age Groups', axes[0, 0]),
        (gender_df, 'Gender', axes[0, 1]),
        (smoking_df, 'Smoking Status', axes[1, 0]),
        (symptom_df, 'Symptom Presentation', axes[1, 1])
    ]
    
    for df, title, ax in datasets:
        # Create bars with width proportional to sample size
        max_count = df['Count'].max()
        widths = df['Count'] / max_count * 0.8  # Scale to max width of 0.8
        
        x_pos = np.arange(len(df))
        bars = ax.bar(x_pos, df['Top-1 Accuracy'], width=widths, alpha=0.7, 
                     color=plt.cm.Set3(np.linspace(0, 1, len(df))))
        
        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, df['Count'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'n={count}', ha='center', va='bottom', fontsize=9)
        
        ax.set_title(f'{title}\n(Bar width ∝ sample size)', fontweight='bold')
        ax.set_xlabel('Group')
        ax.set_ylabel('Top-1 Accuracy')
        ax.set_xticks(x_pos)
        if title == 'Symptom Presentation':
            ax.set_xticklabels(df['Group'], rotation=45, ha='right')
        else:
            ax.set_xticklabels(df['Group'])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sample_size_weighted_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Create all charts
    print("Creating Top-K Accuracy Comparison Chart...")
    create_accuracy_comparison_chart()
    
    print("Creating Bias Magnitude Analysis Chart...")
    create_bias_magnitude_chart()
    
    print("Creating Confidence vs Accuracy Analysis Chart...")
    create_confidence_accuracy_chart()
    
    print("Creating Sample Size Weighted Analysis Chart...")
    create_sample_size_weighted_chart()
    
    print("All charts have been saved as PNG files in the current directory.")