import json
import os
import pandas as pd
from typing import Dict, Any, List
import numpy as np
import glob

# --- Constants with correct file paths ---
LOG_PATH = "base_files/logs"
BIAS_CORRECTED_FILE = "base_files/logs/MedQA_Ext_none_bias_corrected_20250725_114108.json"
ORIGINAL_FILE = "base_files/logs/original_agentclinic_run_latest.json"
# Verify files exist
if not os.path.exists(BIAS_CORRECTED_FILE):
    raise FileNotFoundError(f"Bias-corrected file not found: {BIAS_CORRECTED_FILE}")
if not os.path.exists(ORIGINAL_FILE):
    raise FileNotFoundError(f"Original file not found: {ORIGINAL_FILE}")

print(f"Using bias-corrected file: {BIAS_CORRECTED_FILE}")
print(f"Using original file: {ORIGINAL_FILE}")

def load_bias_corrected_results(filepath: str) -> pd.DataFrame:
    """Load bias-corrected results (new format)"""
    print(f"Loading bias-corrected results: {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Bias-corrected file not found: {filepath}")
    
    with open(filepath, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        df = pd.json_normalize(data)
        print(f"Loaded bias-corrected results. Shape: {df.shape}")
        
        # Debug: Show dialogue-related columns
        dialogue_cols = [col for col in df.columns if 'dialogue' in col.lower() or 'turn' in col.lower()]
        print(f"Dialogue-related columns found: {dialogue_cols}")
        
        # Check if we have dialogue data
        if 'num_dialogue_turns' in df.columns:
            sample_turns = df['num_dialogue_turns'].head(5).tolist()
            avg_turns = df['num_dialogue_turns'].mean()
            print(f"num_dialogue_turns samples: {sample_turns}, average: {avg_turns:.1f}")
        
        if 'dialogue_history' in df.columns:
            sample_lengths = [len(hist) if isinstance(hist, list) else 0 for hist in df['dialogue_history'].head(3)]
            print(f"dialogue_history sample lengths: {sample_lengths}")
        
        return df
    else:
        raise ValueError("Expected list format for bias-corrected file")

def load_original_demographics(filepath: str) -> pd.DataFrame:
    """Load original results with demographics (old format)"""
    print(f"Loading original demographics: {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Original file not found: {filepath}")
    
    with open(filepath, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "logs" in data:
        df = pd.json_normalize(data["logs"])
        print(f"Loaded original demographics. Shape: {df.shape}")
        return df
    else:
        raise ValueError("Expected dict format with 'logs' key for original file")

def extract_demographics_data(demo_str: str) -> pd.Series:
    """Extract structured data from demographics string"""
    result = {
        "age_group": "Unknown",
        "gender": "Other",
        "smoking_status": "Unknown", 
        "alcohol_use": "Unknown",
    }
    
    if isinstance(demo_str, str):
        for line in demo_str.split('\n'):
            if not line.strip(): 
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key, *value_parts = parts
            value = ' '.join(value_parts)

            if 'age_group' in key:
                result['age_group'] = value
            elif 'gender' in key:
                result['gender'] = value
            elif 'smoking_status' in key:
                result['smoking_status'] = value
            elif 'alcohol_use' in key:
                result['alcohol_use'] = value
                
    return pd.Series(result)

def count_dialogue_turns(dialogue_history: Any) -> int:
    """Count the number of dialogue turns from dialogue_history"""
    if not isinstance(dialogue_history, list):
        return 0
    
    # Count meaningful dialogue turns (excluding empty or very short entries)
    meaningful_turns = 0
    for entry in dialogue_history:
        if isinstance(entry, dict) and 'text' in entry:
            text = str(entry['text']).strip()
            if len(text) > 5:  # Only count meaningful dialogue turns
                meaningful_turns += 1
        elif isinstance(entry, str) and len(entry.strip()) > 5:
            meaningful_turns += 1
    
    return meaningful_turns

def extract_dialogue_count_from_original(original_df: pd.DataFrame) -> pd.DataFrame:
    """Extract dialogue count from original data if available"""
    dialogue_counts = []
    
    for _, row in original_df.iterrows():
        dialogue_count = 0
        
        # Try multiple potential sources for dialogue count
        potential_fields = [
            'dialogue_history', 'conversation_history', 'turns', 
            'dialogue_turns', 'conversation_turns', 'messages'
        ]
        
        for field in potential_fields:
            if field in row and pd.notna(row[field]):
                if field == 'dialogue_history':
                    dialogue_count = count_dialogue_turns(row[field])
                    break
                elif isinstance(row[field], (int, float)):
                    dialogue_count = int(row[field])
                    break
                elif isinstance(row[field], list):
                    dialogue_count = len(row[field])
                    break
        
        dialogue_counts.append(dialogue_count)
    
    result_df = original_df[['scenario_id']].copy()
    result_df['num_dialogues'] = dialogue_counts
    
    return result_df

def is_valid_dialogue_data(value):
    """Safely check if dialogue data is valid"""
    try:
        if pd.isna(value):
            return False
        if isinstance(value, (list, np.ndarray)):
            return len(value) > 0
        if isinstance(value, str):
            return len(value.strip()) > 0
        return bool(value)
    except:
        return False

def merge_results_with_demographics(bias_corrected_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    """Merge bias-corrected results with original demographics by scenario_id"""
    print("Merging bias-corrected results with original demographics...")
    
    # Extract demographics from original data
    if "demographics" in original_df.columns:
        demo_data = original_df["demographics"].apply(extract_demographics_data)
        original_with_demo = pd.concat([original_df[["scenario_id"]], demo_data], axis=1)
    else:
        print("Warning: No demographics column in original data")
        return bias_corrected_df
    
    # Extract dialogue counts from bias-corrected data first, then original as fallback
    dialogue_counts_new = []
    dialogue_source = "bias-corrected data"
    
    for _, row in bias_corrected_df.iterrows():
        dialogue_count = 0
        if 'dialogue_history' in row and is_valid_dialogue_data(row['dialogue_history']):
            dialogue_count = count_dialogue_turns(row['dialogue_history'])
        dialogue_counts_new.append(dialogue_count)
    
    # If bias-corrected data doesn't have dialogue info, try original
    if sum(dialogue_counts_new) == 0:
        print("No dialogue data in bias-corrected file, checking original...")
        dialogue_source = "original data"
        dialogue_df = extract_dialogue_count_from_original(original_df)
        original_with_demo = pd.merge(original_with_demo, dialogue_df, on="scenario_id", how="left")
    else:
        original_with_demo['num_dialogues'] = 0  # Will be filled from bias-corrected data
    
    # Keep key columns from bias-corrected results INCLUDING diagnoses considered count
    key_columns = ["scenario_id", "is_correct", "final_doctor_diagnosis", 
                   "correct_diagnosis", "determined_specialist", 
                   "tests_requested_count", "self_confidence"]
    
    # Add consultation analysis columns if they exist
    if "consultation_analysis.diagnoses_considered_count" in bias_corrected_df.columns:
        key_columns.append("consultation_analysis.diagnoses_considered_count")
    if "consultation_analysis.diagnoses_considered" in bias_corrected_df.columns:
        key_columns.append("consultation_analysis.diagnoses_considered")
    
    bias_corrected_key = bias_corrected_df[key_columns].copy()
    
    # Add dialogue count from bias-corrected data
    bias_corrected_key['num_dialogues'] = dialogue_counts_new
    
    # Merge on scenario_id
    merged_df = pd.merge(bias_corrected_key, original_with_demo, on="scenario_id", how="left")
    
    # Use dialogue count from bias-corrected data if available, otherwise from original
    if 'num_dialogues_y' in merged_df.columns:
        merged_df['num_dialogues'] = merged_df['num_dialogues_x'].fillna(merged_df['num_dialogues_y'])
        merged_df = merged_df.drop(['num_dialogues_x', 'num_dialogues_y'], axis=1)
    
    print(f"Merged dataset shape: {merged_df.shape}")
    print(f"Scenarios with demographics before consolidation: {merged_df['age_group'].notna().sum()}")
    print(f"Scenarios with dialogue data: {(merged_df['num_dialogues'] > 0).sum()}")
    print(f"Dialogue data source: {dialogue_source}")
    print(f"Average dialogue turns: {merged_df['num_dialogues'].mean():.1f}")
    
    # Add diagnoses considered count if available
    if "consultation_analysis.diagnoses_considered_count" in merged_df.columns:
        merged_df['num_diagnoses_considered'] = merged_df["consultation_analysis.diagnoses_considered_count"]
        print(f"Diagnoses considered data available for analysis")
        print(f"Average diagnoses considered: {merged_df['num_diagnoses_considered'].mean():.1f}")
    else:
        print("Warning: No diagnoses considered count found")
        merged_df['num_diagnoses_considered'] = 0
    
    # Age group consolidation
    if 'age_group' in merged_df.columns:
        print("\nAge group distribution before consolidation:")
        print(merged_df['age_group'].value_counts(dropna=False))
        
        merged_df['age_group'] = merged_df['age_group'].replace('0-1', '0-10')
        merged_df['age_group'] = merged_df['age_group'].fillna('Unknown')
        
        print("\nAge group distribution after consolidation:")
        print(merged_df['age_group'].value_counts(dropna=False))
        
        print(f"Scenarios with demographics after consolidation: {(merged_df['age_group'] != 'Unknown').sum()}")
    
    return merged_df

def calculate_accuracy_by_group(df: pd.DataFrame, group_col: str, actual_col: str = 'is_correct', 
                              group_order: List[str] = None, baseline_accuracy: float = None) -> pd.DataFrame:
    """Calculate accuracy metrics for demographic groups with enhanced metrics"""
    if group_col not in df.columns:
        raise KeyError(f"Grouping column '{group_col}' not found.")
    if actual_col not in df.columns:
        raise KeyError(f"Actual column '{actual_col}' not found.")

    valid_df = df.dropna(subset=[actual_col, group_col]).copy()
    valid_df[actual_col] = pd.to_numeric(valid_df[actual_col], errors='coerce').astype('boolean')

    def calculate_metrics(group_data):
        correct = group_data[actual_col].sum()
        incorrect = (~group_data[actual_col]).sum()
        total_cases = len(group_data)
        accuracy = (correct / total_cases * 100) if total_cases > 0 else 0
        
        # Calculate difference from baseline if provided
        diff_from_baseline = (accuracy - baseline_accuracy) if baseline_accuracy is not None else 0
        
        # Add average diagnoses considered if available
        avg_diagnoses = group_data['num_diagnoses_considered'].mean() if 'num_diagnoses_considered' in group_data else 0
        
        # Add average number of dialogues if available
        avg_dialogues = group_data['num_dialogues'].mean() if 'num_dialogues' in group_data else 0
        
        # Calculate 95% confidence interval for accuracy
        if total_cases > 0:
            p = correct / total_cases
            margin_error = 1.96 * np.sqrt((p * (1 - p)) / total_cases)
            ci_lower = max(0, (p - margin_error) * 100)
            ci_upper = min(100, (p + margin_error) * 100)
            confidence_interval = f"±{margin_error*100:.1f}%"
        else:
            confidence_interval = "N/A"
        
        return pd.Series({
            'Total_Cases': total_cases,
            'Correct_Diagnoses': correct,
            'Incorrect_Diagnoses': incorrect,
            'Accuracy_%': accuracy,
            'Diff_from_Baseline_%': diff_from_baseline,
            'Avg_Diagnoses_Considered': avg_diagnoses,
            'Avg_Dialogues': avg_dialogues,
            'Confidence_Interval': confidence_interval
        })

    grouped = valid_df.groupby(group_col, observed=True).apply(calculate_metrics, include_groups=False).reset_index()
    if group_order:
        grouped[group_col] = pd.Categorical(grouped[group_col], categories=group_order, ordered=True)
        grouped = grouped.sort_values(group_col)
    return grouped

def print_enhanced_metrics(metrics_df: pd.DataFrame, group_title: str, group_col: str, baseline_accuracy: float = None):
    """Print enhanced accuracy metrics including dialogues and baseline difference"""
    print("\n" + "="*70)
    print(f"ENHANCED BIAS-CORRECTED ANALYSIS BY {group_title.upper()}")
    print("="*70)
    
    if baseline_accuracy:
        print(f"Baseline Accuracy: {baseline_accuracy:.1f}%")
        print("-" * 50)

    if metrics_df.empty:
        print(f"No {group_title.lower()} data available.")
        return

    for _, row in metrics_df.iterrows():
        group_name = row[group_col]
        accuracy = row['Accuracy_%']
        diff_baseline = row['Diff_from_Baseline_%']
        total_cases = int(row['Total_Cases'])
        correct = int(row['Correct_Diagnoses'])
        avg_dialogues = row['Avg_Dialogues']
        confidence = row['Confidence_Interval']
        
        print(f"\n{group_col.replace('_', ' ').title()}: {group_name}")
        print(f"  Accuracy: {accuracy:.1f}% ({confidence})")
        if baseline_accuracy:
            print(f"  Diff from Baseline: {diff_baseline:+.1f} percentage points")
        print(f"  Cases: {correct}/{total_cases} correct")
        print(f"  Avg Dialogues: {avg_dialogues:.1f}")
        
        if 'Avg_Diagnoses_Considered' in row and row['Avg_Diagnoses_Considered'] > 0:
            print(f"  Avg Diagnoses Considered: {row['Avg_Diagnoses_Considered']:.1f}")

def analyze_dialogue_patterns(merged_df: pd.DataFrame):
    """Analyze patterns in dialogue counts and their relationship to accuracy"""
    print("\n" + "="*70)
    print("DIALOGUE PATTERN ANALYSIS")
    print("="*70)
    
    if 'num_dialogues' not in merged_df.columns or merged_df['num_dialogues'].sum() == 0:
        print("No dialogue data available for analysis.")
        return
    
    # Overall dialogue statistics
    dialogue_stats = merged_df['num_dialogues'].describe()
    print(f"\nDialogue Count Statistics:")
    print(f"  Mean: {dialogue_stats['mean']:.1f}")
    print(f"  Median: {dialogue_stats['50%']:.1f}")
    print(f"  Std Dev: {dialogue_stats['std']:.1f}")
    print(f"  Range: {dialogue_stats['min']:.0f} - {dialogue_stats['max']:.0f}")
    
    # Categorize by dialogue complexity
    merged_df['dialogue_complexity'] = pd.cut(
        merged_df['num_dialogues'], 
        bins=[0, 3, 6, 10, float('inf')], 
        labels=["Brief (1-3)", "Moderate (4-6)", "Extended (7-10)", "Lengthy (11+)"],
        include_lowest=True
    )
    
    # Calculate metrics by dialogue complexity
    baseline_acc = merged_df['is_correct'].mean() * 100
    dialogue_metrics = calculate_accuracy_by_group(merged_df, 'dialogue_complexity', baseline_accuracy=baseline_acc)
    print_enhanced_metrics(dialogue_metrics, "Dialogue Complexity", "dialogue_complexity", baseline_acc)
    
    # Correlation analysis
    correlation = merged_df[['num_dialogues', 'is_correct']].corr().iloc[0, 1]
    print(f"\nDialogue-Accuracy Correlation: {correlation:.3f}")
    
    if correlation > 0.1:
        print("  -> Longer dialogues tend to have better outcomes")
    elif correlation < -0.1:
        print("  -> Longer dialogues tend to have worse outcomes")
    else:
        print("  -> No strong correlation between dialogue length and outcomes")

def analyze_by_diagnoses_considered(merged_df: pd.DataFrame):
    """Analyze performance by number of diagnoses considered with enhanced metrics"""
    print("\n" + "="*70)
    print("ENHANCED ANALYSIS BY NUMBER OF DIAGNOSES CONSIDERED")
    print("="*70)
    
    if 'num_diagnoses_considered' not in merged_df.columns or merged_df['num_diagnoses_considered'].sum() == 0:
        print("No diagnoses considered data available for analysis.")
        return
    
    # Show distribution
    print("\nDistribution of diagnoses considered:")
    diag_counts = merged_df['num_diagnoses_considered'].value_counts().sort_index()
    for count, cases in diag_counts.items():
        percentage = (cases / len(merged_df)) * 100
        avg_accuracy = merged_df[merged_df['num_diagnoses_considered'] == count]['is_correct'].mean() * 100
        print(f"  {count} diagnoses: {cases} cases ({percentage:.1f}%) - {avg_accuracy:.1f}% accuracy")
    
    # Categorize by complexity
    merged_df['diagnostic_complexity'] = pd.cut(
        merged_df['num_diagnoses_considered'], 
        bins=[0, 2, 4, 6, float('inf')], 
        labels=["Simple (1-2)", "Moderate (3-4)", "Complex (5-6)", "Very Complex (7+)"],
        include_lowest=True
    )
    
    # Calculate metrics by complexity with baseline
    baseline_acc = merged_df['is_correct'].mean() * 100
    complexity_metrics = calculate_accuracy_by_group(merged_df, 'diagnostic_complexity', baseline_accuracy=baseline_acc)
    print_enhanced_metrics(complexity_metrics, "Diagnostic Complexity", "diagnostic_complexity", baseline_acc)

def compare_with_original_performance(merged_df: pd.DataFrame, original_df: pd.DataFrame):
    """Compare new vs old performance by demographics with enhanced metrics"""
    print("\n" + "="*70)
    print("BIAS-CORRECTION IMPACT BY DEMOGRAPHICS")
    print("="*70)
    
    if "demographics" in original_df.columns:
        original_demo = original_df["demographics"].apply(extract_demographics_data)
        original_with_demo = pd.concat([original_df[["is_correct"]], original_demo], axis=1)
        
        # Add same consolidation to original data
        original_with_demo['age_group'] = original_with_demo['age_group'].replace('0-1', '0-10')
        
        # Calculate overall baseline accuracies
        original_baseline = original_with_demo['is_correct'].mean() * 100
        new_baseline = merged_df['is_correct'].mean() * 100
        overall_improvement = new_baseline - original_baseline
        
        print(f"Overall Performance Comparison:")
        print(f"  Original Accuracy: {original_baseline:.1f}%")
        print(f"  Bias-Corrected Accuracy: {new_baseline:.1f}%")
        print(f"  Overall Improvement: {overall_improvement:+.1f} percentage points")
        
        for group_col in ["age_group", "gender", "smoking_status", "alcohol_use"]:
            print(f"\n--- {group_col.replace('_', ' ').title()} Comparison ---")
            
            original_metrics = calculate_accuracy_by_group(original_with_demo, group_col)
            new_metrics = calculate_accuracy_by_group(merged_df, group_col)
            
            comparison = pd.merge(original_metrics, new_metrics, on=group_col, suffixes=('_Original', '_BiasFixed'))
            comparison['Improvement'] = comparison['Accuracy_%_BiasFixed'] - comparison['Accuracy_%_Original']
            
            for _, row in comparison.iterrows():
                group_name = row[group_col]
                orig_acc = row['Accuracy_%_Original']
                new_acc = row['Accuracy_%_BiasFixed']
                improvement = row['Improvement']
                avg_dialogues = row.get('Avg_Dialogues_BiasFixed', 0)
                
                print(f"  {group_name}:")
                print(f"    Original: {orig_acc:.1f}% → Bias-Fixed: {new_acc:.1f}%")
                print(f"    Improvement: {improvement:+.1f} percentage points")
                if avg_dialogues > 0:
                    print(f"    Avg Dialogues: {avg_dialogues:.1f}")

def generate_summary_table(merged_df: pd.DataFrame):
    """Generate a comprehensive summary table with all key metrics"""
    print("\n" + "="*80)
    print("COMPREHENSIVE SUMMARY TABLE")
    print("="*80)
    
    baseline_accuracy = merged_df['is_correct'].mean() * 100
    
    # Analyze each demographic category
    categories = [
        ('age_group', ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60+", "Unknown"]),
        ('gender', ["Male", "Female", "Other"]),
        ('smoking_status', ["Smoker", "Non-smoker", "Unknown"]),
        ('alcohol_use', ["Drinker", "Non-drinker", "Unknown"])
    ]
    
    print(f"\n{'Category':<20} {'Group':<15} {'Accuracy':<10} {'Diff':<8} {'Cases':<8} {'Dialogues':<10} {'Diagnoses':<10}")
    print("-" * 90)
    
    for category, order in categories:
        if category in merged_df.columns:
            metrics = calculate_accuracy_by_group(merged_df, category, group_order=order, baseline_accuracy=baseline_accuracy)
            
            for _, row in metrics.iterrows():
                cat_display = category.replace('_', ' ').title() if row.name == 0 else ""
                group_name = row[category]
                accuracy = row['Accuracy_%']
                diff_baseline = row['Diff_from_Baseline_%']
                cases = row['Total_Cases']
                dialogues = row['Avg_Dialogues']
                diagnoses = row['Avg_Diagnoses_Considered']
                
                print(f"{cat_display:<20} {group_name:<15} {accuracy:>6.1f}%   {diff_baseline:>+5.1f}   {cases:>5.0f}    {diagnoses:>6.1f}")

# --- Main Execution ---
if __name__ == "__main__":
    try:
        print("Current working directory:", os.getcwd())
        print(f"Using bias-corrected file: {BIAS_CORRECTED_FILE}")
        print(f"Using original file: {ORIGINAL_FILE}")
        
        # Load both datasets
        bias_corrected_df = load_bias_corrected_results(BIAS_CORRECTED_FILE)
        original_df = load_original_demographics(ORIGINAL_FILE)
        
        # Merge the datasets
        merged_df = merge_results_with_demographics(bias_corrected_df, original_df)
        
        # Overall performance
        total_cases = len(merged_df)
        correct_cases = merged_df['is_correct'].sum()
        overall_accuracy = (correct_cases / total_cases) * 100
        avg_diagnoses_considered = merged_df['num_diagnoses_considered'].mean()
        
        print("\n" + "="*70)
        print("ENHANCED BIAS-CORRECTED PERFORMANCE WITH DEMOGRAPHICS")
        print("="*70)
        print(f"Total Cases: {total_cases}")
        print(f"Correct Diagnoses: {correct_cases}")
        print(f"Overall Accuracy: {overall_accuracy:.1f}%")
        print(f"Average Diagnoses Considered: {avg_diagnoses_considered:.1f}")
        
        # Calculate baseline accuracy for demographic analysis
        baseline_accuracy = overall_accuracy
        
        # Analyze by demographics with enhanced metrics
        demographic_categories = [
            ("age_group", "Age Group", ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60+", "Unknown"]),
            ("gender", "Gender", ["Male", "Female", "Other"]),
            ("smoking_status", "Smoking Status", ["Smoker", "Non-smoker", "Unknown"]),
            ("alcohol_use", "Alcohol Use", ["Drinker", "Non-drinker", "Unknown"])
        ]
        
        for col, title, order in demographic_categories:
            if col in merged_df.columns:
                metrics = calculate_accuracy_by_group(merged_df, col, group_order=order, baseline_accuracy=baseline_accuracy)
                print_enhanced_metrics(metrics, title, col, baseline_accuracy)
        
        # Analyze dialogue patterns
        analyze_dialogue_patterns(merged_df)
        
        # Analyze by diagnoses considered
        analyze_by_diagnoses_considered(merged_df)
        
        # Compare with original performance
        compare_with_original_performance(merged_df, original_df)
        
        # Generate summary table
        generate_summary_table(merged_df)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check that the file paths are correct and the files exist.")
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()