import json
import os
import pandas as pd
from typing import Dict, Any, List

# --- Constants ---
LOG_PATH = "base_files/logs"
LOG_FILE = "agentclinic_run_latest.json"

def load_and_process_log(filepath: str) -> pd.DataFrame:
    """Loads a JSON log file and normalizes the 'logs' section into a DataFrame."""
    print(f"Loading and processing log file: {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Log file not found: {filepath}")
    with open(filepath, "r", encoding='utf-8') as f:
        data = json.load(f)
    if "logs" not in data:
        raise KeyError("The key 'logs' was not found in the JSON file.")
    df = pd.json_normalize(data["logs"])
    print(f"Successfully loaded log. Shape: {df.shape}")
    return df

def extract_demographics_data(demo_str: str) -> pd.Series:
    """Extracts structured data from the demographics string."""
    # Initialize default values for all relevant demographic fields
    result = {
        "age_group": "Unknown",
        "gender": "Other",
        "smoking_status": "Unknown",
        "alcohol_use": "Unknown",
    }
    if isinstance(demo_str, str):
        for line in demo_str.split('\n'):
            if not line: continue
            key, *value_parts = line.split()
            value = ' '.join(value_parts)

            # Check for each key and assign the value
            if 'age_group' in key:
                result['age_group'] = value
            elif 'gender' in key:
                result['gender'] = value
            elif 'smoking_status' in key:
                result['smoking_status'] = value
            elif 'alcohol_use' in key:
                result['alcohol_use'] = value
                
    return pd.Series(result)

def extract_demographics_info(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts demographic information from the relevant column."""
    print("Extracting demographic information...")
    demo_columns = [col for col in df.columns if 'demo' in col.lower()]
    if not demo_columns:
        print("No demographics column found. Creating placeholder columns...")
        for col in ['age_group', 'gender', 'smoking_status', 'alcohol_use']: df[col] = 'Unknown'
        return df
    
    demo_col = demo_columns[0]
    categorized_results = df[demo_col].apply(extract_demographics_data)
    final_df = pd.concat([df, categorized_results], axis=1)
    if 'age_group' in final_df.columns:
        final_df['age_group'] = final_df['age_group'].replace('0-1', '0-10')
    print("Demographic extraction complete.")
    return final_df

def calculate_sensitivity_by_group(df: pd.DataFrame, group_col: str, actual_col: str = 'is_correct', group_order: List[str] = None) -> pd.DataFrame:
    """Calculates Sensitivity/Recall for a given demographic group."""
    if group_col not in df.columns: raise KeyError(f"Grouping column '{group_col}' not found.")
    if actual_col not in df.columns: raise KeyError(f"Actual column '{actual_col}' not found.")

    valid_df = df.dropna(subset=[actual_col, group_col]).copy()
    valid_df[actual_col] = pd.to_numeric(valid_df[actual_col], errors='coerce').astype('boolean')

    def calculate_metrics(group_data):
        tp = group_data[actual_col].sum()
        fn = (~group_data[actual_col]).sum()
        total_cases = len(group_data)
        sensitivity = (tp / total_cases * 100) if total_cases > 0 else 0
        return pd.Series({'Total_Cases': total_cases, 'Correct Diagnoses': tp, 'Missed Diagnoses': fn, 'Sensitivity_%': sensitivity})

    grouped = valid_df.groupby(group_col).apply(calculate_metrics, include_groups=False).reset_index()
    if group_order:
        grouped[group_col] = pd.Categorical(grouped[group_col], categories=group_order, ordered=True)
        grouped = grouped.sort_values(group_col)
    return grouped

def print_formatted_metrics(metrics_df: pd.DataFrame, group_title: str, group_col: str):
    """Prints sensitivity metrics in the user-specified format."""
    print("\n" + "="*50)
    print(f"SENSITIVITY ANALYSIS BY {group_title.upper()}")
    print("="*50)

    if metrics_df.empty:
        print(f"No {group_title.lower()} data was available to analyze.")
        return

    for _, row in metrics_df.iterrows():
        print(f"\n{group_col.replace('_', ' ').title()}: {row[group_col]}")
        print(f"  Sensitivity: {row['Sensitivity_%']:.0f}%")
        print(f"  Correct Diagnoses: {int(row['Correct Diagnoses'])}")
        print(f"  Missed Diagnoses: {int(row['Missed Diagnoses'])}")

# --- Main Execution ---
if __name__ == "__main__":
    try:
        log_filepath = os.path.join(LOG_PATH, LOG_FILE)
        processed_df = load_and_process_log(log_filepath)
        final_df = extract_demographics_info(processed_df)

        # --- Age Group Analysis ---
        age_order = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60+", "Unknown"]
        age_metrics_df = calculate_sensitivity_by_group(final_df, 'age_group', group_order=age_order)
        print_formatted_metrics(age_metrics_df, "Age Group", "age_group")

        # --- Gender Analysis ---
        gender_order = ["Male", "Female", "Other"]
        gender_metrics_df = calculate_sensitivity_by_group(final_df, 'gender', group_order=gender_order)
        print_formatted_metrics(gender_metrics_df, "Gender", "gender")

        # --- Smoking Status Analysis ---
        smoking_order = ["Smoker", "Non-smoker", "Unknown"]
        smoking_metrics_df = calculate_sensitivity_by_group(final_df, 'smoking_status', group_order=smoking_order)
        print_formatted_metrics(smoking_metrics_df, "Smoking Status", "smoking_status")
        
        # --- Alcohol Use Analysis ---
        alcohol_order = ["Drinker", "Non-drinker", "Unknown"]
        alcohol_metrics_df = calculate_sensitivity_by_group(final_df, 'alcohol_use', group_order=alcohol_order)
        print_formatted_metrics(alcohol_metrics_df, "Alcohol Use", "alcohol_use")
        
        print("\n" + "="*50)
        print("\nAnalysis complete. ✔️")

    except (FileNotFoundError, KeyError) as e:
        print(f"\n ERROR: {e}")
    except Exception as e:
        print(f"\n An unexpected error occurred: {e}")