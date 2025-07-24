import json
import os
import pandas as pd

# --- Constants ---
LOG_PATH = "base_files/logs"
LOG_FILE = "agentclinic_run_latest.json"

def load_and_process_log(filepath: str) -> pd.DataFrame:
    """
    Loads a JSON log file and normalizes the 'logs' section into a DataFrame.

    Args:
        filepath: The full path to the JSON log file.

    Returns:
        A processed pandas DataFrame.
    """
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
    """Extract structured data from the demographics string."""
    result = {
        "age_group": "Unknown",
        "gender": "Other",
        "smoking_status": "Unknown",
        "alcohol_use": "Unknown",
        "drug_use": "Unknown",
        "occupation_type": "Unknown",
        "comorbidity_status": "Unknown",
        "symptom_presentation": "Unknown"
    }
    
    if isinstance(demo_str, str):
        lines = demo_str.split('\n')
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            key = parts[0]
            value = ' '.join(parts[1:])
            
            if 'age_group' in key:
                result['age_group'] = value
            elif 'gender' in key:
                result['gender'] = value
            elif 'smoking_status' in key:
                result['smoking_status'] = value
            # ... (other keys can be added here if needed)
    
    return pd.Series(result)

def calculate_sensitivity_by_demographic(df: pd.DataFrame, group_col: str, actual_col: str = 'is_correct') -> pd.DataFrame:
    """
    Calculates diagnostic sensitivity for a given demographic group.
    Sensitivity = TP / (TP + FN) = True Positives / All Actual Positives

    Args:
        df: The DataFrame containing diagnosis results and demographic data.
        group_col: The column to group by (e.g., 'age_group', 'gender').
        actual_col: The column containing correctness values (True/False).

    Returns:
        A DataFrame summarizing sensitivity metrics for each demographic group.
    """
    if group_col not in df.columns:
        raise KeyError(f"Demographic column '{group_col}' not found in DataFrame.")
    if actual_col not in df.columns:
        raise KeyError(f"Actuals column '{actual_col}' not found in DataFrame.")

    # Ensure the correctness column is boolean and drop invalid rows
    df[actual_col] = pd.to_numeric(df[actual_col], errors='coerce').astype('boolean')
    valid_df = df.dropna(subset=[actual_col, group_col])
    
    def calculate_metrics(group_data):
        """Helper to calculate metrics for a single group."""
        tp = group_data[actual_col].sum()  # True Positives (correct diagnoses)
        fn = (~group_data[actual_col]).sum() # False Negatives (missed diagnoses)
        total_cases = len(group_data)
        sensitivity = (tp / total_cases * 100) if total_cases > 0 else 0
        
        return pd.Series({
            'Total_Cases': total_cases,
            'Correct_Diagnoses (TP)': tp,
            'Missed_Diagnoses (FN)': fn,
            'Sensitivity_%': sensitivity
        })

    # Group, apply calculation, and return
    grouped = valid_df.groupby(group_col).apply(calculate_metrics, include_groups=False).reset_index()
    
    # Optional: Sort by a predefined order if the group is categorical
    sort_orders = {
        'age_group': ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60+", "Unknown"],
        'gender': ["Male", "Female", "Other"],
        'smoking_status': ["Smoker", "Non-smoker", "Unknown"]
    }
    if group_col in sort_orders:
        grouped[group_col] = pd.Categorical(grouped[group_col], categories=sort_orders[group_col], ordered=True)
        grouped = grouped.sort_values(group_col)

    return grouped

def extract_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts all demographic features from the relevant column.
    """
    print("Extracting demographic features...")
    demo_col = next((col for col in df.columns if 'demo' in col.lower()), None)
    
    if demo_col:
        print(f"Found demographics column: '{demo_col}'")
        categorized_results = df[demo_col].apply(extract_demographics_data)
        final_df = pd.concat([df, categorized_results], axis=1)
        
        # Standardize age group '0-1' to '0-10'
        if 'age_group' in final_df.columns:
            final_df['age_group'] = final_df['age_group'].replace('0-1', '0-10')
        
        print("Demographic extraction complete.")
        return final_df
    else:
        print("No demographics column found. Creating placeholder columns.")
        df['age_group'] = 'Unknown'
        df['gender'] = 'Unknown'
        df['smoking_status'] = 'Unknown'
        return df

def display_sensitivity_report(metrics_df: pd.DataFrame, group_name: str):
    """Prints a formatted sensitivity analysis report."""
    print("\n" + "="*80)
    print(f"SENSITIVITY ANALYSIS BY {group_name.upper()}")
    print("="*80)

    if metrics_df.empty:
        print(f"No data was available to analyze for {group_name}.")
        return

    print(f"\nAnalyzing {len(metrics_df)} different {group_name} groups:")
    print("-" * 80)

    # Print detailed results for each group
    for _, row in metrics_df.iterrows():
        print(f"\nGroup: {row[0]}") # Assumes first column is the group
        print(f"  Total Cases: {int(row['Total_Cases'])}")
        print(f"  Correct Diagnoses (TP): {int(row['Correct_Diagnoses (TP)'])}")
        print(f"  Missed Diagnoses (FN): {int(row['Missed_Diagnoses (FN)'])}")
        print(f"  Sensitivity: {row['Sensitivity_%']:.1f}%")

    print("\n" + "-" * 80)
    
    # Summary statistics
    total_cases = metrics_df['Total_Cases'].sum()
    total_correct = metrics_df['Correct_Diagnoses (TP)'].sum()
    overall_sensitivity = (total_correct / total_cases * 100) if total_cases > 0 else 0

    print("\nSUMMARY STATISTICS:")
    print(f"Overall Sensitivity (All Cases): {overall_sensitivity:.1f}%")
    print(f"Total Cases Analyzed: {total_cases}")
    
    # Best and worst performing groups
    best_group = metrics_df.loc[metrics_df['Sensitivity_%'].idxmax()]
    worst_group = metrics_df.loc[metrics_df['Sensitivity_%'].idxmin()]
    
    print(f"\nBest Performing Group: {best_group[0]} ({best_group['Sensitivity_%']:.1f}%)")
    print(f"Worst Performing Group: {worst_group[0]} ({worst_group['Sensitivity_%']:.1f}%)")
    print("\n" + "="*80)


# Main execution block
if __name__ == "__main__":
    try:
        # 1. Load and process the log file
        log_filepath = os.path.join(LOG_PATH, LOG_FILE)
        processed_df = load_and_process_log(log_filepath)
        
        # 2. Extract and structure demographic features
        final_df = extract_demographic_features(processed_df)
        
        # 3. Calculate and display sensitivity for each demographic
        demographics_to_analyze = {
            "Age Group": "age_group",
            "Gender": "gender",
            "Smoking Status": "smoking_status"
        }

        for name, col in demographics_to_analyze.items():
            metrics_df = calculate_sensitivity_by_demographic(final_df, col)
            display_sensitivity_report(metrics_df, name)

    except Exception as e:
        print(f"\nAn error occurred during processing: {e}")