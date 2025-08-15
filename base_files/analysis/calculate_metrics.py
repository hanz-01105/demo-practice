import pandas as pd
import json
import os

def analyze_demographic_metrics(patient_csv_path, json_log_path, output_dir):
    """
    Loads patient and simulation data, calculates performance metrics for various
    demographic groups, and saves the results to CSV files.
    """
    # --- Data Loading ---
    try:
        patient_df = pd.read_csv(patient_csv_path)
        with open(json_log_path, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: Could not find a file at the path '{json_log_path}'.")
        print("Please fix the file path in the configuration section at the bottom of this script.")
        return

    # --- Data Preparation ---
    print("Preparing and merging data...")
    results_df = pd.json_normalize(results_data.get('results', []))

    if 'consultation_analysis.diagnoses_considered' in results_df.columns:
        results_df = results_df.rename(columns={'consultation_analysis.diagnoses_considered': 'diagnoses_considered'})
    else:
        results_df['diagnoses_considered'] = [[] for _ in range(len(results_df))]


    confidence_col_name = 'top_1_confidence' 

    columns_to_keep = {
        'scenario_id': 'scenario_id',
        'correct_diagnosis': 'correct_diagnosis',
        'diagnoses_considered': 'diagnoses_considered',
    }
    if confidence_col_name in results_df.columns:
        columns_to_keep[confidence_col_name] = 'top_1_confidence'

    existing_cols_to_keep = {k: v for k, v in columns_to_keep.items() if k in results_df.columns}
    results_df_clean = results_df[list(existing_cols_to_keep.keys())].rename(columns=existing_cols_to_keep)
    
    merged_df = pd.merge(patient_df, results_df_clean, on='scenario_id')
    merged_df['diagnoses_considered'] = merged_df['diagnoses_considered'].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # --- Metric Calculation Function ---
    def calculate_metrics_for_group(group_df):
        total_count = len(group_df)
        if total_count == 0: return None

        def top_n_accuracy(n):
            return group_df.apply(
                lambda row: row['correct_diagnosis'] in row['diagnoses_considered'][:n], axis=1
            ).sum() / total_count

        recall = group_df.apply(
            lambda row: row['correct_diagnosis'] in row['diagnoses_considered'], axis=1
        ).sum() / total_count
        
        avg_diagnoses_count = group_df['diagnoses_considered'].apply(len).mean()

        # --- NEW METRIC CALCULATION ---
        avg_confidence = 'N/A'
        avg_top_1_correct_confidence = 'N/A'
        
        if 'top_1_confidence' in group_df.columns:
            # Calculate average confidence for all cases in the group
            avg_confidence = pd.to_numeric(group_df['top_1_confidence'], errors='coerce').mean()

            # Filter for rows where the top-1 diagnosis was correct
            top_1_correct_df = group_df[group_df.apply(
                lambda row: len(row['diagnoses_considered']) > 0 and row['correct_diagnosis'] == row['diagnoses_considered'][0], axis=1
            )]

            # Calculate avg confidence for only those top-1 correct cases
            if not top_1_correct_df.empty:
                avg_top_1_correct_confidence = pd.to_numeric(top_1_correct_df['top_1_confidence'], errors='coerce').mean()
            else:
                avg_top_1_correct_confidence = 0.0 # No top-1 correct cases in this group

        return {
            'Top-1 Accuracy': top_n_accuracy(1),
            'Top-3 Accuracy': top_n_accuracy(3),
            'Top-5 Accuracy': top_n_accuracy(5),
            'Recall': recall,
            'Avg Diagnoses Considered': avg_diagnoses_count,
            'Avg Confidence (All Cases)': avg_confidence,
            'Avg Top-1 Correct Confidence': avg_top_1_correct_confidence, # New Metric
            'Count': total_count
        }

    # --- Grouping and CSV Generation ---
    categories = [col for col in patient_df.columns if col != 'scenario_id']
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nProcessing categories and generating reports...")
    for category in categories:
        # ... (The rest of the script is the same)
        grouped = merged_df.groupby(category)
        category_metrics = []
        for group_name, group_df in grouped:
            metrics = calculate_metrics_for_group(group_df)
            if metrics:
                metrics['Group'] = group_name
                category_metrics.append(metrics)
        
        if not category_metrics: continue

        metrics_df = pd.DataFrame(category_metrics)
        cols = ['Group', 'Count'] + [col for col in metrics_df.columns if col not in ['Group', 'Count']]
        metrics_df = metrics_df[cols]

        output_filename = os.path.join(output_dir, f"{category.replace(' ', '_')}_metrics.csv")
        metrics_df.to_csv(output_filename, index=False, float_format='%.3f')
        print(f"- Generated: {output_filename}")

    print("\n--- Processing Complete ---")

if __name__ == '__main__':
    # --- Configuration ---
    PATIENT_DATA_CSV = 'categorized_patients.csv'
    SIMULATION_LOG_JSON = 'base_files/logs/medqa_run_20250803_001542.json'
    OUTPUT_FOLDER = 'demographic_metrics'
    
    analyze_demographic_metrics(PATIENT_DATA_CSV, SIMULATION_LOG_JSON, OUTPUT_FOLDER)