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
        patient_df = pd.read_csv(patient_csv_path, encoding='utf-8')
        with open(json_log_path, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Could not find a file at the path provided: {e}")
        print("Please ensure the file paths in the configuration section are correct.")
        return

    # --- Data Preparation ---
    print("Preparing and merging data...")
    results_df = pd.json_normalize(results_data.get('results', []))

    # Define all the columns we need, including the top-N diagnosis lists
    columns_to_keep = {
        'scenario_id': 'scenario_id',
        'is_correct': 'is_correct',
        'correct_diagnosis': 'correct_diagnosis',
        'top_1_confidence': 'confidence_score',
        'top_3_diagnoses': 'top_3_diagnoses',
        'top_5_diagnoses': 'top_5_diagnoses',
        'top_7_diagnoses': 'top_7_diagnoses',
        'consultation_analysis.diagnoses_considered': 'diagnoses_considered'
    }
    
    existing_cols = [col for col in columns_to_keep.keys() if col in results_df.columns]
    results_df_clean = results_df[existing_cols].rename(columns=columns_to_keep)
    
    merged_df = pd.merge(patient_df, results_df_clean, on='scenario_id')

    # Ensure list-based columns are always lists to prevent errors
    for col in ['diagnoses_considered', 'top_3_diagnoses', 'top_5_diagnoses', 'top_7_diagnoses']:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].apply(lambda x: x if isinstance(x, list) else [])

    # --- Metric Calculation Function ---
    def calculate_metrics_for_group(group_df):
        total_count = len(group_df)
        if total_count == 0: return None

        # Top-1 Accuracy is the simple mean of the boolean 'is_correct' column.
        top_1_accuracy = group_df['is_correct'].mean() if 'is_correct' in group_df.columns else 0.0

        def normalize_text(text):
            return str(text).lower().strip()

        # Helper function for Top-N list-based accuracy
        def calculate_top_n_accuracy(n_df, list_col_name):
            if list_col_name not in n_df.columns or 'correct_diagnosis' not in n_df.columns:
                return 0.0
            
            def is_correct_in_list(row):
                correct_diag_norm = normalize_text(row['correct_diagnosis'])
                considered_diags_norm = [normalize_text(d) for d in row[list_col_name]]
                return correct_diag_norm in considered_diags_norm

            return n_df.apply(is_correct_in_list, axis=1).sum() / total_count
        
        # Calculate Top-3, 5, and 7 accuracy using the helper
        top_3_accuracy = calculate_top_n_accuracy(group_df, 'top_3_diagnoses')
        top_5_accuracy = calculate_top_n_accuracy(group_df, 'top_5_diagnoses')
        top_7_accuracy = calculate_top_n_accuracy(group_df, 'top_7_diagnoses')

        # Recall uses the full 'diagnoses_considered' list
        recall = calculate_top_n_accuracy(group_df, 'diagnoses_considered')
        
        avg_diagnoses_count = group_df['diagnoses_considered'].apply(len).mean() if 'diagnoses_considered' in group_df.columns else 0.0

        # Confidence metrics
        avg_confidence, avg_top_1_correct_confidence = 'N/A', 'N/A'
        if 'confidence_score' in group_df.columns:
            avg_confidence = pd.to_numeric(group_df['confidence_score'], errors='coerce').mean()
            correct_cases_df = group_df[group_df['is_correct'] == True]
            if not correct_cases_df.empty:
                avg_top_1_correct_confidence = pd.to_numeric(correct_cases_df['confidence_score'], errors='coerce').mean()
            else:
                avg_top_1_correct_confidence = 0.0

        return {
            'Top-1 Accuracy': top_1_accuracy,
            'Top-3 Accuracy': top_3_accuracy,
            'Top-5 Accuracy': top_5_accuracy,
            'Top-7 Accuracy': top_7_accuracy,
            'Recall': recall,
            'Avg Diagnoses Considered': avg_diagnoses_count,
            'Avg Confidence (All Cases)': avg_confidence,
            'Avg Top-1 Correct Confidence': avg_top_1_correct_confidence,
            'Count': total_count
        }

    # --- Grouping and CSV Generation ---
    categories = [col for col in patient_df.columns if col != 'scenario_id']
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nProcessing categories and generating reports...")
    for category in categories:
        grouped = merged_df.groupby(category)
        category_metrics = []
        for group_name, group_df in grouped:
            metrics = calculate_metrics_for_group(group_df)
            if metrics:
                metrics['Group'] = group_name
                category_metrics.append(metrics)
        
        if not category_metrics: continue

        metrics_df = pd.DataFrame(category_metrics)
        # Define the final column order for the CSV report
        cols = ['Group', 'Count', 'Top-1 Accuracy', 'Top-3 Accuracy', 'Top-5 Accuracy', 'Top-7 Accuracy', 'Recall', 
                'Avg Diagnoses Considered', 'Avg Confidence (All Cases)', 'Avg Top-1 Correct Confidence']
        metrics_df = metrics_df[cols]

        output_filename = os.path.join(output_dir, f"{category.replace(' ', '_')}_metrics.csv")
        metrics_df.to_csv(output_filename, index=False, float_format='%.3f')
        print(f"- Generated: {output_filename}")

    print("\n--- Processing Complete ---")

if __name__ == '__main__':
    # --- Configuration ---
    # !!! ACTION REQUIRED: Replace these placeholders with your actual file paths !!!
    
    YOUR_PATIENT_CSV_PATH = 'categorized_patients.csv'
    YOUR_JSON_LOG_PATH = 'base_files/logs/medqa_run_20250803_001542.json'
    
    OUTPUT_FOLDER = 'demographic_metrics'
    
    analyze_demographic_metrics(YOUR_PATIENT_CSV_PATH, YOUR_JSON_LOG_PATH, OUTPUT_FOLDER)