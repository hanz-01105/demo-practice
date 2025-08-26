import pandas as pd
import json
import os
import openai
from openai import OpenAI
client = OpenAI()
openai.api_key = os.getenv("OPENAI_API_KEY")

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
        return

    # --- Data Preparation ---
    print("Preparing and merging data...")
    results_df = pd.json_normalize(results_data.get('results', []))

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

    existing_cols = [col for col in columns_to_keep if col in results_df.columns]
    results_df_clean = results_df[existing_cols].rename(columns=columns_to_keep)

    merged_df = pd.merge(patient_df, results_df_clean, on='scenario_id')

    for col in ['diagnoses_considered', 'top_3_diagnoses', 'top_5_diagnoses', 'top_7_diagnoses']:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].apply(lambda x: x if isinstance(x, list) else [])

    # --- LLM helper ---
    def is_semantically_correct(correct_diagnosis, predicted_list):
        try:
            prompt = (
                f"Correct diagnosis: {correct_diagnosis}\n"
                f"Predicted diagnoses: {predicted_list}\n\n"
                f"Is the correct diagnosis semantically present among the predicted diagnoses? "
                f"Only respond 'Yes' or 'No'."
            )

            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )

            answer = response.choices[0].message.content.strip().lower()
            return answer.startswith("yes")

        except Exception as e:
            print(f"LLM error: {e}")
            return False
    # --- Metric Calculation Function ---
    def calculate_metrics_for_group(group_df):
        total_count = len(group_df)
        if total_count == 0: return None

        top_1_accuracy = group_df['is_correct'].mean() if 'is_correct' in group_df.columns else 0.0

        # LLM-based Top-N Accuracy
        def calculate_top_n_accuracy(n_df, list_col_name):
            correct_count = 0
            for _, row in n_df.iterrows():
                correct = row.get('correct_diagnosis', '')
                preds = row.get(list_col_name, [])
                if correct and isinstance(preds, list):
                    correct_count += int(is_semantically_correct(correct, preds))
            return correct_count / total_count

        top_3_accuracy = calculate_top_n_accuracy(group_df, 'top_3_diagnoses')
        top_5_accuracy = calculate_top_n_accuracy(group_df, 'top_5_diagnoses')
        top_7_accuracy = calculate_top_n_accuracy(group_df, 'top_7_diagnoses')
        recall = calculate_top_n_accuracy(group_df, 'diagnoses_considered')

        avg_diagnoses_count = group_df['diagnoses_considered'].apply(len).mean() if 'diagnoses_considered' in group_df.columns else 0.0

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
        cols = ['Group', 'Count', 'Top-1 Accuracy', 'Top-3 Accuracy', 'Top-5 Accuracy', 'Top-7 Accuracy', 'Recall',
                'Avg Diagnoses Considered', 'Avg Confidence (All Cases)', 'Avg Top-1 Correct Confidence']
        metrics_df = metrics_df[cols]

        output_filename = os.path.join(output_dir, f"{category.replace(' ', '_')}_metrics.csv")
        metrics_df.to_csv(output_filename, index=False, float_format='%.3f')
        print(f"- Generated: {output_filename}")

    print("\n--- Processing Complete ---")

if __name__ == '__main__':
    YOUR_PATIENT_CSV_PATH = 'categorized_patients.csv'
    YOUR_JSON_LOG_PATH = 'base_files/logs/medqa_run_20250803_001542.json'
    OUTPUT_FOLDER = 'demographic_metrics'

    analyze_demographic_metrics(YOUR_PATIENT_CSV_PATH, YOUR_JSON_LOG_PATH, OUTPUT_FOLDER)
