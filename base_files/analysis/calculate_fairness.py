import pandas as pd
import os
import glob

def enhance_metric_reports(metrics_folder):
    """
    Reads existing metric CSVs and adds detailed and summary fairness
    metrics directly as columns to each file.
    """
    # Find all the metric CSVs in the specified folder
    csv_files = glob.glob(os.path.join(metrics_folder, '*_metrics.csv'))

    if not csv_files:
        print(f"Error: No metric CSVs found in the '{metrics_folder}' directory.")
        return

    print(f"--- Found {len(csv_files)} metric files. Enhancing now... ---")


    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            if df.empty or 'Count' not in df.columns or 'Top-1 Accuracy' not in df.columns:
                print(f"- Skipping '{os.path.basename(file_path)}': missing required columns.")
                continue

            # --- Calculate Category-Wide Summary Metrics ---
            accuracy_scores = df['Top-1 Accuracy']
            performance_volatility = accuracy_scores.std()
            worst_group_disadvantage = accuracy_scores.max() - accuracy_scores.min()

            # --- Add All New Metrics as Columns ---
            
            # Add summary metrics (value is repeated for each row)
            df['Performance Volatility (Std Dev)'] = performance_volatility
            df['Worst-Group Disadvantage (Max-Min Gap)'] = worst_group_disadvantage

            # Add row-specific metrics
            baseline_group = df.loc[df['Count'].idxmax()]
            baseline_accuracy = baseline_group['Top-1 Accuracy']
            df['Parity Gap (vs Baseline)'] = df['Top-1 Accuracy'] - baseline_accuracy

            if 'Avg Confidence (All Cases)' in df.columns:
                df['Avg Confidence (All Cases)'] = pd.to_numeric(df['Avg Confidence (All Cases)'], errors='coerce')
                df['Calibration Error'] = df['Avg Confidence (All Cases)'] - df['Top-1 Accuracy']
            
            # Save the enhanced file, overwriting the original
            df.to_csv(file_path, index=False, float_format='%.3f')
            print(f"- Added all fairness columns to: {os.path.basename(file_path)}")

        except Exception as e:
            print(f"  Could not process {os.path.basename(file_path)}: {e}")

    print("\n--- All CSV files have been updated successfully. ---")


if __name__ == '__main__':
    enhance_metric_reports('demographic_metrics')