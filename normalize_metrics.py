import pandas as pd
import os
import glob

target_folder = 'demographic_metrics'

search_path = os.path.join(target_folder, '*.csv')
csv_files = glob.glob(search_path)


if not csv_files:
    print(f"No CSV files found in the '{target_folder}' directory. Please check the folder name and path.")
else:
    print(f"Found {len(csv_files)} CSV files to process in-place...")

for file_path in csv_files:
    try:

        df = pd.read_csv(file_path)

        # Convert confidence columns from percent (0-100) to a 0-1 scale
        df['Avg Confidence (All Cases)'] = df['Avg Confidence (All Cases)'] / 100
        df['Avg Top-1 Correct Confidence'] = df['Avg Top-1 Correct Confidence'] / 100

        # Recalculate the 'Calibration Error'
        df['Calibration Error'] = df['Avg Confidence (All Cases)'] - df['Top-1 Accuracy']

        df.to_csv(file_path, index=False)

        print(f"✅ Processed and overwrote '{file_path}'")

    except KeyError as e:
        print(f"❌ Could not process file '{file_path}'. A required column is missing: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred with file '{file_path}': {e}")

print("\nBatch processing complete!")