import json
import os
import pandas as pd
from collections import defaultdict
from base_files.analysis.categorizer import categorize_demographics


def load_log_file(log_path):
    with open(log_path, "r") as f:
        data = json.load(f)
    return data.get("results", [])


def categorize_and_summarize(log_path):
    results = load_log_file(log_path)
    print(f"Loaded {len(results)} results from {log_path}")

    bucket_data = defaultdict(list)

    for result in results:
        demographics = result.get("demographics", {})
        categories = categorize_demographics(demographics)

        for category_name, category_value in categories.items():
            bucket_data[(category_name, category_value)].append(result)

    os.makedirs("base_files/analysis/buckets", exist_ok=True)

    for (category_name, bucket_label), items in bucket_data.items():
        df = pd.json_normalize(items)
        csv_name = f"base_files/analysis/buckets/{category_name}_{bucket_label}.csv"
        df.to_csv(csv_name, index=False)
        print(f"Saved {len(df)} rows to {csv_name}")


if __name__ == "__main__":
    log_filename = "base_files/logs/medqa_run_20250724_141812.json"
    categorize_and_summarize(log_filename)
