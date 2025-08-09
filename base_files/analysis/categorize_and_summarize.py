import json
import os
import pandas as pd
from collections import defaultdict
from categorizer import categorize_patient
import time
import openai

def parse_confidence(conf):
    """Parse confidence values, handling lists or single values."""
    if isinstance(conf, list):
        return [float(c) for c in conf if c is not None]
    elif conf is None:
        return []
    else:
        try:
            return [float(str(conf).replace("%", "").strip())]
        except ValueError:
            return []

def calculate_metrics(df):
    """Calculate performance metrics for a bucket subgroup."""
    total_cases = len(df)
    if total_cases == 0:
        return {
            "total_cases": 0,
            "accuracy": None,
            "recall": None,
            "avg_num_diagnoses": None,
            "avg_confidence": None,
            "top1_accuracy": None,
            "top3_accuracy": None,
            "top5_accuracy": None,
            "top7_accuracy": None,
            "top1_confidence": None,
            "top3_confidence": None,
            "top5_confidence": None,
            "top7_confidence": None
        }

    # Parse confidence values
    df["parsed_confidence"] = df["self_confidence"].apply(parse_confidence)

    # Basic metrics
    accuracy = df["is_correct"].mean()

    # Recall: correct diagnosis present in diagnoses_considered
    df["recall_hit"] = df.apply(
        lambda row: row["correct_diagnosis"] in (row.get("consultation_analysis.diagnoses_considered") or []),
        axis=1
    )
    recall = df["recall_hit"].mean()

    # Avg diagnoses considered
    avg_num_diagnoses = df["consultation_analysis.diagnoses_considered"].apply(lambda d: len(d) if isinstance(d, list) else 0).mean()

    # Confidence rating (average of all confidence values flattened)
    all_conf_values = [c for conf_list in df["parsed_confidence"] for c in conf_list]
    avg_confidence = sum(all_conf_values) / len(all_conf_values) if all_conf_values else None

    # Top-K accuracy & confidence
    def top_k_accuracy(row, k):
        if isinstance(row.get("final_doctor_diagnosis"), list):
            return row["correct_diagnosis"] in row["final_doctor_diagnosis"][:k]
        return False

    def top_k_confidence(row, k):
        conf = row.get("parsed_confidence", [])
        return sum(conf[:k]) if conf else 0

    metrics = {}
    for k in [1, 3, 5, 7]:
        metrics[f"top{k}_accuracy"] = df.apply(lambda r: top_k_accuracy(r, k), axis=1).mean()
        metrics[f"top{k}_confidence"] = df.apply(lambda r: top_k_confidence(r, k), axis=1).mean()

    return {
        "total_cases": total_cases,
        "accuracy": accuracy,
        "recall": recall,
        "avg_num_diagnoses": avg_num_diagnoses,
        "avg_confidence": avg_confidence,
        **metrics
    }


def categorize_and_summarize(log_path):
    """Categorize scenarios and calculate metrics for each demographic category."""
    with open(log_path, "r") as f:
        data = json.load(f)

    results = data.get("results", [])
    print(f"Loaded {len(results)} results from {log_path}")

    bucket_data = defaultdict(list)

    # Bucket assignment with indefinite retries for API calls
    for i, result in enumerate(results):
        print(f"Processing result {i + 1}/{len(results)}...")
        demo = result.get("demographics", {})
        social = result.get("social_history", "")
        pmh = result.get("past_medical_history", "")
        history = result.get("history", "")

        demo_dict = {
            "Demographics": demo,
            "Social_History": social,
            "Past_Medical_History": pmh,
            "History": history
        }

        # --- START: Indefinite Retry Logic ---
        attempt = 0
        base_delay = 1  # seconds
        
        while True: # This loop will run until it succeeds
            try:
                categories = categorize_patient(demo_dict)
                for category_name, category_value in categories.items():
                    bucket_data[(category_name, category_value)].append(result)
                break  # Success! Exit the 'while' loop and move to the next result.

            except openai.RateLimitError:
                # On rate limit, wait and try again. The wait time increases exponentially.
                # We'll cap the max wait time at 60 seconds to be reasonable.
                wait_time = min(base_delay * (2 ** attempt), 60)
                print(f"⏳ Rate limit hit. Pausing for {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                attempt += 1

            except Exception as e:
                # For any other error, we stop the script to let you investigate.
                print(f"🛑 A non-rate-limit error occurred on item {i+1}: {e}")
                print("Stopping script to prevent data loss or infinite loops. Please check the error.")
                raise # Re-raise the exception to halt execution and show the full error.
        # --- END: Indefinite Retry Logic ---


    # Metrics calculation grouped per demographic category
    category_metrics = defaultdict(list)
    for (category_name, bucket_label), items in bucket_data.items():
        df = pd.json_normalize(items)
        metrics = calculate_metrics(df)
        metrics["bucket"] = bucket_label
        category_metrics[category_name].append(metrics)

    # Save one CSV per demographic category
    os.makedirs("base_files/analysis/buckets", exist_ok=True)
    for category_name, metrics_list in category_metrics.items():
        metrics_df = pd.DataFrame(metrics_list)
        sanitized_category_name = category_name.replace(' ', '_')
        csv_name = f"base_files/analysis/buckets/{sanitized_category_name}_metrics.csv"
        metrics_df.to_csv(csv_name, index=False)
        print(f"Metrics for {category_name} saved to {csv_name}")


if __name__ == "__main__":
    log_filename = "base_files/logs/medqa_run_20250803_001542.json"
    categorize_and_summarize(log_filename)