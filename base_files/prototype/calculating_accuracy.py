import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import ols

LOG_PATH = "base_files/prototype/logs"

def run_analysis(log_file="base_files/prototype/logs/MedQA_Ext_none_bias_corrected_20250725_114108.json"):
    log_filepath = log_file  # Use the full path directly since you're providing absolute path

    if not os.path.exists(log_filepath):
        raise FileNotFoundError(f"Log file not found: {log_filepath}")

    with open(log_filepath, "r") as f:
        data = json.load(f)

    # Handle both old and new log file formats
    if isinstance(data, dict) and "logs" in data:
        # Old format: {"logs": [...], "run_timestamp": "..."}
        logs = data["logs"]
        print(f"Old format detected. Run timestamp: {data.get('run_timestamp', 'Unknown')}")
    elif isinstance(data, list):
        # New format: [...]
        logs = data
        print("New format detected (direct list of scenarios)")
    else:
        raise ValueError(f"Unexpected log file format")

    df = pd.json_normalize(logs)
    print("\nLoaded log file with shape:", df.shape)

    # Check if demographics column exists
    if "demographics" in df.columns and df["demographics"].notnull().sum() > 0:
        df = df[df["demographics"].notnull()].copy()
        demo_df = pd.json_normalize(df["demographics"])
        df = pd.concat([df.reset_index(drop=True), demo_df.add_prefix("demo_").reset_index(drop=True)], axis=1)

        def parse_demographics(demo_str):
            if isinstance(demo_str, str):
                try:
                    demo_dict = {}
                    for line in demo_str.strip().split("\n"):
                        if line.strip() and "dtype" not in line:
                            key, value = line.split(maxsplit=1)
                            demo_dict[key.strip()] = value.strip()
                    return demo_dict
                except:
                    return {}
            return {}

        df_demo_parsed = df["demographics"].apply(parse_demographics).apply(pd.Series)
        df_demo_parsed.columns = [f"demo_{col}" for col in df_demo_parsed.columns]
        df = pd.concat([df, df_demo_parsed], axis=1)
    else:
        print("Warning: No demographics data found - skipping demographic analysis")

    df["self_confidence"] = pd.to_numeric(df["self_confidence"].str.replace("%", ""), errors="coerce") / 100.0
    
    # Handle consultation analysis columns safely
    if "consultation_analysis.diagnoses_considered_count" in df.columns:
        df["num_diagnoses_considered"] = df["consultation_analysis.diagnoses_considered_count"]
    else:
        print("Warning: consultation_analysis.diagnoses_considered_count not found")
        df["num_diagnoses_considered"] = 0

    df["num_dialogue_turns"] = pd.to_numeric(df["num_dialogue_turns"], errors="coerce")
    df["is_correct"] = df["is_correct"].astype(bool)
    df["correct_diagnosis_considered"] = df["is_correct"]

    # Calculate and display core metrics
    print(f"\n=== CORE PERFORMANCE METRICS ===")
    accuracy = df["is_correct"].mean()
    total_cases = len(df)
    correct_cases = df["is_correct"].sum()
    
    print(f"Total cases: {total_cases}")
    print(f"Correct diagnoses: {correct_cases}")
    print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Confidence analysis
    avg_confidence = df["self_confidence"].mean()
    print(f"Average confidence: {avg_confidence:.3f} ({avg_confidence*100:.1f}%)")
    
    # Test utilization analysis
    if "tests_requested_count" in df.columns:
        avg_tests = df["tests_requested_count"].mean()
        print(f"Average tests requested: {avg_tests:.2f}")
    
    # Dialogue efficiency
    avg_dialogue_turns = df["num_dialogue_turns"].mean()
    print(f"Average dialogue turns: {avg_dialogue_turns:.1f}")

    def demographic_summary(df, group_col):
        if group_col not in df.columns:
            print(f"Warning: Column {group_col} not found - skipping")
            return pd.DataFrame()
        
        grouped = df.groupby(group_col).agg(
            count=("is_correct", "count"),
            accuracy=("is_correct", "mean"),
            num_diagnoses_considered=("num_diagnoses_considered", "mean"),
            num_dialogue_turns=("num_dialogue_turns", "mean"),
            correct_diagnosis_considered=("correct_diagnosis_considered", "sum"),
            confidence_rating=("self_confidence", "mean")
        ).reset_index()
        baseline = grouped["accuracy"].mean()
        grouped["diff_from_baseline"] = grouped["accuracy"] - baseline
        best_group_acc = grouped["accuracy"].max()
        grouped["demographic_parity"] = best_group_acc - grouped["accuracy"]
        return grouped

    os.makedirs("base_files/analysis", exist_ok=True)

    # Only analyze demographics if they exist
    demographic_cols = [col for col in df.columns if col.startswith("demo_")]
    if demographic_cols:
        for category in [
            "demo_gender",
            "demo_age_group",
            "demo_smoking_status",
            "demo_alcohol_use",
            "demo_symptom_presentation"
        ]:
            if category in df.columns:
                print(f"\n--- Summary for {category.replace('demo_', '').title()} ---")
                summary = demographic_summary(df, category)
                if not summary.empty:
                    print(summary)
                    summary.to_csv(f"base_files/analysis/{category}_summary.csv", index=False)
                    print(f"Saved: {category}_summary.csv")
    else:
        print("No demographic columns found - skipping demographic analysis")

    # Save final merged df
    df.to_csv("base_files/analysis/merged_analysis_data.csv", index=False)
    print("\nMerged DataFrame saved to: merged_analysis_data.csv")
    
    return df

if __name__ == "__main__":
    run_analysis(log_file="base_files/prototype/logs/MedQA_Ext_none_bias_corrected_20250725_114108.json"
)