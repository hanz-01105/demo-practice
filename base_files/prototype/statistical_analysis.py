
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import ols

LOG_PATH = "base_files/logs"

def run_analysis(log_file="agentclinic_run_latest.json"):
    log_filepath = os.path.join(LOG_PATH, log_file)

    if not os.path.exists(log_filepath):
        raise FileNotFoundError(f"Log file not found: {log_filepath}")

    with open(log_filepath, "r") as f:
        data = json.load(f)

    logs = data["logs"]
    df = pd.json_normalize(logs)
    print("\nLoaded log file with shape:", df.shape)

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

    df["self_confidence"] = pd.to_numeric(df["self_confidence"].str.replace("%", ""), errors="coerce") / 100.0
    df["num_diagnoses_considered"] = df["consultation_analysis.diagnoses_considered_count"]
    df["num_dialogue_turns"] = pd.to_numeric(df["num_dialogue_turns"], errors="coerce")
    df["is_correct"] = df["is_correct"].astype(bool)
    df["correct_diagnosis_considered"] = df["is_correct"]

    def demographic_summary(df, group_col):
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

    for category in [
        "demo_gender",
        "demo_age_group",
        "demo_smoking_status",
        "demo_alcohol_use",
        "demo_symptom_presentation"
    ]:
        print(f"\n--- Summary for {category.replace('demo_', '').title()} ---")
        summary = demographic_summary(df, category)
        print(summary)
        summary.to_csv(f"base_files/analysis/{category}_summary.csv", index=False)
        print(f"Saved: {category}_summary.csv")

    # Save final merged df
    df.to_csv("base_files/analysis/merged_analysis_data.csv", index=False)
    print("\nMerged DataFrame saved to: merged_analysis_data.csv")

if __name__ == "__main__":
    run_analysis(log_file="original_agentclinic_run_latest.json")
