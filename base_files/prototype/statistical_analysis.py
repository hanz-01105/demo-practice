import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import chi2_contingency

LOG_PATH = "base_files/logs"

def run_analysis(log_file="agentclinic_run_latest.json"):
    log_filepath = os.path.join(LOG_PATH, log_file)

    if not os.path.exists(log_filepath):
        raise FileNotFoundError(f"Log file not found: {log_filepath}")

    # Load log data
    with open(log_filepath, "r") as f:
        data = json.load(f)

    logs = data["logs"]
    df = pd.json_normalize(logs)
    print("\nLoaded log file with shape:", df.shape)

    # Drop rows with missing demographics
    df = df[df["demographics"].notnull()].copy()

    # Expand demographics columns
    demo_df = pd.json_normalize(df["demographics"])
    print("\nDemographic Columns Present:", demo_df.columns.tolist())

    # Fill missing expected keys with 'Unknown'
    expected_columns = ["gender", "age_group", "comorbidity_status"]
    for col in expected_columns:
        if col not in demo_df.columns:
            demo_df[col] = "Unknown"

    df = pd.concat([df.reset_index(drop=True), demo_df.add_prefix("demo_").reset_index(drop=True)], axis=1)

    # Basic accuracy
    if "is_correct" not in df.columns:
        raise KeyError("Column 'is_correct' not found in log data.")
    df["is_correct"] = df["is_correct"].astype(bool)

    # Accuracy by gender
    gender_acc = df.groupby("demo_gender")["is_correct"].mean().reset_index()
    print("\nAccuracy by Gender:")
    print(gender_acc)

    # Accuracy by age group
    age_acc = df.groupby("demo_age_group")["is_correct"].mean().reset_index()
    print("\nAccuracy by Age Group:")
    print(age_acc)

    # Accuracy by comorbidity
    comorbidity_acc = df.groupby("demo_comorbidity_status")["is_correct"].mean().reset_index()
    print("\nAccuracy by Comorbidity Status:")
    print(comorbidity_acc)

    # Chi-square test: Gender fairness
    contingency = pd.crosstab(df["demo_gender"], df["is_correct"])
    chi2, p, dof, expected = chi2_contingency(contingency)
    print("\nChi-Square Test for Gender Fairness:")
    print("Chi2 Statistic:", chi2)
    print("p-value:", p)

    # Visualization
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.barplot(x="demo_gender", y="is_correct", data=df)
    plt.title("Accuracy by Gender")
    plt.ylabel("Accuracy")
    plt.xlabel("Gender")
    plt.ylim(0, 1)
    plt.tight_layout()
    os.makedirs("base_files/analysis", exist_ok=True)
    plt.savefig("base_files/analysis/accuracy_by_gender.png")
    print("\nSaved plot: accuracy_by_gender.png")

    # Save DataFrame to CSV
    df.to_csv("base_files/analysis/merged_analysis_data.csv", index=False)
    print("\nSaved merged data to: merged_analysis_data.csv")
