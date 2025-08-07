import json
import os
import glob
import pandas as pd
from datetime import datetime

# Load Logs
log_files = glob.glob("base_files/logs/MedQA_Ext_none_bias_corrected_20250804_211039.json")
print(f"Found {len(log_files)} log files in base_files/logs/")

all_logs = []
for file in log_files:
    with open(file) as f:
        data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            # If the file contains a direct list of logs
            print(f"File {file} contains a direct list of {len(data)} logs")
            for log in data:
                # Add a timestamp if it doesn't exist
                if "run_timestamp" not in log:
                    # Extract timestamp from filename or use current time
                    filename_timestamp = file.split('_')[-1].replace('.json', '')
                    log["run_timestamp"] = filename_timestamp
                all_logs.append(log)
                
        elif isinstance(data, dict):
            if "logs" in data:
                # If the file has the expected structure with "logs" key
                print(f"File {file} contains {len(data['logs'])} logs in structured format")
                for log in data["logs"]:
                    log["run_timestamp"] = data.get("run_timestamp", "unknown")
                    all_logs.append(log)
            else:
                # If the file contains a single log object
                print(f"File {file} contains a single log object")
                if "run_timestamp" not in data:
                    filename_timestamp = file.split('_')[-1].replace('.json', '')
                    data["run_timestamp"] = filename_timestamp
                all_logs.append(data)
        else:
            print(f"Warning: Unrecognized format in file {file}")

print(f"Loaded {len(all_logs)} scenario logs from all runs.")

# Debug: Show the structure of the first log
if all_logs:
    print("\nFirst log structure:")
    print(f"Keys: {list(all_logs[0].keys())}")
    print(f"Sample data: {json.dumps({k: v for k, v in list(all_logs[0].items())[:3]}, indent=2)}")

# Load MedQA
try:
    with open("base_files/data/agentclinic_medqa_extended.jsonl") as f:
        scenarios = [json.loads(line) for line in f]
    print(f"Loaded {len(scenarios)} MedQA scenarios.")
except FileNotFoundError:
    print("Warning: MedQA file not found. Creating dummy scenarios for testing.")
    scenarios = [{"OSCE_Examination": {"Patient_Actor": f"dummy_actor_{i}"}} for i in range(100)]

# Join Logs to MedQA
successful_joins = 0
for log in all_logs:
    try:
        sid = log.get("scenario_id")
        if sid is not None and sid < len(scenarios):
            scenario = scenarios[sid]
            if "OSCE_Examination" in scenario and "Patient_Actor" in scenario["OSCE_Examination"]:
                actor = scenario["OSCE_Examination"]["Patient_Actor"]
                log["patient_actor"] = actor
                successful_joins += 1
            else:
                print(f"Warning: Scenario {sid} missing expected structure")
                log["patient_actor"] = "unknown"
        else:
            print(f"Warning: Invalid or missing scenario_id: {sid}")
            log["patient_actor"] = "unknown"
    except Exception as e:
        print(f"Error processing log: {e}")
        log["patient_actor"] = "unknown"

print(f"Successfully joined {successful_joins} logs with MedQA scenarios.")

# Create DataFrame for analysis
df = pd.DataFrame(all_logs)
print(f"\nDataFrame created with {len(df)} rows and {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")

# Basic analysis
if len(df) > 0:
    print(f"\nBasic Statistics:")
    print(f"- Unique models: {df['model'].nunique() if 'model' in df.columns else 'N/A'}")
    print(f"- Unique datasets: {df['dataset'].nunique() if 'dataset' in df.columns else 'N/A'}")
    print(f"- Correct diagnoses: {df['is_correct'].sum() if 'is_correct' in df.columns else 'N/A'}")
    print(f"- Total scenarios: {len(df)}")
    
    if 'is_correct' in df.columns:
        accuracy = df['is_correct'].mean()
        print(f"- Accuracy: {accuracy:.2%}")
    
    # Show sample of data
    print(f"\nSample data:")
    display_columns = ['scenario_id', 'correct_diagnosis', 'final_doctor_diagnosis', 'is_correct'] 
    available_columns = [col for col in display_columns if col in df.columns]
    if available_columns:
        print(df[available_columns].head())
    else:
        print("Key columns not found, showing first few columns:")
        print(df.iloc[:, :5].head())