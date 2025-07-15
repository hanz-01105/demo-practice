import json
import os
import glob
import pandas as pd

#Load Logs
log_files = glob.glob("base_files/logs/agentclinic_run_20250714_123157.json")
print(f"Found {len(log_files)} log files in base_files/logs/")

all_logs = []
for file in log_files:
    with open(file) as f:
        run = json.load(f)
        for log in run["logs"]:
            log["run_timestamp"] = run["run_timestamp"]
            all_logs.append(log)

print(f"Loaded {len(all_logs)} scenario logs from all runs.")

#Load MedQA
with open("base_files/data/agentclinic_medqa_extended.jsonl") as f:
    scenarios = [json.loads(line) for line in f]

print(f"Loaded {len(scenarios)} MedQA scenarios.")

#Join Logs to MedQA
for log in all_logs:
    sid = log["scenario_id"]
    scenario = scenarios[sid]
    actor = scenario["OSCE_Examination"]["Patient_Actor"]

