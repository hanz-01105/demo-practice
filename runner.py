import json
import os
from datetime import datetime
from base_files.prototype.demo import run_single_scenario
from base_files.prototype.scenario import ScenarioMedQA, BaseScenario

# Load scenarios
with open("base_files/data/agentclinic_medqa_extended.jsonl", "r") as f:
    scenarios = [ScenarioMedQA(json.loads(line)) for line in f]
#scenarios = scenarios[13:14]  # limit to first 5 for testing

# Config
DATASET = "MedQA_Ext"
TOTAL_INFERENCES = 10
MAX_CONSULTATION_TURNS = 5

all_results = []

for idx, scenario in enumerate(scenarios):
    print(f"scenario {idx}")
    result, is_correct = run_single_scenario(
        scenario=scenario,
        dataset=DATASET,
        total_inferences=TOTAL_INFERENCES,
        max_consultation_turns=MAX_CONSULTATION_TURNS,
        scenario_idx=idx,
        bias=None
    )

    # Remove dialogue
    result.pop("dialogue_history", None)

    # Add demographics from that scenario
    patient = scenario.patient_information()
    result["demographics"] = patient

    # Add doctor turns count if available
    if "num_dialogue_turns" not in result:
        result["num_dialogue_turns"] = len([entry for entry in result.get("consultation_analysis", {}).get("turns", [])])

    # Append
    all_results.append(result)

# Save logs under logs/ with timestamp
os.makedirs("base_files/logs", exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
outfile = f"base_files/logs/medqa_run_{ts}.json"
with open(outfile, "w") as f:
    json.dump({
        "run_timestamp": ts,
        "dataset": DATASET,
        "results": all_results
    }, f, indent=2, default=str)

print(f" Logs saved to {outfile}")

