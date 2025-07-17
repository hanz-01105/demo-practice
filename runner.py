import os
import json
from datetime import datetime
from base_files.prototype.demo import run_single_scenario
from base_files.prototype.scenario import ScenarioMedQA
from base_files.prototype.categorizer import categorize_scenario
#from venv.statistical_analysis import run_analysis

# Ensure logs directory exists
os.makedirs("base_files/logs", exist_ok=True)

# Load scenarios
with open("base_files/data/agentclinic_medqa_extended.jsonl", "r") as f:
    scenarios = [ScenarioMedQA(json.loads(line)) for line in f]

# Optional: Limit for debugging
#scenarios = scenarios[:5]

# Parameters
DATASET = "MedQA"
TOTAL_INFERENCES = 10
MAX_CONSULTATION_TURNS = 5

all_results = []

# Run and categorize scenarios
for idx, scenario in enumerate(scenarios):
    result, is_correct = run_single_scenario(
        scenario=scenario,
        dataset=DATASET,
        total_inferences=TOTAL_INFERENCES,
        max_consultation_turns=MAX_CONSULTATION_TURNS,
        scenario_idx=idx,
        bias=None
    )

    if "dialogue_history" in result:
        del result["dialogue_history"]

    # Get patient info and demographics
    patient_info = scenario.patient_information()
    demographics = categorize_scenario(patient_info)

    # Print for debugging
    print(f"\n[Scenario {idx}]")
    print("Patient Info:", patient_info)
    print("Extracted Demographics:", demographics)

    result["demographics"] = demographics
    result["is_correct"] = is_correct  # Ensure this is included

    all_results.append(result)

# Save with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"base_files/logs/agentclinic_run_{timestamp}.json"

metadata = {
    "run_timestamp": timestamp,
    "num_scenarios": len(all_results),
    "logs": all_results
}

with open(output_path, "w") as f:
    json.dump(metadata, f, indent=2, default=str)

# Also save as latest for downstream analysis
latest_output_path = f"{os.path.dirname(output_path)}/agentclinic_run_latest.json"
with open(latest_output_path, "w") as f:
    json.dump(metadata, f, indent=2, default=str)

print(f"\nSaved {len(all_results)} scenario logs to {output_path}")

# Run statistical analysis on latest results
#run_analysis("agentclinic_run_latest.json")
