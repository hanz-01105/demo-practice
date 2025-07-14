import os
import json
from datetime import datetime
from base_files.prototype.demo import run_single_scenario
from base_files.prototype.scenario import ScenarioMedQA


os.makedirs("base_files/logs", exist_ok=True)


with open("base_files/data/agentclinic_medqa_extended.jsonl", "r") as f:
    scenarios = [ScenarioMedQA(json.loads(line)) for line in f]

# For debugging
scenarios = scenarios[:2]

DATASET = "MedQA"
TOTAL_INFERENCES = 10
MAX_CONSULTATION_TURNS = 5

all_results = []

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
    all_results.append(result)

# Save logs with timestamped filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"base_files/logs/agentclinic_run_{timestamp}.json"

metadata = {
    "run_timestamp": timestamp,
    "num_scenarios": len(all_results),
    "logs": all_results
}

with open(output_path, "w") as f:
    json.dump(metadata, f, indent=2, default=str)

print(f"\n✅ Saved {len(all_results)} scenario logs to {output_path}")


