import json
from base_files.prototype.demo import run_single_scenario

with open("base_agents/data/agentclinic_medqa_extended.jsonl", "r") as f:
    scenarios = json.load(f)

DATASET = "MedQA"
TOTAL_INFERENCES = 10
MAX_CONSULTATION_TURNS = 5

all_results = []

for idx, scenario in enumerate(scenarios):
    result = run_single_scenario(
        scenario=scenario,
        dataset=DATASET,
        total_inferences=TOTAL_INFERENCES,
        max_consultation_turns=MAX_CONSULTATION_TURNS,
        scenario_idx=idx,
        bias=None
    )
    all_results.append(result)

# Save logs
with open("base_agents/data/all_scenarios_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"Done! Saved {len(all_results)} scenario logs.")