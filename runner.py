import json
from base_files.prototype.demo import run_single_scenario
from base_files.prototype.scenario import ScenarioMedQA

with open("base_files/data/agentclinic_medqa_extended.jsonl", "r") as f:
    scenarios = [ScenarioMedQA(json.loads(line)) for line in f]

#for debugging
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
    all_results.append(result)

print("\n=== Scenario Logs (excluding dialogue_history) ===")
for idx, log in enumerate(all_results):
    print(f"\n--- Scenario {idx + 1} ---")
    log_copy = log.copy()
    if "dialogue_history" in log_copy:
        del log_copy["dialogue_history"]
    print(json.dumps(log_copy, indent=2, default=str))
    print(f"Dialogue turns: {len(log['dialogue_history'])}")