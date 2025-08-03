from datetime import datetime
import argparse, os, json, time
from .prompts import ALL_BIASES
from .util import compare_results, get_log_file, log_scenario_data, analyze_consultation, get_completed_scenarios
from .scenario import ScenarioLoader
from .agent import PatientAgent, DoctorAgent, MeasurementAgent, SpecialistAgent

""" 
RELEASE NOTES:
1. Credit to Hassan and Liu et al. along with AgentClinic authors for code template
2. Adapting original MAS for to simulate multi-agent architectures
"""

# --- Constants ---
BASE_LOG_DIR = "logs"
MODEL_NAME = "gpt-4.1" 

# --- Simulation Configuration Constants ---
AGENT_DATASET = "MedQA"  # Start with MedQA as requested
NUM_SCENARIOS = 25       # Minimum 50 scenarios per bias-dataset combo
TOTAL_INFERENCES = 10
CONSULTATION_TURNS = 5

# --- Main Simulation Logic ---
def run_single_scenario(scenario, dataset, total_inferences, max_consultation_turns, scenario_idx, bias=None):
    patient_agent = PatientAgent(scenario=scenario)
    doctor_agent = DoctorAgent(scenario=scenario, max_infs=total_inferences, bias=bias)
    meas_agent = MeasurementAgent(scenario=scenario)
    specialist_agent = None

    available_tests = scenario.get_available_tests()
    run_log = {
        "timestamp": datetime.now(),
        "model": MODEL_NAME,
        "dataset": dataset,
        "scenario_id": scenario_idx,
        "bias_applied": bias,
        "max_patient_turns": total_inferences,
        "max_consultation_turns": max_consultation_turns,
        "correct_diagnosis": scenario.diagnosis_information(),
        "dialogue_history": [],
        "num_dialogue_turns": 0,
        "requested_tests": [],
        "tests_requested_count": 0,
        "available_tests": available_tests,
        "determined_specialist": None,
        "consultation_analysis": {},
        "final_doctor_diagnosis": None,
        "is_correct": None,
        "top_1_diagnosis": None,
        "top_1_confidence": None,
        "top_3_diagnoses": None,
        "top_3_confidence": None,
        "top_5_diagnoses": None,
        "top_5_confidence": None,
        "top_7_diagnoses": None,
        "top_7_confidence": None,
    }

    # --- Patient Interaction Phase --- # 
    # --------------------------------- # 
    print(f"\n--- Phase 1: Patient Interaction (Max {total_inferences} turns) ---")
    doctor_dialogue, state = doctor_agent.inference_doctor("Patient presents with initial information.", mode="patient")
    print(f"Doctor [Turn 0]: {doctor_dialogue}")
    run_log["dialogue_history"].append({"speaker": "Doctor", "turn": 0, "phase": "patient", "text": doctor_dialogue})
    meas_agent.add_hist(f"Doctor: {doctor_dialogue}")
    next_input_for_doctor = scenario.examiner_information()

    for turn in range(1, total_inferences + 1):
        if "REQUEST TEST" in doctor_dialogue: #Interact with measurement agent 
            try:
                test_name = doctor_dialogue.split("REQUEST TEST:", 1)[1].strip().rstrip('.?!')
                if test_name:
                    run_log["requested_tests"].append(test_name)
                    print(f"System: Logged test request - {test_name}")
            except IndexError:
                print("Warning: Could not parse test name from doctor request.")
                test_name = "Unknown Test"

            result = meas_agent.inference_measurement(doctor_dialogue)
            print(f"Measurement [Turn {turn}]: {result}")
            next_input_for_doctor = result
            run_log["dialogue_history"].append({"speaker": "Measurement", "turn": turn, "phase": "patient", "text": result})
            history_update = f"Doctor: {doctor_dialogue}\n\nMeasurement: {result}"
            patient_agent.add_hist(history_update)
            meas_agent.add_hist(history_update)
        else: #Interact with patient agent 
            patient_response = patient_agent.inference_patient(doctor_dialogue)
            print(f"Patient [Turn {turn}]: {patient_response}")
            next_input_for_doctor = patient_response
            run_log["dialogue_history"].append({"speaker": "Patient", "turn": turn, "phase": "patient", "text": patient_response})
            history_update = f"Patient: {patient_response}"
            meas_agent.add_hist(f"Doctor: {doctor_dialogue}\n\nPatient: {patient_response}") #TODO ~ Why don't we update the history for the pat agent here? 

        # Transition state ~
        doctor_dialogue, state = doctor_agent.inference_doctor(next_input_for_doctor, mode="patient")
        print(f"Doctor [Turn {turn}]: {doctor_dialogue}")
        run_log["dialogue_history"].append({"speaker": "Doctor", "turn": turn, "phase": "patient", "text": doctor_dialogue})
        meas_agent.add_hist(f"Doctor: {doctor_dialogue}")

        if state == "consultation_needed" or turn == total_inferences:
             print("\nPatient interaction phase complete.")
             break

        time.sleep(0.5)

    run_log["tests_requested_count"] = len(run_log["requested_tests"])
    run_log["tests_left_out"] = list(set(available_tests) - set(run_log["requested_tests"]))
    print(f"Total tests requested during patient interaction: {run_log['tests_requested_count']}")
    print(f"Tests left out: {run_log['tests_left_out']}")

    # --- Specialist Determination Phase --- # 
    # -------------------------------------- # 
    print(f"\n--- Phase 2: Determining Specialist ---")
    specialist_type, specialist_reason = doctor_agent.determine_specialist()
    run_log["determined_specialist"] = specialist_type
    run_log["specialist_reason"] = specialist_reason
    specialist_agent = SpecialistAgent(scenario=scenario, specialty=specialist_type)
    specialist_agent.agent_hist = doctor_agent.agent_hist
    last_specialist_response = "I have reviewed the patient's case notes. Please share your thoughts to begin the consultation."
    run_log["dialogue_history"].append({"speaker": "System", "turn": total_inferences + 1, "phase": "consultation", "text": f"Consultation started with {specialist_type}. Reason: {specialist_reason}"})


    # --- Specialist Consultation Phase --- #
    # ------------------------------------- # 
    print(f"\n--- Phase 3: Specialist Consultation (Max {max_consultation_turns} turns) ---")
    for consult_turn in range(1, max_consultation_turns + 1):
        full_turn = total_inferences + consult_turn

        # Doctor response
        doctor_consult_msg, state = doctor_agent.inference_doctor(last_specialist_response, mode="consultation")
        print(f"Doctor [Consult Turn {consult_turn}]: {doctor_consult_msg}")
        doctor_entry = {"speaker": "Doctor", "turn": full_turn, "phase": "consultation", "text": doctor_consult_msg}
        run_log["dialogue_history"].append(doctor_entry)

        # Specialist response 
        specialist_response = specialist_agent.inference_specialist(doctor_consult_msg)
        print(f"Specialist ({specialist_type}) [Consult Turn {consult_turn}]: {specialist_response}")
        specialist_entry = {"speaker": f"Specialist ({specialist_type})", "turn": full_turn, "phase": "consultation", "text": specialist_response}
        run_log["dialogue_history"].append(specialist_entry)
        last_specialist_response = specialist_response
        time.sleep(0.5)

    # --- Final Diagnosis Phase ---
    print("\n--- Phase 4: Final Diagnosis ---")
    final_diagnosis_result = doctor_agent.get_final_diagnosis()

    diagnoses_list = final_diagnosis_result.get("diagnoses", [])
    confidence_list = final_diagnosis_result.get("confidences", [])

    print(f"\nFinal Diagnosis: {diagnoses_list}")
    print(f"Confidence Scores: {confidence_list}")

    if len(confidence_list) < len(diagnoses_list):
        confidence_list.extend([None] * (len(diagnoses_list) - len(confidence_list)))

    # Log the final diagnosis and confidence
    print(f"\nFinal Diagnosis by Doctor: {diagnoses_list}")
    print(f"Correct Diagnosis: {scenario.diagnosis_information()}")
    print(f"Doctor's Self-Confidence: {confidence_list}")

    # Compare top-1 diagnosis to correct diagnosis for correctness
    is_correct = compare_results(diagnoses_list[0], scenario.diagnosis_information())
    print(f"Scenario {scenario_idx}: Diagnosis was {'CORRECT' if is_correct else 'INCORRECT'}")


    # Log all top-k diagnoses and confidence scores
    run_log["final_doctor_diagnosis"] = diagnoses_list[0]
    run_log["self_confidence"] = confidence_list
    run_log["is_correct"] = is_correct
    run_log["num_dialogue_turns"] = len(run_log["dialogue_history"])

    # Store top-k explicitly (if available)
    run_log["top_1_diagnosis"] = diagnoses_list[0] if len(diagnoses_list) > 0 else None
    run_log["top_1_confidence"] = confidence_list[0] if len(confidence_list) > 0 else None

    run_log["top_3_diagnoses"] = diagnoses_list[:3]
    run_log["top_3_confidence"] = confidence_list[:3]

    run_log["top_5_diagnoses"] = diagnoses_list[:5]
    run_log["top_5_confidence"] = confidence_list[:5]

    run_log["top_7_diagnoses"] = diagnoses_list[:7]
    run_log["top_7_confidence"] = confidence_list[:7]

    # --- Consultation Analysis Phase (Moved here) --- # 
    # ------------------------------------------------ # 
    print("\n--- Phase 5: Consultation Analysis ---")
    consultation_history_text = "\n".join([f"{entry['speaker']}: {entry['text']}" for entry in run_log["dialogue_history"] if entry["phase"] == "consultation"])
    if consultation_history_text:
        consultation_analysis_results = analyze_consultation(consultation_history_text)
        run_log["consultation_analysis"] = consultation_analysis_results
        print("Consultation Analysis Results:")
        if consultation_analysis_results:
            for key, value in consultation_analysis_results.items():
                if key != "test_density":
                     print(f"- {key.replace('_', ' ').title()}: {value}")
        else:
            print("Analysis could not be performed.")
    else:
        print("No consultation dialogue to analyze.")
        run_log["consultation_analysis"] = {"error": "No consultation dialogue recorded"}
    return run_log, run_log.get("is_correct", False)

def run_bias_dataset_combination(dataset, bias, num_scenarios, total_inferences, consultation_turns):
    """Run a single bias-dataset combination test"""
    # Custom filename for bias-corrected run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(BASE_LOG_DIR, f"{dataset}_{bias}_bias_corrected_{timestamp}.json")
    completed_scenario_ids = get_completed_scenarios(log_file) # Renamed for clarity
    
    print(f"\n=== Testing {bias} bias on {dataset} dataset ===")
    print(f"Log file: {log_file}")
    print(f"Already completed scenario IDs: {len(completed_scenario_ids)}")

    # Calculate number of correct scenarios from previous runs
    num_correct_from_previous_runs = 0
    if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
        with open(log_file, 'r') as f:
            try:
                previous_log_data = json.load(f)
                if isinstance(previous_log_data, list):
                    num_correct_from_previous_runs = sum(
                        1 for entry in previous_log_data
                        if entry.get("scenario_id") in completed_scenario_ids and entry.get("is_correct")
                    )
            except json.JSONDecodeError:
                print(f"Warning: Could not parse log file {log_file} for calculating previous accuracy.")
    
    scenario_loader = ScenarioLoader(dataset=dataset)
    max_available = scenario_loader.num_scenarios
    scenarios_to_run = min(num_scenarios, max_available)
    
    total_correct_current_session = 0 # Renamed for clarity
    total_simulated_current_session = 0 # Renamed for clarity
    
    # Create a list of scenarios to run, skipping already completed ones
    scenarios_to_process = [i for i in range(scenarios_to_run) if i not in completed_scenario_ids]
    print(f"Scenarios to run in this session: {len(scenarios_to_process)} of {scenarios_to_run} total planned")
    
    for scenario_idx in scenarios_to_process:
        print(f"\n--- Running Scenario {scenario_idx + 1}/{scenarios_to_run} with {bias} bias ---")
        scenario = scenario_loader.get_scenario(id=scenario_idx)

        if scenario is None:
            print(f"Error loading scenario {scenario_idx}, skipping.")
            continue

        total_simulated_current_session += 1
        run_log, is_correct = run_single_scenario(
            scenario, dataset, total_inferences, consultation_turns, scenario_idx, bias
        )        


        if is_correct:
            total_correct_current_session += 1

        log_scenario_data(run_log, log_file)
        print(f"Tests requested in Scenario {scenario_idx + 1}: {run_log.get('requested_tests', [])}")
        
        # Update progress
        if total_simulated_current_session > 0:
            accuracy_current_session = (total_correct_current_session / total_simulated_current_session) * 100
            print(f"\nCurrent Accuracy for this session ({bias} bias on {dataset}): {accuracy_current_session:.2f}% ({total_correct_current_session}/{total_simulated_current_session})")
            
            # Calculate overall progress including previously completed scenarios
            overall_completed_count = len(completed_scenario_ids) + total_simulated_current_session
            overall_correct_count = num_correct_from_previous_runs + total_correct_current_session
            
            overall_accuracy_so_far = (overall_correct_count / overall_completed_count) * 100 if overall_completed_count > 0 else 0
            # The original problematic line was:
            # overall_accuracy = ((len([s for s in completed_scenarios if s in run_log.get("is_correct", False)]) + total_correct) / 
            #                    overall_completed) * 100 if overall_completed > 0 else 0
            # This is now correctly calculated as overall_accuracy_so_far.
            print(f"Overall Progress for {bias} on {dataset}: {overall_completed_count}/{scenarios_to_run} scenarios completed. Overall Accuracy: {overall_accuracy_so_far:.2f}% ({overall_correct_count}/{overall_completed_count})")
    
    # Calculate final statistics for this combination
    final_completed_count = len(completed_scenario_ids) + total_simulated_current_session
    if final_completed_count > 0:
        # Load all results to get accurate count
        all_results = []
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                try:
                    all_results = json.load(f)
                    if not isinstance(all_results, list): # Ensure it's a list
                        all_results = []
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse final log file {log_file} for final stats. Results may be inaccurate.")
                    all_results = []

        correct_count_total = sum(1 for entry in all_results if entry.get("is_correct")) # Ensure entry.get("is_correct") is True
        
        # Ensure final_completed_count matches the number of entries if all were logged correctly
        # This uses the actual number of entries in the log file for accuracy if possible.
        actual_entries_in_log = len(all_results)
        
        final_accuracy = (correct_count_total / actual_entries_in_log) * 100 if actual_entries_in_log > 0 else 0
        
        print(f"\n=== Results for {bias} bias on {dataset} dataset ===")
        print(f"Total Scenarios Logged: {actual_entries_in_log} (planned: {scenarios_to_run}, completed this/prev sessions: {final_completed_count})")
        print(f"Final Accuracy: {final_accuracy:.2f}% ({correct_count_total}/{actual_entries_in_log})")
    
    return final_completed_count >= scenarios_to_run

def main():
    # Create argument parser for optional parameters
    parser = argparse.ArgumentParser(description='Run medical diagnosis simulation with bias testing')
    parser.add_argument('--dataset', choices=['MedQA', 'MedQA_Ext', 'NEJM', 'NEJM_Ext', 'all'], default='all',
                  help='Which dataset to use (default: all)')
    parser.add_argument('--bias', help='Specific bias to test (default: test all biases)')
    parser.add_argument('--scenarios', type=int, default=NUM_SCENARIOS,
                      help=f'Number of scenarios to run per combination (default: {NUM_SCENARIOS})')
    args = parser.parse_args()
    
    # Determine which datasets to test
    # Force to test only MedQA_Ext with no bias
    datasets_to_test = ['MedQA_Ext']
    biases_to_test = ['none']
    args.scenarios = 214  # Force 214 scenarios
    
    print(f"Starting comprehensive bias testing across {len(datasets_to_test)} datasets and {len(biases_to_test)} biases")
    print(f"Base settings: {args.scenarios} scenarios per combination, {TOTAL_INFERENCES} patient interactions, {CONSULTATION_TURNS} consultation turns")
    
    # Create summary report structures
    summary = {
        "start_time": datetime.now().isoformat(),
        "completed_combinations": 0,
        "total_combinations": len(datasets_to_test) * len(biases_to_test),
        "results_by_combination": {}
    }
    
    # Run each combination
    for dataset in datasets_to_test:
        for bias in biases_to_test:
            print(f"\n\n{'='*80}")
            print(f"TESTING: Dataset={dataset}, Bias={bias}")
            print(f"{'='*80}")
            
            try:
                completed = run_bias_dataset_combination(
                    dataset, bias, args.scenarios, TOTAL_INFERENCES, CONSULTATION_TURNS
                )
                
                # Update summary
                combination_key = f"{dataset}_{bias}"
                log_file = get_log_file(dataset, bias)
                
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        results = json.load(f)
                        correct_count = sum(1 for entry in results if entry.get("is_correct", False))
                        total_count = len(results);
                        
                        summary["results_by_combination"][combination_key] = {
                            "completed": completed,
                            "scenarios_run": total_count,
                            "correct_diagnoses": correct_count,
                            "accuracy": (correct_count / total_count) * 100 if total_count > 0 else 0
                        }
                
                if completed:
                    summary["completed_combinations"] += 1
                
            except Exception as e:
                print(f"Error running {dataset} with {bias} bias: {e}")
                # Continue with next combination even if this one fails
    
    # Save summary report
    summary["end_time"] = datetime.now().isoformat()
    summary["total_duration_seconds"] = (datetime.fromisoformat(summary["end_time"]) - 
                                        datetime.fromisoformat(summary["start_time"])).total_seconds()
    
    os.makedirs(BASE_LOG_DIR, exist_ok=True)  # Ensure directory exists
    with open(os.path.join(BASE_LOG_DIR, "bias_testing_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n\n=== BIAS TESTING COMPLETE ===")
    print(f"Completed {summary['completed_combinations']}/{summary['total_combinations']} combinations")
    print(f"Total duration: {summary['total_duration_seconds']/3600:.2f} hours")
    print(f"Full results saved to {os.path.join(BASE_LOG_DIR, 'bias_testing_summary.json')}")
    print("=== PATH DEBUGGING ===")
    print(f"Current working directory: {os.getcwd()}")
    print()

    # Test different possible paths
    possible_paths = [
        "base_files/data/agentclinic_medqa_extended.jsonl",
        "../data/agentclinic_medqa_extended.jsonl", 
        "../../base_files/data/agentclinic_medqa_extended.jsonl",
        "../base_files/data/agentclinic_medqa_extended.jsonl",
        "./base_files/data/agentclinic_medqa_extended.jsonl"
    ]

    print("Testing possible file paths:")
    for path in possible_paths:
        exists = os.path.exists(path)
        abs_path = os.path.abspath(path)
        print(f"  {path:<50} {'✅ EXISTS' if exists else '❌ NOT FOUND'}")
        if exists:
            print(f"    Absolute path: {abs_path}")
            print(f"    File size: {os.path.getsize(path)} bytes")

    print()
    print("Directory contents:")
    try:
        print("Current directory:", os.listdir("."))
    except:
        print("Could not list current directory")

    try:
        if os.path.exists("base_files"):
            print("base_files directory:", os.listdir("base_files"))
        if os.path.exists("base_files/data"):
            print("base_files/data directory:", os.listdir("base_files/data"))
    except:
        print("Could not list base_files directories")

if __name__ == "__main__":
    main()