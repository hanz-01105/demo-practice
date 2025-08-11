import anthropic, re, random, time, json, os
from datetime import datetime
import argparse
import glob
from prompts import COGNITIVE_BIASES, DEMOGRAPHIC_BIASES

BASE_LOG_DIR = "base_files/logs" # TODO ~ Add to .gitignore
MODEL_NAME = "claude-sonnet-4-20250514" 

# --- Utility Functions ---
def query_model(prompt, system_prompt, max_tokens=200):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)  # Change this line
    
    # Anthropic format - system prompt is separate
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        system=system_prompt,  # System prompt goes here
        messages=[
            {"role": "user", "content": prompt}  # Only user message here
        ],
        temperature=0.05,
        max_tokens=max_tokens,
    )
    
    # Extract text from Anthropic response
    return response.content[0].text

def compare_results(diagnosis, correct_diagnosis):
    prompt = f"Here is the correct diagnosis: {correct_diagnosis}\nHere was the doctor dialogue/diagnosis: {diagnosis}\nAre these referring to the same underlying medical condition? Please respond only with Yes or No."
    system_prompt = "You are an expert medical evaluator. Determine if the provided doctor's diagnosis matches the correct diagnosis in meaning, even if phrased differently. Respond only with 'Yes' or 'No'."
    answer = query_model(prompt, system_prompt)
    return answer.strip().lower() == "yes"

def get_log_file(dataset, bias_name):
    """
    Creates a log file name.
    Uses a specific, hardcoded file for the 'MedQA_Ext'/'none' run,
    and generates standard names for all other runs.
    """
    os.makedirs(BASE_LOG_DIR, exist_ok=True)

    # Check if this is the specific run you want to use the special file for
    if dataset == 'MedQA_Ext' and bias_name == 'none':
        # For this exact combination, return your desired filename
        return os.path.join(BASE_LOG_DIR, "MedQA_Ext_none_bias_corrected_20250804_211039.json")
    else:
        # For all other future experiments, create a clean, standard filename
        return os.path.join(BASE_LOG_DIR, f"{dataset}_{bias_name}_bias_corrected.json")

def log_scenario_data(data, log_file):
    """Log data to a specific log file"""
    # Ensure datetime is serializable
    if isinstance(data.get("timestamp"), datetime):
        data["timestamp"] = data["timestamp"].isoformat()
    
    existing_data = []
    if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
        with open(log_file, 'r') as f:
            existing_data = json.load(f)
            if not isinstance(existing_data, list):
                existing_data = []
    
    existing_data.append(data)
    with open(log_file, 'w') as f:
        json.dump(existing_data, f, indent=2)

def analyze_consultation(consultation_history):
    """
    Analyzes the doctor-specialist consultation dialogue using an LLM.

    Args:
        consultation_history (str): The string containing the dialogue between
                                     the doctor and the specialist.

    Returns:
        dict: A dictionary containing the analysis metrics.
              Returns an empty dict if analysis fails.
    """
    prompt = f"""
        Analyze the following medical consultation dialogue between a primary doctor and a specialist. Provide the analysis in JSON format with the following keys:
        - "premature_conclusion": (Boolean) Did the primary doctor jump to a conclusion without sufficient discussion or evidence gathering during the consultation?
        - "diagnoses_considered": (List) List all distinct potential diagnoses explicitly mentioned or discussed during the consultation.
        - "diagnoses_considered_count": (Integer) Count the number of distinct potential diagnoses explicitly mentioned or discussed during the consultation.
        - "disagreements": (Integer) Count the number of explicit disagreements or significant divergences in opinion between the doctor and the specialist.

        Consultation Dialogue:
        ---
        {consultation_history}
        ---

        Respond ONLY with the JSON object.
        """
    system_prompt = "You are a medical education evaluator analyzing a consultation dialogue. Extract specific metrics and provide them in JSON format."

    analysis_json_str = query_model(prompt, system_prompt, max_tokens=300)

    try:
        # Clean potential markdown code block fences
        if analysis_json_str.startswith("```json"):
            analysis_json_str = analysis_json_str[7:]
        if analysis_json_str.endswith("```"):
            analysis_json_str = analysis_json_str[:-3]
        
        analysis_results = json.loads(analysis_json_str.strip())
        required_keys = ["premature_conclusion", "diagnoses_considered", "diagnoses_considered_count", "disagreements"]
        if all(key in analysis_results for key in required_keys):
            return analysis_results
        else:
            print(f"Warning: LLM analysis response missing required keys. Response: {analysis_json_str}")
            return {}
    except json.JSONDecodeError:
        print(f"Warning: Failed to parse LLM analysis response as JSON. Response: {analysis_json_str}")
        return {}
    except Exception as e:
        print(f"Warning: An error occurred during consultation analysis: {e}")
        return {}

def get_completed_scenarios(log_file):
    """Get list of scenario IDs that have already been completed"""
    if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
        return []
    
    with open(log_file, 'r') as f:
        try:
            data = json.load(f)
            return [entry.get("scenario_id") for entry in data if entry.get("scenario_id") is not None]
        except json.JSONDecodeError:
            print(f"Warning: Could not parse log file {log_file}. Starting from scratch.")
            return []
