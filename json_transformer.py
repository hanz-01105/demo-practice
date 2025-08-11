import json
import os
from datetime import datetime
import sys
import re

# Add the prototype directory to path to import scenario classes
sys.path.append('prototype')

# Regex patterns for demographic extraction
male_pattern = re.compile(r"\b(male|man|boy)\b", re.IGNORECASE)
female_pattern = re.compile(r"\b(female|woman|girl)\b", re.IGNORECASE)
age_pattern = re.compile(r"(\d+)[-\s]*year", re.IGNORECASE)

def extract_demographics_from_scenario_data(scenario_id, dataset_file):
    """
    Extract demographics directly from the original MedQA scenario data
    """
    try:
        with open(dataset_file, 'r') as f:
            scenarios = [json.loads(line) for line in f]
        
        if scenario_id >= len(scenarios):
            print(f"Scenario {scenario_id} not found in dataset")
            return None
        
        scenario = scenarios[scenario_id]
        
        # Extract patient actor information
        if "OSCE_Examination" in scenario and "Patient_Actor" in scenario["OSCE_Examination"]:
            patient_info = scenario["OSCE_Examination"]["Patient_Actor"]
            
            # If patient_info is a dict, look for demographic fields
            if isinstance(patient_info, dict):
                demographics_text = ""
                for key, value in patient_info.items():
                    demographics_text += f"{key}: {value} "
            else:
                # If it's a string, use it directly
                demographics_text = str(patient_info)
            
            return extract_demographics_from_text(demographics_text)
        
        print(f"No patient actor info found for scenario {scenario_id}")
        return None
        
    except Exception as e:
        print(f"Error extracting demographics for scenario {scenario_id}: {e}")
        return None

def extract_demographics_from_dialogue(dialogue_history):
    """
    Extract demographic information from dialogue history between doctor and patient
    """
    if not dialogue_history:
        return "No demographic data available"
    
    # Combine all dialogue text for analysis
    all_text = ""
    for turn in dialogue_history:
        if isinstance(turn, dict) and 'text' in turn:
            all_text += turn['text'] + " "
    
    all_text_lower = all_text.lower()
    
    # Age Group extraction - look for specific age mentions with fallback
    age_group = "30-40"  # Default fallback instead of "Unknown"
    age_patterns = [
        r"(\d+)[-\s]*year[-\s]*old",
        r"i'?m\s+(\d+)",
        r"age\s+(\d+)",
        r"(\d+)\s+years?\s+old"
    ]
    
    for pattern in age_patterns:
        age_match = re.search(pattern, all_text_lower)
        if age_match:
            age = int(age_match.group(1))
            age_group = (
                "0-10" if age <= 10 else
                "10-20" if age <= 20 else
                "20-30" if age <= 30 else
                "30-40" if age <= 40 else
                "40-50" if age <= 50 else
                "50-60" if age <= 60 else
                "60+"
            )
            break
    
    # If no age found, try to infer from context (medical complexity, work status, etc.)
    if age_group == "30-40" and not any(re.search(pattern, all_text_lower) for pattern in age_patterns):
        # Try to infer age from context clues
        if any(word in all_text_lower for word in ["student", "college", "university"]):
            age_group = "20-30"
        elif any(word in all_text_lower for word in ["retired", "retirement"]):
            age_group = "60+"
        elif any(word in all_text_lower for word in ["child", "pediatric", "school-age"]):
            age_group = "0-10"
        # else: keep default "30-40"
    
    # Gender extraction - look for explicit statements, default to "Other"
    gender = "Other"  # Always default to "Other", never "Unknown"
    
    # Check for female indicators first
    if re.search(r"(female|woman)", all_text_lower) or re.search(r"\b(she|her)\b", all_text_lower):
        gender = "Female"
    elif re.search(r"(male|man)", all_text_lower) or re.search(r"\b(he|him|his)\b", all_text_lower):
        gender = "Male"
    # else: keep "Other"
    
    # Smoking status - more comprehensive patterns, Unknown is acceptable
    smoking_status = "Unknown"  # Unknown is acceptable for smoking
    smoking_patterns = [
        r"(non-?smoker|never\s+smok|don'?t\s+smoke|do\s+not\s+smoke)",
        r"(quit\s+smoking|former\s+smoker|stopped\s+smoking|ex-?smoker|used\s+to\s+smoke)",
        r"(smoke|smoking|smoker|cigarette|pack\s+per\s+day|tobacco)"
    ]
    
    if re.search(smoking_patterns[0], all_text_lower):
        smoking_status = "Non-smoker"
    elif re.search(smoking_patterns[1], all_text_lower):
        smoking_status = "Non-smoker"  # Former smokers count as non-smokers
    elif re.search(smoking_patterns[2], all_text_lower):
        smoking_status = "Smoker"
    # else: keep "Unknown" - this is acceptable
    
    # Alcohol use patterns, Unknown is acceptable
    alcohol_use = "Unknown"  # Unknown is acceptable for alcohol
    alcohol_patterns = [
        r"(non-?drinker|never\s+drink|don'?t\s+drink|do\s+not\s+drink|teetotal)",
        r"(social\s+drink|occasional|wine|beer|alcohol|drink\s+wine|drinks?\s+occasionally)"
    ]
    
    if re.search(alcohol_patterns[0], all_text_lower):
        alcohol_use = "Non-drinker"
    elif re.search(alcohol_patterns[1], all_text_lower):
        alcohol_use = "Drinker"
    # else: keep "Unknown" - this is acceptable
    
    # Drug use with fallback
    drug_use = "Non-drug User"  # Default fallback instead of "Unknown"
    if any(phrase in all_text_lower for phrase in ["no drug", "never used drugs", "no illicit", "no recreational drugs"]):
        drug_use = "Non-drug User"
    elif any(word in all_text_lower for word in ["drug use", "cocaine", "heroin", "marijuana", "cannabis", "illicit drugs"]):
        drug_use = "Drug User"
    # else: keep default "Non-drug User"
    
    # Occupation type - look for job titles with better fallback
    occupation_type = "Knowledge Worker"  # Default fallback
    
    # Knowledge worker occupations
    knowledge_jobs = [
        "designer", "graphic designer", "engineer", "teacher", "doctor", "nurse", 
        "lawyer", "programmer", "analyst", "manager", "consultant", "architect",
        "accountant", "therapist", "counselor", "researcher", "scientist"
    ]
    
    # Manual labor occupations  
    manual_jobs = [
        "construction", "factory", "mechanic", "plumber", "electrician", 
        "carpenter", "welder", "driver", "cleaner", "cook", "chef"
    ]
    
    if any(job in all_text_lower for job in knowledge_jobs):
        occupation_type = "Knowledge Worker"
    elif any(job in all_text_lower for job in manual_jobs):
        occupation_type = "Manual Labor"
    elif any(word in all_text_lower for word in ["student", "college", "university", "school"]):
        occupation_type = "Student"
    elif any(word in all_text_lower for word in ["retired", "retirement"]):
        occupation_type = "Retired"
    elif any(word in all_text_lower for word in ["unemployed", "jobless", "no job"]):
        occupation_type = "Unemployed"
    # else: keep default "Knowledge Worker"
    
    # Comorbidity status - look for medical history mentions with fallback
    comorbidity_status = "No Significant PMHx"  # Default fallback
    
    chronic_conditions = [
        "diabetes", "hypertension", "high blood pressure", "cancer", "heart disease", 
        "copd", "asthma", "arthritis", "kidney disease", "liver disease"
    ]
    
    immunosuppressed_terms = [
        "immunosuppressed", "chemotherapy", "transplant", "immunocompromised", 
        "steroids", "immunosuppressive"
    ]
    
    no_history_phrases = [
        "no significant past medical", "no significant medical history", 
        "no past medical history", "no medical problems", "no health problems",
        "no significant past medical problems", "unremarkable medical history"
    ]
    
    if any(condition in all_text_lower for condition in chronic_conditions):
        comorbidity_status = "Chronic Condition Present"
    elif any(term in all_text_lower for term in immunosuppressed_terms):
        comorbidity_status = "Immunosuppressed/Special Treatment"
    elif any(phrase in all_text_lower for phrase in no_history_phrases):
        comorbidity_status = "No Significant PMHx"
    # else: keep default "No Significant PMHx"
    
    # Symptom presentation - analyze the complexity and nature of symptoms described
    symptom_presentation = "Classic Textbook"  # Default for medical cases
    
    # Look for complexity indicators
    symptom_keywords = [
        "pain", "headache", "nausea", "vomiting", "fever", "cough", 
        "shortness of breath", "fatigue", "weakness", "dizziness", 
        "double vision", "difficulty", "problems"
    ]
    
    symptom_count = sum(1 for symptom in symptom_keywords if symptom in all_text_lower)
    
    # Check for presentation descriptors
    if any(word in all_text_lower for word in ["vague", "atypical", "unusual", "strange", "non-specific", "unclear"]):
        symptom_presentation = "Atypical/Vague Wording"
    elif any(word in all_text_lower for word in ["classic", "typical", "textbook", "characteristic"]):
        symptom_presentation = "Classic Textbook"
    elif symptom_count >= 4:  # Multiple symptoms across systems
        symptom_presentation = "Multi-System Complex"
    elif symptom_count == 1:
        symptom_presentation = "Single Symptom Only"
    # else: keep "Classic Textbook" as default
    
    # Format as pandas Series string representation
    demographics_string = f"""age_group                             {age_group}
gender                               {gender}
smoking_status                   {smoking_status}
alcohol_use                         {alcohol_use}
drug_use                            {drug_use}
occupation_type            {occupation_type}
comorbidity_status      {comorbidity_status}
symptom_presentation                {symptom_presentation}
dtype: object"""
    
    return demographics_string

def extract_log_data(log_file_path, output_file_path):
    """
    Extract data from the existing log file and reformat it with proper demographics
    """
    
    # Read the existing log file
    if not os.path.exists(log_file_path):
        print(f"Error: Log file {log_file_path} not found")
        return
    
    with open(log_file_path, 'r') as f:
        try:
            existing_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Could not parse JSON from {log_file_path}")
            return
    
    if not isinstance(existing_data, list):
        print("Error: Expected a list of entries in the log file")
        return
    
    # Extract timestamp from filename
    filename = os.path.basename(log_file_path)
    if "_" in filename:
        parts = filename.split("_")
        run_timestamp = None
        for i, part in enumerate(parts):
            if len(part) == 8 and part.isdigit():  # YYYYMMDD
                if i + 1 < len(parts):
                    time_part = parts[i + 1].split(".")[0]  # Remove .json extension
                    if len(time_part) == 6 and time_part.isdigit():  # HHMMSS
                        run_timestamp = f"{part}_{time_part}"
                        break
        if not run_timestamp:
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine dataset from first entry
    dataset = "Unknown"
    if existing_data:
        dataset = existing_data[0].get("dataset", "Unknown")
    
    print(f"Extracting demographics from dialogue history for {len(existing_data)} entries")
    
    # Reformat each entry
    reformatted_results = []
    for entry in existing_data:
        # Convert timestamp if it's a datetime object
        timestamp = entry.get("timestamp")
        if isinstance(timestamp, dict):
            timestamp = str(timestamp)
        elif not isinstance(timestamp, str):
            timestamp = str(timestamp) if timestamp else datetime.now().isoformat()
        
        # Extract demographics from dialogue history instead of scenario data
        demographics = "No demographic data available"
        dialogue_history = entry.get("dialogue_history", [])
        if dialogue_history:
            demographics = extract_demographics_from_dialogue(dialogue_history)
        
        reformatted_entry = {
            "timestamp": timestamp,
            "model": entry.get("model", "unknown"),
            "dataset": entry.get("dataset", dataset),
            "scenario_id": entry.get("scenario_id"),
            "bias_applied": entry.get("bias_applied"),
            "max_patient_turns": entry.get("max_patient_turns"),
            "max_consultation_turns": entry.get("max_consultation_turns"),
            "correct_diagnosis": entry.get("correct_diagnosis"),
            "num_dialogue_turns": entry.get("num_dialogue_turns"),
            "requested_tests": entry.get("requested_tests", []),
            "tests_requested_count": entry.get("tests_requested_count", 0),
            "available_tests": entry.get("available_tests", []),
            "determined_specialist": entry.get("determined_specialist"),
            "consultation_analysis": entry.get("consultation_analysis", {}),
            "final_doctor_diagnosis": entry.get("final_doctor_diagnosis"),
            "is_correct": entry.get("is_correct"),
            "tests_left_out": entry.get("tests_left_out", []),
            "specialist_reason": entry.get("specialist_reason"),
            "self_confidence": entry.get("self_confidence"),
            "demographics": demographics
        }
        
        reformatted_results.append(reformatted_entry)
    
    # Create the final structure
    final_structure = {
        "run_timestamp": run_timestamp,
        "dataset": dataset,
        "results": reformatted_results
    }
    
    # Write to output file
    with open(output_file_path, 'w') as f:
        json.dump(final_structure, f, indent=2, default=str)
    
    print(f"Successfully extracted {len(reformatted_results)} entries")
    print(f"Output saved to: {output_file_path}")
    return final_structure

def main():
    # Running from base_files directory
    input_log_file = "base_files/logs/MedQA_Ext_none_claude.json"
    output_file = "base_files/logs/extracted_results_MedQA_Ext_claude.json"
    
    print(f"Extracting data from: {input_log_file}")
    print(f"Output will be saved to: {output_file}")
    
    if not os.path.exists(input_log_file):
        print(f"Error: Log file {input_log_file} not found")
        return
    
    result = extract_log_data(input_log_file, output_file)
    
    if result:
        print(f"\nSummary:")
        print(f"- Run timestamp: {result['run_timestamp']}")
        print(f"- Dataset: {result['dataset']}")
        print(f"- Total scenarios: {len(result['results'])}")
        
        # Calculate some basic stats
        correct_count = sum(1 for r in result['results'] if r.get('is_correct'))
        accuracy = (correct_count / len(result['results'])) * 100 if result['results'] else 0
        print(f"- Correct diagnoses: {correct_count}/{len(result['results'])}")
        print(f"- Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()