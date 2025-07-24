import json
import os
import re
import pandas as pd
from typing import Dict, Any, List
import numpy as np # Imported for dummy data generation

# --- Constants ---
LOG_PATH = "base_files/logs"
LOG_FILE = "agentclinic_run_latest.json"

# Symptom categorization logic
CLASSIC_KEYWORDS = [
    "chest pain", "shortness of breath", "double vision", "ptosis", "palpitations", "seizure",
    "visual loss", "hematuria", "fever", "rigors", "photophobia", "stiff neck", "rash",
    "syncope", "hemoptysis", "meningismus", "clubbing", "cyanosis", "jaundice"
]

VAGUE_KEYWORDS = [
    "vague", "weird", "unclear", "confused", "nonspecific", "funny feeling", "off", "just not right",
    "not myself", "weird sensation", "general malaise", "feeling unwell", "uncomfortable"
]

MULTISYSTEM_KEYWORDS = {
    "neuro": ["ataxia", "dizziness", "numbness", "tingling", "weakness", "headache", "ptosis", "vision loss", "slurred speech", "seizure", "confusion", "gait instability", "tremor", "memory loss"],
    "gi": ["nausea", "vomiting", "diarrhea", "abdominal pain", "cramping", "constipation", "weight loss", "hematemesis", "melena", "bloating", "early satiety"],
    "resp": ["cough", "dyspnea", "shortness of breath", "wheezing", "hemoptysis", "chest tightness", "pleuritic pain"],
    "cardiac": ["palpitations", "chest pain", "syncope", "orthopnea", "paroxysmal nocturnal dyspnea", "edema", "tachycardia", "bradycardia"],
    "msk": ["joint pain", "muscle pain", "swelling", "stiffness", "back pain", "difficulty walking", "limb weakness", "loss of coordination"],
    "derm": ["rash", "itching", "hives", "bruising", "skin discoloration", "dry skin", "lesion"],
    "gu": ["dysuria", "hematuria", "incontinence", "frequency", "urgency", "flank pain", "retention"]
}


def load_and_process_log(filepath: str) -> pd.DataFrame:
    """
    Loads a JSON log file and normalizes the 'logs' section into a DataFrame.
    """
    print(f"Loading and processing log file: {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Log file not found: {filepath}")

    with open(filepath, "r", encoding='utf-8') as f:
        data = json.load(f)

    if "logs" not in data:
        raise KeyError("The key 'logs' was not found in the JSON file.")

    df = pd.json_normalize(data["logs"])
    print(f"Successfully loaded log. Shape: {df.shape}")
    return df

# --- Helper function for deep JSON searching ---
def deep_explore_json_structure(obj: Any, path: str = "", max_depth: int = 4, current_depth: int = 0) -> List[Dict]:
    """
    Recursively explores a JSON-like object to find potential locations of symptom data.
    """
    findings = []
    if current_depth > max_depth:
        return findings

    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            key_lower = key.lower()
            # Check if this key might contain symptom information based on its name
            if any(word in key_lower for word in ['symptom', 'complaint', 'present', 'chief', 'history', 'scenario', 'case']):
                findings.append({'path': new_path, 'value': value})

            # Recurse into nested structures
            if isinstance(value, (dict, list)):
                findings.extend(deep_explore_json_structure(value, new_path, max_depth, current_depth + 1))

    elif isinstance(obj, list) and obj:
        # For lists, examine the first few items
        for i, item in enumerate(obj[:3]):
            new_path = f"{path}[{i}]"
            if isinstance(item, (dict, list)):
                findings.extend(deep_explore_json_structure(item, new_path, max_depth, current_depth + 1))

    return findings

# --- Enhanced symptom extraction logic ---
def enhanced_extract_symptoms_from_log(row: pd.Series) -> Dict[str, Any]:
    """
    Enhanced symptom extraction using a multi-step, robust search strategy.
    """
    # Method 1: Check the original hard-coded path
    if 'OSCE_Examination.Patient_Actor.Symptoms' in row and isinstance(row['OSCE_Examination.Patient_Actor.Symptoms'], dict):
        symptoms = row['OSCE_Examination.Patient_Actor.Symptoms']
        if symptoms:
            return symptoms

    # Method 2: Check other common direct field names
    direct_fields = ['symptoms', 'Symptoms', 'symptom_data', 'chief_complaint', 'presenting_complaint']
    for field in direct_fields:
        if field in row and row[field]:
            if isinstance(row[field], dict):
                return row[field]
            if isinstance(row[field], str):
                return {'Primary_Symptom': row[field], 'Secondary_Symptoms': []}

    # Method 3: Combine text from various description fields
    text_fields = ['scenario', 'case_description', 'patient_history', 'description', 'specialist_reason']
    combined_text = ' '.join([str(row[field]) for field in text_fields if field in row and isinstance(row[field], str)])

    # Method 4: If no text yet, perform a deep search on the entire row object
    if not combined_text.strip():
        row_dict = row.to_dict()
        findings = deep_explore_json_structure(row_dict)
        for find in findings:
            if isinstance(find['value'], str):
                combined_text += ' ' + find['value']
            elif isinstance(find['value'], dict) and 'Primary_Symptom' in find['value']:
                 return find['value'] # Found a structured dict, return it

    # Final result: create a pseudo-symptom object from the combined text
    if combined_text.strip():
        return {'Primary_Symptom': combined_text.strip(), 'Secondary_Symptoms': []}

    return {} # Return empty dict if nothing is found

def categorize_symptom_presentation(symptom_data: Dict[str, Any]) -> str:
    """
    Categorize symptom presentation with corrected logic for single symptom detection.
    """
    if not isinstance(symptom_data, dict) or not symptom_data:
        return 'Unclassified'

    primary = symptom_data.get('Primary_Symptom', '')
    secondary = symptom_data.get('Secondary_Symptoms', [])

    if isinstance(secondary, str):
        secondary = [secondary]
    elif not isinstance(secondary, list):
        secondary = []

    all_symptoms = [primary] + secondary
    text = ' '.join([str(s) for s in all_symptoms if s]).lower()

    if not text.strip():
        return 'Unclassified'

    # --- Count all keywords found in the text ---
    all_system_keywords = [kw for sublist in MULTISYSTEM_KEYWORDS.values() for kw in sublist]
    all_defined_keywords = CLASSIC_KEYWORDS + VAGUE_KEYWORDS + all_system_keywords
    # Use a set to count unique keywords found in the text
    found_keywords = {kw for kw in all_defined_keywords if kw in text}
    keyword_count = len(found_keywords)

    # --- REVISED LOGIC ORDER ---

    # 1. Check for a classic "textbook" presentation first. This is the highest priority.
    if any(kw in text for kw in CLASSIC_KEYWORDS):
        return 'Classic Textbook'

    # 2. Check for multi-system involvement next. This is also a specific category.
    systems_triggered = sum(1 for system_keywords in MULTISYSTEM_KEYWORDS.values() if any(kw in text for kw in system_keywords))
    if systems_triggered >= 2:
        return 'Multi-System Complex'

    # 3. Only after ruling out the above, check for explicitly vague keywords.
    if any(kw in text for kw in VAGUE_KEYWORDS):
        return 'Atypical/Vague Wording'

    # 4. ACCURATE SINGLE SYMPTOM CHECK: Check if exactly one keyword was found.
    if keyword_count == 1:
        return 'Single Symptom Only'

    # 5. As a final fallback, classify as atypical.
    return 'Atypical/Vague Wording'

def diagnose_categorization(symptom_data: Dict[str, Any]) -> str:
    """
    Provides diagnostic reasoning for why each case was categorized.
    This function explains the categorization decision step by step.
    """
    if not isinstance(symptom_data, dict) or not symptom_data:
        return 'REASON: Empty or invalid symptom data - classified as Unclassified'

    primary = symptom_data.get('Primary_Symptom', '')
    secondary = symptom_data.get('Secondary_Symptoms', [])

    if isinstance(secondary, str):
        secondary = [secondary]
    elif not isinstance(secondary, list):
        secondary = []

    all_symptoms = [primary] + secondary
    text = ' '.join([str(s) for s in all_symptoms if s]).lower()

    if not text.strip():
        return 'REASON: No symptom text found - classified as Unclassified'

    # Analyze what keywords are found
    classic_found = [kw for kw in CLASSIC_KEYWORDS if kw in text]
    vague_found = [kw for kw in VAGUE_KEYWORDS if kw in text]
    
    # Check system involvement
    systems_with_symptoms = []
    for system, keywords in MULTISYSTEM_KEYWORDS.items():
        system_symptoms = [kw for kw in keywords if kw in text]
        if system_symptoms:
            systems_with_symptoms.append(f"{system}: {system_symptoms}")
    
    # Count unique keywords
    all_system_keywords = [kw for sublist in MULTISYSTEM_KEYWORDS.values() for kw in sublist]
    all_defined_keywords = CLASSIC_KEYWORDS + VAGUE_KEYWORDS + all_system_keywords
    found_keywords = {kw for kw in all_defined_keywords if kw in text}
    keyword_count = len(found_keywords)

    # Build reasoning based on categorization logic
    reasoning_parts = []
    
    if classic_found:
        reasoning_parts.append(f"Classic keywords found: {classic_found}")
        return f"REASON: {'; '.join(reasoning_parts)} → Classic Textbook"
    
    if len(systems_with_symptoms) >= 2:
        reasoning_parts.append(f"Multi-system involvement ({len(systems_with_symptoms)} systems): {systems_with_symptoms}")
        return f"REASON: {'; '.join(reasoning_parts)} → Multi-System Complex"
    
    if vague_found:
        reasoning_parts.append(f"Vague keywords found: {vague_found}")
        return f"REASON: {'; '.join(reasoning_parts)} → Atypical/Vague Wording"
    
    if keyword_count == 1:
        reasoning_parts.append(f"Exactly one medical keyword found: {list(found_keywords)}")
        return f"REASON: {'; '.join(reasoning_parts)} → Single Symptom Only"
    
    # Fallback case
    if systems_with_symptoms:
        reasoning_parts.append(f"Single system involvement: {systems_with_symptoms}")
    if keyword_count > 1:
        reasoning_parts.append(f"Multiple keywords ({keyword_count}) but no classic pattern: {list(found_keywords)}")
    elif keyword_count == 0:
        reasoning_parts.append("No defined medical keywords found")
    
    return f"REASON: {'; '.join(reasoning_parts) if reasoning_parts else 'No clear pattern identified'} → Atypical/Vague Wording"

def calculate_performance_metrics(df: pd.DataFrame, group_col: str, actual_col: str = 'is_correct') -> pd.DataFrame:
    """
    Calculates key performance metrics (Sensitivity, Correct/Incorrect Counts) for each category
    in a specified group column.
    """
    if group_col not in df.columns:
        raise KeyError(f"Grouping column '{group_col}' not found in the DataFrame.")
    
    if actual_col not in df.columns:
        print(f"Warning: Actual outcome column '{actual_col}' not found. Creating a dummy column for demonstration.")
        rng = np.random.default_rng(seed=42)
        df[actual_col] = rng.choice([True, False], size=len(df), p=[0.75, 0.25])

    df[actual_col] = pd.to_numeric(df[actual_col], errors='coerce').astype('boolean')
    valid_df = df.dropna(subset=[actual_col, group_col])

    def calculate_metrics(group_data):
        correct_cases = group_data[actual_col].sum()
        total_cases = len(group_data)
        incorrect_cases = total_cases - correct_cases
        sensitivity = (correct_cases / total_cases * 100) if total_cases > 0 else 0

        return pd.Series({
            'Sensitivity_%': sensitivity,
            'Total_Cases': total_cases,
            'Correct_Cases': correct_cases,
            'Incorrect_Cases': incorrect_cases,
        })

    grouped_metrics = valid_df.groupby(group_col).apply(calculate_metrics, include_groups=False).reset_index()
    grouped_metrics = grouped_metrics.sort_values(by='Sensitivity_%', ascending=False).rename(columns={'symptom_presentation': 'Symptom Category'})
    
    return grouped_metrics


def extract_demographics_data(demo_str: str) -> pd.Series:
    """Extract structured data from the demographics string if it exists."""
    result = {"age_group": "Unknown", "gender": "Other"}
    if not isinstance(demo_str, str):
        return pd.Series(result)

    age_match = re.search(r"age_group:\s*([\w-]+)", demo_str)
    gender_match = re.search(r"gender:\s*(\w+)", demo_str)

    if age_match:
        result['age_group'] = age_match.group(1)
    if gender_match:
        result['gender'] = gender_match.group(1)
        
    return pd.Series(result)


# Main execution block
if __name__ == "__main__":
    try:
        # This is a placeholder. Create the directory and a dummy file if they don't exist.
        if not os.path.exists(LOG_PATH):
            os.makedirs(LOG_PATH)
        log_filepath = os.path.join(LOG_PATH, LOG_FILE)
        if not os.path.exists(log_filepath):
            print(f"Creating dummy log file at: {log_filepath}")
            dummy_data = {
                "logs": [
                    {"scenario_id": 1, "is_correct": True, "OSCE_Examination": {"Patient_Actor": {"Symptoms": {"Primary_Symptom": "chest pain", "Secondary_Symptoms": ["shortness of breath"]}}}},
                    {"scenario_id": 2, "is_correct": False, "description": "Patient feels a funny feeling and general malaise."},
                    {"scenario_id": 3, "is_correct": True, "case_description": "45-year-old with sudden onset of weakness and dizziness."},
                    {"scenario_id": 4, "is_correct": False, "patient_history": "Presents with a rash and joint pain."},
                    {"scenario_id": 5, "is_correct": True, "symptoms": {"Primary_Symptom": "headache", "Secondary_Symptoms": []}},
                    {"scenario_id": 6, "is_correct": True, "specialist_reason": "Patient has had a persistent cough."},
                    {"scenario_id": 7, "is_correct": True, "scenario": "Unclear symptoms, patient is confused."}
                ]
            }
            with open(log_filepath, 'w') as f:
                json.dump(dummy_data, f)
        
        processed_df = load_and_process_log(log_filepath)

        # --- STEP 1: Use the enhanced functions to get symptom data and categorize ---
        print("\nExtracting and categorizing symptoms with enhanced logic...")
        processed_df['symptoms_raw'] = processed_df.apply(enhanced_extract_symptoms_from_log, axis=1)
        processed_df['symptom_presentation'] = processed_df['symptoms_raw'].apply(categorize_symptom_presentation)
        print("Symptom categorization complete.")

        # --- STEP 2: Run and display diagnostics for categorization ---
        print("\n" + "="*25 + " RUNNING DIAGNOSTICS " + "="*25)
        print("The 'diagnosis_reasoning' column shows why each case was categorized.")
        processed_df['diagnosis_reasoning'] = processed_df['symptoms_raw'].apply(diagnose_categorization)
        diagnostic_view = processed_df[['scenario_id', 'symptom_presentation', 'diagnosis_reasoning']]
        print(diagnostic_view.to_string())
        print("="*75 + "\n")

        # --- STEP 3: Show categorization distribution summary ---
        print("\n" + "="*25 + " CATEGORIZATION SUMMARY " + "="*25)
        category_counts = processed_df['symptom_presentation'].value_counts()
        print("Distribution of symptom presentations:")
        for category, count in category_counts.items():
            percentage = (count / len(processed_df)) * 100
            print(f"  {category}: {count} cases ({percentage:.1f}%)")
        print("="*75 + "\n")
        
        # --- STEP 4: Calculate and display sensitivity metrics ---
        print("\n" + "="*25 + " SENSITIVITY METRICS BY CATEGORY " + "="*25)
        print("Calculating model sensitivity for each symptom presentation category...")
        
        performance_df = calculate_performance_metrics(
            processed_df,
            group_col='symptom_presentation',
            actual_col='is_correct'
        )

        # Loop through the results and print in the custom format
        for index, row in performance_df.iterrows():
            category = row['Symptom Category']
            sensitivity = row['Sensitivity_%']
            correct = row['Correct_Cases']
            incorrect = row['Incorrect_Cases']

            print(f"\n--- {category} ---")
            print(f"  Sensitivity: {sensitivity:.0f}%")
            print(f"  Correct Diagnoses: {correct}")
            print(f"  Missed Diagnoses: {incorrect}")
            
        print("\n" + "="*75 + "\n")

    except (FileNotFoundError, json.JSONDecodeError, KeyError, Exception) as e:
        print(f"\nAn error occurred during processing: {e}")
        import traceback
        traceback.print_exc()