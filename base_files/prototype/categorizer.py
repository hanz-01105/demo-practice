import json
import re
import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

male_pattern = re.compile(r"\b(male|man|boy)\b", re.IGNORECASE)
female_pattern = re.compile(r"\b(female|woman|girl)\b", re.IGNORECASE)
age_pattern = re.compile(r"(\d+)[-\s]*year", re.IGNORECASE)


def classify_bias_llm(demographics, social_history, past_medical_history, history_of_present_illness):
    prompt = f"""
You are a careful medical data classifier.

Given the following patient Demographics, Social_History, Past_Medical_History, and History_of_Present_Illness, classify each bias variable into one of the allowed categories below. If there is no clear mention, return "Unknown" for that variable.

---

Allowed values:

- Smoking Status: ["Smoker", "Non-smoker", "Unknown"]
- Alcohol Use: ["Drinker", "Non-drinker", "Unknown"]
- Drug Use: ["Drug User", "Non-drug User", "Unknown"]
- Occupation Type: ["Manual Labor", "Knowledge Worker", "Student", "Retired", "Unemployed", "Unknown"]
- Comorbidity Status: ["Chronic Condition Present", "Immunosuppressed/Special Treatment", "No Significant PMHx", "Unknown"]
- Symptom Presentation: ["Classic Textbook", "Atypical/Vague Wording", "Multi-System Complex", "Single Symptom Only", "Unknown"]

---

Example Input:

Demographics: "65-year-old female"
Social_History: "Smokes 1 pack per day, drinks socially."
Past_Medical_History: "Hypertension, Diabetes Mellitus."
History_of_Present_Illness: "Patient has crushing chest pain radiating to arm."

Example Output:

{{
  "Smoking Status": "Smoker",
  "Alcohol Use": "Drinker",
  "Drug Use": "Unknown",
  "Occupation Type": "Unknown",
  "Comorbidity Status": "Hypertension",
  "Symptom Presentation": "Classic Textbook"
}}

---

Now classify this text:

Demographics: "{demographics}"
Social_History: "{social_history}"
Past_Medical_History: "{past_medical_history}"
History_of_Present_Illness: "{history_of_present_illness}"

Only output strict JSON.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a careful medical data classifier."},
            {"role": "user", "content": prompt}
        ]
    )

    raw_output = response.choices[0].message.content.strip()
    raw_output = re.sub(r"```[a-z]*", "", raw_output).strip()
    raw_output = re.sub(r"```", "", raw_output).strip()
    return json.loads(raw_output)


def categorize_scenario(patient_info):
    demographics = patient_info.get('Demographics', '').strip()
    demographics_lower = demographics.lower()

    # Age Group
    age_group = "Unknown"
    match = age_pattern.search(demographics_lower)
    if match:
        age = int(match.group(1))
        age_group = (
            "0-10" if age <= 10 else
            "10-20" if age <= 20 else
            "20-30" if age <= 30 else
            "30-40" if age <= 40 else
            "40-50" if age <= 50 else
            "50-60" if age <= 60 else
            "60+"
        )

    # Gender
    gender = "Other"
    if male_pattern.search(demographics_lower):
        gender = "Male"
    elif female_pattern.search(demographics_lower):
        gender = "Female"

    # Prepare text for LLM
    def clean_field(val):
        if isinstance(val, list):
            return " ".join(val)
        elif isinstance(val, dict):
            return " ".join(f"{k}: {v}" for k, v in val.items())
        return str(val)

    sh = clean_field(patient_info.get('Social_History', ''))
    pmh = clean_field(patient_info.get('Past_Medical_History', ''))
    hpi = clean_field(patient_info.get('History_of_Present_Illness', ''))

    try:
        llm_result = classify_bias_llm(demographics, sh, pmh, hpi)
    except Exception as e:
        print(f"LLM failure: {e}")
        llm_result = {k: "Unknown" for k in [
            "Smoking Status", "Alcohol Use", "Drug Use", "Occupation Type",
            "Comorbidity Status", "Symptom Presentation"
        ]}

    # Combine both LLM and regex-derived values
    all_categories = {
        "age_group": age_group,
        "gender": gender,
        "smoking_status": llm_result.get("Smoking Status", "Unknown"),
        "alcohol_use": llm_result.get("Alcohol Use", "Unknown"),
        "drug_use": llm_result.get("Drug Use", "Unknown"),
        "occupation_type": llm_result.get("Occupation Type", "Unknown"),
        "comorbidity_status": llm_result.get("Comorbidity Status", "Unknown"),
        "symptom_presentation": llm_result.get("Symptom Presentation", "Unknown")
    }

    return pd.Series(all_categories)

