import re
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI()

male_pattern = re.compile(r"\b(male|man|boy)\b", re.IGNORECASE)
female_pattern = re.compile(r"\b(female|woman|girl)\b", re.IGNORECASE)
age_pattern = re.compile(r"(\d+)[-\s]*year", re.IGNORECASE)

def call_llm_classifier(demographics, social_history, past_medical_history, history):
    prompt = f"""
You are a careful medical data classifier.

Classify each variable based on the patient data below. If no info, return "Unknown".

--- Symptom Presentation Category Guide ---
- "Classic Textbook": Clear textbook-matching symptoms (e.g., chest pain radiating to arm for MI).
- "Atypical/Vague Wording": Non-specific or vague symptoms not clearly matching textbook patterns.
- "Multi-System Complex": Symptoms span multiple unrelated organ systems.
- "Single Symptom Only": Only one isolated symptom reported.

--- Input ---
Demographics: "{demographics}"
Social_History: "{social_history}"
Past_Medical_History: "{past_medical_history}"
History_of_Present_Illness: "{history}"

Return strict JSON with these keys:
- Smoking Status [Smoker, Non-smoker, Unknown]
- Alcohol Use [Drinker, Non-drinker, Unknown]
- Drug Use [Drug User, Non-drug User, Unknown]
- Occupation Type [Manual Labor, Knowledge Worker, Student, Retired, Unemployed, Unknown]
- Comorbidity Status [Chronic Condition Present, Immunosuppressed/Special Treatment, None]
- Symptom Presentation [Classic Textbook, Atypical/Vague Wording, Multi-System Complex, Single Symptom Only, Unknown]
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a careful medical data classifier."},
            {"role": "user", "content": prompt}
        ]
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"```[a-z]*|```", "", raw)  # Remove any accidental markdown
    return json.loads(raw)

def categorize_patient(demo_dict):
    """
    Categorizes a patient based on demographics dictionary from each scenario log.
    """
    demo_text = demo_dict.get("Demographics", "")
    social_text = demo_dict.get("Social_History", "")
    pmh_text = demo_dict.get("Past_Medical_History", "")
    history_text = demo_dict.get("History", "")

    demographics_lower = str(demo_text).lower()
    age_group = "Unknown"
    gender = "Other"

    age = None
    if any(x in demographics_lower for x in ["month", "newborn", "infant"]):
        age = 0
    else:
        match = age_pattern.search(demographics_lower)
        if match:
            age = int(match.group(1))

    if age is not None:
        age_group = (
            "0-10" if age <= 10 else
            "10-20" if age <= 20 else
            "20-30" if age <= 30 else
            "30-40" if age <= 40 else
            "40-50" if age <= 50 else
            "50-60" if age <= 60 else "60+"
        )


    if male_pattern.search(demographics_lower):
        gender = "Male"
    elif female_pattern.search(demographics_lower):
        gender = "Female"


    llm = call_llm_classifier(demo_text, social_text, pmh_text, history_text)

    return {
        "age_group": age_group,
        "gender": gender,
        **llm
    }

def run_categorization(input_json="base_files/logs/medqa_run_20250803_001542.json", output_csv="categorized_patients.csv"):
    with open(input_json, "r") as f:
        data = json.load(f)

    results = data["results"]
    categorized_records = []

    for entry in results:
        demographics = entry.get("demographics", {})
        categorization = categorize_patient(demographics)
        categorization["scenario_id"] = entry["scenario_id"]
        categorized_records.append(categorization)

    df = pd.DataFrame(categorized_records)
    df.to_csv(output_csv, index=False)
    print(f"Categorized data saved to {output_csv}")

if __name__ == "__main__":
    run_categorization()
