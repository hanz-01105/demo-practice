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
- "Classic Textbook": The patient's symptoms clearly match textbook descriptions for a known condition. (e.g., crushing chest pain radiating to the arm for myocardial infarction)
- "Atypical/Vague Wording": The symptoms are non-specific, vague, or don't align with common textbook patterns. (e.g., "just feeling off", or mild fatigue without context)
- "Multi-System Complex": Multiple unrelated systems are involved or symptoms span several organ systems. (e.g., neurological, gastrointestinal, and dermatologic symptoms all at once)
- "Single Symptom Only": Only one specific symptom is reported without significant elaboration. (e.g., only "headache" or only "back pain")

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
- Comorbidity Status [Hypertension, Diabetes Mellitus, Cancer, Other, None, Unknown]
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
    raw = re.sub(r"```[a-z]*|```", "", raw)
    return json.loads(raw)

def categorize_patient(demo_text, social_text, pmh_text, history_text):
    demographics_lower = demo_text.lower()
    age_group = "Unknown"
    gender = "Other"

    # Age
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

    # Gender
    if male_pattern.search(demographics_lower):
        gender = "Male"
    elif female_pattern.search(demographics_lower):
        gender = "Female"

    # LLM categories
    llm = call_llm_classifier(demo_text, social_text, pmh_text, history_text)

    return {
        "age_group": age_group,
        "gender": gender,
        **llm
    }


