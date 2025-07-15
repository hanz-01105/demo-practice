import re
import pandas as pd
import json

CATEGORY_RULES = {
    "smoking_status": [
        ("Non-smoker", ["non-smoker", "does not smoke", "never smoked", "quit smoking", "denies smoking"]),
        ("Smoker", ["smokes", "smoker", "currently smokes", "daily smoker"])
    ],
    "alcohol_use": [
        ("Non-drinker", ["does not drink", "non-drinker", "no alcohol", "denies drinking"]),
        ("Drinker", ["drinks", "alcohol use", "moderate alcohol", "drinks socially", "social drinker", "consumes alcohol"])
    ],
    "drug_use": [
        ("None", ["no drug use", "denies drug use", "no substance abuse", "not using drugs"]),
        ("Present", ["drug use", "uses drugs", "substance abuse", "positive for drugs"])
    ],
    "occupation_type": [
        ("Education", ["teacher", "professor", "educator", "school staff"]),
        ("Technical", ["engineer", "developer", "technician", "it specialist"]),
        ("Creative", ["designer", "artist", "writer", "musician", "photographer"]),
        ("Student", ["student", "college student", "high school student"]),
        ("Retired", ["retired", "no longer working", "former employee"]),
        ("Healthcare", ["nurse", "doctor", "physician", "paramedic", "healthcare worker"]),
        ("Service", ["waiter", "cashier", "barista", "customer service"])
    ],
    "comorbidity_status": [
        ("Chronic", [
            "diabetes", "hypertension", "asthma", "copd", "heart failure",
            "chronic kidney disease", "rheumatoid arthritis", "hiv", "hepatitis",
            "osteoporosis", "thyroid disease", "chronic liver disease"
        ]),
        ("Special Treatment", [
            "chemotherapy", "dialysis", "organ transplant", "immunosuppressive therapy",
            "radiation therapy", "biologic therapy", "transplant recipient"
        ]),
        ("None", [
            "no significant past medical", "no past medical history", "no pmhx",
            "healthy", "previously well", "no known medical conditions",
            "unremarkable medical history"
        ])
    ]
}


def match_category_from_keywords(text, rules):
    # Split text into segments on common conjunctions and punctuation
    parts = re.split(r'[.,;]| and | or ', text)
    parts = [part.strip() for part in parts if part.strip()]

    for label, keywords in rules:
        for kw in keywords:
            for part in parts:
                if kw in part:
                    return label
    return "Unknown"

def extract_gender(text):
    if re.search(r"\b(female|woman|girl)\b", text):
        return "Female"
    elif re.search(r"\b(male|man|boy)\b", text):
        return "Male"
    return "Other"

def categorize_scenario(patient_info):
    categories = {
        "age_group": "Unknown",
        "gender": "Unknown",
        "smoking_status": "Unknown",
        "alcohol_use": "Unknown",
        "drug_use": "Unknown",
        "occupation_type": "Unknown",
        "ses_proxy": "Unknown",
        "rare_medication": "Unknown",
        "family_support": "Unknown",
        "comorbidity_status": "Unknown",
        "symptom_presentation": "Unknown"
    }

    text = json.dumps(patient_info).lower()
    symptoms = patient_info.get("Symptoms", {})

    # Match rule-based categories
    for category, rules in CATEGORY_RULES.items():
        categories[category] = match_category_from_keywords(text, rules)

    # Age group
    if re.search(r"\bnewborn\b|\binfant\b", text):
        categories['age_group'] = "0-1"
    else:
        age_match = re.search(r"(\d+)[- ]?(year[- ]old)?", text)
        if age_match:
            age = int(age_match.group(1))
            categories['age_group'] = (
                "0-10" if age <= 10 else
                "10-20" if age <= 20 else
                "20-30" if age <= 30 else
                "30-40" if age <= 40 else
                "40-50" if age <= 50 else
                "50-60" if age <= 60 else "60+"
            )

    # Gender
    categories["gender"] = extract_gender(text)

    # Symptom Presentation
    total_symptoms = 0
    if "Primary_Symptom" in symptoms:
        total_symptoms += 1
    if "Secondary_Symptoms" in symptoms:
        secondary = symptoms["Secondary_Symptoms"]
        if isinstance(secondary, list):
            total_symptoms += len(secondary)
        elif isinstance(secondary, str):
            total_symptoms += 1

    if total_symptoms >= 3:
        categories["symptom_presentation"] = "Multiple Classic Symptoms"
    elif total_symptoms == 1:
        categories["symptom_presentation"] = "Single Symptom Only"
    elif total_symptoms > 1:
        categories["symptom_presentation"] = "Non-Specific Symptoms"
    else:
        categories["symptom_presentation"] = "Unknown"

    return pd.Series(categories)
