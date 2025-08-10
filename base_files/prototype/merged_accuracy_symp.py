import json
import os
import re
import pandas as pd
from typing import Dict, Any, List, Tuple
import numpy as np
from collections import defaultdict
from scipy import stats

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# --- Config ---
LOG_PATH = "base_files/logs"
# Point this to your extracted results file
LOG_FILE = "extracted_results_MedQA_Ext_corrected.json"

# ================================
# Symptom keyword definitions
# ================================
CLASSIC_KEYWORDS = [
    # Cardiovascular
    "chest pain", "palpitations", "syncope", "orthopnea", "paroxysmal nocturnal dyspnea",
    "claudication", "heart murmur", "bradycardia", "tachycardia",

    # Respiratory
    "shortness of breath", "dyspnea", "hemoptysis", "wheezing", "stridor", "pleuritic pain",
    "chest tightness", "productive cough", "dry cough",

    # Neurological
    "double vision", "ptosis", "seizure", "visual loss", "diplopia", "hemiparesis",
    "ataxia", "tremor", "dysarthria", "aphasia", "photophobia", "meningismus",

    # GI/GU
    "hematuria", "melena", "hematemesis", "jaundice", "ascites", "hepatomegaly",
    "splenomegaly", "rebound tenderness", "murphy's sign",

    # Infectious/Constitutional
    "fever", "rigors", "night sweats", "lymphadenopathy", "petechiae",

    # Dermatological
    "rash", "erythema", "purpura", "cyanosis", "clubbing", "koilonychia",

    # MSK
    "joint swelling", "morning stiffness", "bone pain", "muscle weakness"
]

VAGUE_KEYWORDS = [
    "vague", "weird", "unclear", "confused", "nonspecific", "funny feeling", "off",
    "just not right", "not myself", "weird sensation", "general malaise", "feeling unwell",
    "uncomfortable", "tired", "fatigue", "malaise", "unwell", "poorly", "run down",
    "not feeling well", "under the weather", "out of sorts", "lethargic"
]

MULTISYSTEM_KEYWORDS = {
    "neuro": [
        "headache", "dizziness", "numbness", "tingling", "weakness", "ptosis",
        "vision loss", "slurred speech", "seizure", "confusion", "gait instability",
        "tremor", "memory loss", "coordination problems", "balance issues", "vertigo",
        "altered mental status", "cognitive impairment"
    ],
    "gi": [
        "nausea", "vomiting", "diarrhea", "abdominal pain", "cramping", "constipation",
        "weight loss", "hematemesis", "melena", "bloating", "early satiety", "heartburn",
        "acid reflux", "bowel changes", "appetite loss", "dysphagia"
    ],
    "respiratory": [
        "cough", "dyspnea", "shortness of breath", "wheezing", "hemoptysis",
        "chest tightness", "pleuritic pain", "sputum production", "breathing difficulty"
    ],
    "cardiac": [
        "palpitations", "chest pain", "syncope", "orthopnea", "paroxysmal nocturnal dyspnea",
        "edema", "tachycardia", "bradycardia", "irregular heartbeat", "heart racing"
    ],
    "musculoskeletal": [
        "joint pain", "muscle pain", "swelling", "stiffness", "back pain",
        "difficulty walking", "limb weakness", "loss of coordination", "arthralgia", "myalgia"
    ],
    "dermatological": [
        "rash", "itching", "hives", "bruising", "skin discoloration", "dry skin",
        "lesion", "skin changes", "eczema", "dermatitis"
    ],
    "genitourinary": [
        "dysuria", "hematuria", "incontinence", "frequency", "urgency", "flank pain",
        "retention", "oliguria", "polyuria", "nocturia"
    ],
    "constitutional": [
        "fever", "chills", "night sweats", "weight loss", "weight gain", "fatigue",
        "malaise", "anorexia", "decreased appetite"
    ]
}

# ================================
# Utilities
# ================================
def load_and_process_log(filepath: str) -> pd.DataFrame:
    """
    Loads a JSON log file and normalizes into a DataFrame.
    Supports extracted-results format: {"results":[...]}.
    """
    print(f"Loading and processing log file: {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Log file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Formats:
    # 1) {"results":[ ... entries ... ]}
    # 2) [ ... entries ... ]
    # 3) {"logs":[ ... ]} (legacy)
    if isinstance(data, dict) and "results" in data:
        print("Extracted-results format detected")
        df = pd.json_normalize(data["results"])
    elif isinstance(data, list):
        print("New format detected (list of entries)")
        df = pd.json_normalize(data)
    elif isinstance(data, dict) and "logs" in data:
        print("Old format detected")
        df = pd.json_normalize(data["logs"])
    else:
        raise ValueError("Unexpected log file format")

    # Map diagnoses considered count if present in nested field
    if "consultation_analysis.diagnoses_considered_count" in df.columns:
        df["num_diagnoses_considered"] = df["consultation_analysis.diagnoses_considered_count"]
    elif "num_diagnoses_considered" not in df.columns:
        df["num_diagnoses_considered"] = 0

    # is_correct to numeric/bool
    if "is_correct" in df.columns:
        df["is_correct"] = pd.to_numeric(df["is_correct"], errors="coerce").fillna(0).astype(int)
    else:
        df["is_correct"] = 0

    return df

def extract_text_content(obj: Any, max_length: int = 10000) -> str:
    if isinstance(obj, str):
        return obj[:max_length]
    elif isinstance(obj, dict):
        texts = []
        for _, value in obj.items():
            if isinstance(value, (str, dict, list)):
                texts.append(extract_text_content(value, max_length))
        return ' '.join(texts)[:max_length]
    elif isinstance(obj, list):
        texts = []
        for item in obj:
            if isinstance(item, (str, dict, list)):
                texts.append(extract_text_content(item, max_length))
        return ' '.join(texts)[:max_length]
    else:
        return str(obj)

def is_meaningful_content(value: Any) -> bool:
    try:
        if pd.isna(value):
            return False
        if isinstance(value, (list, np.ndarray)):
            return len(value) > 0 and any(is_meaningful_content(item) for item in value)
        if isinstance(value, dict):
            return len(value) > 0
        if isinstance(value, str):
            return len(value.strip()) > 0
        return bool(value)
    except:
        return False

# ================================
# Symptom extraction + categorization
# ================================
def enhanced_extract_symptoms_from_log(row: pd.Series) -> Dict[str, Any]:
    extracted_text = ""
    extraction_strategies = [
        {
            'fields': ['dialogue_history'],
            'processor': lambda x: ' '.join([entry.get('text', '') for entry in x if isinstance(entry, dict) and 'text' in entry]) if isinstance(x, list) else ''
        },
        {
            'fields': ['correct_diagnosis', 'final_doctor_diagnosis', 'diagnosis'],
            'processor': lambda x: str(x) if x else ''
        },
        {
            'fields': ['symptoms', 'Symptoms', 'symptom_data', 'chief_complaint', 'presenting_complaint'],
            'processor': lambda x: extract_text_content(x) if is_meaningful_content(x) else ''
        },
        {
            'fields': ['scenario', 'case_description', 'patient_history', 'description', 'specialist_reason'],
            'processor': lambda x: str(x) if x else ''
        }
    ]
    for strategy in extraction_strategies:
        for field in strategy['fields']:
            if field in row and is_meaningful_content(row[field]):
                try:
                    text = strategy['processor'](row[field])
                    if text and len(text.strip()) > 10:
                        if isinstance(row[field], dict) and 'Primary_Symptom' in row[field]:
                            return row[field]
                        extracted_text = text.strip()
                        break
                except Exception as e:
                    print(f"Warning: Error processing field {field}: {e}")
                    continue
        if extracted_text:
            break
    if not extracted_text:
        try:
            row_dict = row.to_dict()
            extracted_text = extract_text_content(row_dict, max_length=5000)
        except Exception as e:
            print(f"Warning: Error in fallback extraction: {e}")
            extracted_text = "Unknown presentation"

    return {
        'Primary_Symptom': extracted_text or 'Unknown presentation',
        'Secondary_Symptoms': [],
        'extraction_source': 'enhanced_extraction'
    }

def analyze_symptom_keywords(text: str) -> Dict[str, Any]:
    tl = text.lower()
    classic_matches = [kw for kw in CLASSIC_KEYWORDS if kw in tl]
    vague_matches = [kw for kw in VAGUE_KEYWORDS if kw in tl]
    system_matches = {}
    for system, keywords in MULTISYSTEM_KEYWORDS.items():
        matches = [kw for kw in keywords if kw in tl]
        if matches:
            system_matches[system] = matches
    return {
        'classic_keywords': classic_matches,
        'vague_keywords': vague_matches,
        'system_keywords': system_matches,
        'systems_involved': list(system_matches.keys()),
        'total_keywords': len(classic_matches) + len(vague_matches) + sum(len(m) for m in system_matches.values())
    }

def enhanced_categorize_symptom_presentation(symptom_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    if not isinstance(symptom_data, dict) or not symptom_data:
        return 'Unclassified', {'reason': 'Invalid symptom data'}

    primary = symptom_data.get('Primary_Symptom', '')
    secondary = symptom_data.get('Secondary_Symptoms', [])
    if isinstance(secondary, str):
        secondary = [secondary]
    elif not isinstance(secondary, list):
        secondary = []

    text = ' '.join([str(s) for s in [primary] + secondary if s]).strip()
    if not text or text == 'Unknown presentation':
        return 'Unclassified', {'reason': 'No symptom text available'}

    ka = analyze_symptom_keywords(text)
    info = {'text_analyzed': text[:200] + '...' if len(text) > 200 else text, 'keyword_analysis': ka}

    if ka['classic_keywords']:
        return 'Classic Textbook', {**info, 'reason': f"Classic keywords found: {ka['classic_keywords'][:3]}", 'confidence': 'high'}
    if len(ka['systems_involved']) >= 2:
        return 'Multi-System Complex', {**info, 'reason': f"Multiple systems involved: {ka['systems_involved']}", 'confidence': 'high'}
    if ka['vague_keywords']:
        return 'Atypical/Vague Wording', {**info, 'reason': f"Vague language detected: {ka['vague_keywords'][:3]}", 'confidence': 'high'}
    if (len(ka['systems_involved']) == 1 and len(ka['system_keywords'][ka['systems_involved'][0]]) == 1):
        return 'Single Symptom Only', {**info, 'reason': f"Single symptom from {ka['systems_involved'][0]} system", 'confidence': 'medium'}
    if len(ka['systems_involved']) == 1:
        return 'Single System Multiple Symptoms', {**info, 'reason': f"Multiple symptoms from {ka['systems_involved'][0]} system", 'confidence': 'medium'}
    return 'Atypical/Vague Wording', {**info, 'reason': 'No clear pattern identified - defaulting to atypical', 'confidence': 'low'}

# ================================
# Metrics + analysis
# ================================
def calculate_enhanced_performance_metrics(df: pd.DataFrame, group_col: str, actual_col: str = 'is_correct') -> pd.DataFrame:
    if group_col not in df.columns:
        raise KeyError(f"Grouping column '{group_col}' not found in the DataFrame.")
    if actual_col not in df.columns:
        raise KeyError(f"Actual outcome column '{actual_col}' not found in the DataFrame.")

    df_clean = df.copy()
    df_clean[actual_col] = pd.to_numeric(df_clean[actual_col], errors='coerce').astype('boolean')
    valid_df = df_clean.dropna(subset=[actual_col, group_col])

    def calculate_metrics(group_data):
        correct_cases = group_data[actual_col].sum()
        total_cases = len(group_data)
        if total_cases == 0:
            return pd.Series({
                'Accuracy_%': 0, 'Total_Cases': 0, 'Correct_Cases': 0, 'Incorrect_Cases': 0,
                'Confidence_Interval_95%': 'N/A', 'Avg_Diagnoses_Considered': 0
            })
        accuracy = (correct_cases / total_cases) * 100
        p = correct_cases / total_cases
        margin_error = 1.96 * np.sqrt((p * (1 - p)) / total_cases) if total_cases > 0 else 0
        ci_lower = max(0, (p - margin_error) * 100)
        ci_upper = min(100, (p + margin_error) * 100)
        avg_diagnoses = group_data['num_diagnoses_considered'].mean() if 'num_diagnoses_considered' in group_data.columns else 0
        return pd.Series({
            'Accuracy_%': accuracy,
            'Total_Cases': total_cases,
            'Correct_Cases': correct_cases,
            'Incorrect_Cases': total_cases - correct_cases,
            'Confidence_Interval_95%': f"({ci_lower:.1f}%-{ci_upper:.1f}%)",
            'Avg_Diagnoses_Considered': avg_diagnoses
        })

    grouped_metrics = valid_df.groupby(group_col, observed=True).apply(calculate_metrics, include_groups=False).reset_index()
    return grouped_metrics.sort_values(by='Accuracy_%', ascending=False)

def analyze_diagnostic_complexity_enhanced(df: pd.DataFrame):
    print("\n" + "="*70)
    print("ENHANCED DIAGNOSTIC COMPLEXITY ANALYSIS")
    print("="*70)

    if 'num_diagnoses_considered' not in df.columns or df['num_diagnoses_considered'].sum() == 0:
        print("No diagnoses considered data available for complexity analysis.")
        return

    complexity_stats = df['num_diagnoses_considered'].describe()
    print(f"\nDiagnostic Complexity Statistics:")
    print(f"  Mean: {complexity_stats['mean']:.2f}")
    print(f"  Median: {complexity_stats['50%']:.1f}")
    print(f"  Std Dev: {complexity_stats['std']:.2f}")
    print(f"  Range: {complexity_stats['min']:.0f} - {complexity_stats['max']:.0f}")

    df['diagnostic_complexity'] = pd.cut(
        df['num_diagnoses_considered'],
        bins=[0, 2, 4, 6, float('inf')],
        labels=["Simple (1-2)", "Moderate (3-4)", "Complex (5-6)", "Very Complex (7+)"],
        include_lowest=True
    )

    print(f"\nComplexity Distribution:")
    complexity_dist = df['diagnostic_complexity'].value_counts()
    for category, count in complexity_dist.items():
        percentage = (count / len(df)) * 100
        avg_accuracy = df[df['diagnostic_complexity'] == category]['is_correct'].mean() * 100
        print(f"  {category}: {count} cases ({percentage:.1f}%) - Avg Accuracy: {avg_accuracy:.1f}%")

    print(f"\nComplexity Analysis by Symptom Presentation:")
    for symptom_category in df['symptom_presentation'].unique():
        if symptom_category in ['Unclassified']:
            continue
        category_data = df[df['symptom_presentation'] == symptom_category].copy()
        if len(category_data) < 3:
            continue
        print(f"\n--- {symptom_category} ({len(category_data)} cases) ---")
        try:
            complexity_metrics = calculate_enhanced_performance_metrics(category_data, 'diagnostic_complexity')
            for _, row in complexity_metrics.iterrows():
                complexity = row['diagnostic_complexity']
                accuracy = row['Accuracy_%']
                total = row['Total_Cases']
                ci = row['Confidence_Interval_95%']
                if total >= 2:
                    print(f"  {complexity}: {accuracy:.1f}% accuracy ({total} cases) {ci}")
        except Exception as e:
            print(f"  Error analyzing {symptom_category}: {e}")

def generate_summary_insights(df: pd.DataFrame):
    print("\n" + "="*70)
    print("KEY INSIGHTS & RECOMMENDATIONS")
    print("="*70)

    category_performance = df.groupby('symptom_presentation')['is_correct'].agg(['mean', 'count']).reset_index()
    category_performance['accuracy'] = category_performance['mean'] * 100
    category_performance = category_performance.sort_values('accuracy', ascending=False)

    print("\n Performance Rankings:")
    for _, row in category_performance.iterrows():
        if row['count'] >= 3:
            print(f"  {row['symptom_presentation']}: {row['accuracy']:.1f}% ({row['count']} cases)")

    if 'num_diagnoses_considered' in df.columns:
        complexity_corr = df[['num_diagnoses_considered', 'is_correct']].corr().iloc[0, 1]
        print(f"\n Complexity Correlation:")
        print(f"  Correlation between diagnostic complexity and accuracy: {complexity_corr:.3f}")
        if complexity_corr > 0.1:
            print("  → More complex cases tend to have slightly better outcomes")
        elif complexity_corr < -0.1:
            print("  → More complex cases tend to have worse outcomes")
        else:
            print("  → No strong correlation between complexity and outcomes")

    best_category = category_performance.iloc[0]
    worst_category = category_performance.iloc[-1]
    print(f"\n Recommendations:")
    print(f"  1. Focus improvement efforts on '{worst_category['symptom_presentation']}' cases")
    print(f"     (Currently {worst_category['accuracy']:.1f}% accuracy)")
    if best_category['accuracy'] > 80:
        print(f"  2. Study successful patterns from '{best_category['symptom_presentation']}' cases")
        print(f"     (Achieving {best_category['accuracy']:.1f}% accuracy)")
    if 'num_diagnoses_considered' in df.columns:
        high_complexity = df[df['num_diagnoses_considered'] >= 6]
        if len(high_complexity) > 10:
            high_complex_accuracy = high_complexity['is_correct'].mean() * 100
            print(f"  3. Review high-complexity cases (6+ diagnoses considered)")
            print(f"     ({len(high_complexity)} cases with {high_complex_accuracy:.1f}% accuracy)")

# ================================
# Confidence & demographic parity
# ================================
def calculate_confidence_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    results = []
    for group_name in df[group_col].unique():
        if group_name == 'Unclassified':
            continue
        group_data = df[df[group_col] == group_name].copy()
        if len(group_data) == 0:
            continue
        total_cases = len(group_data)
        correct_cases = group_data['is_correct'].sum()
        accuracy = (correct_cases / total_cases) * 100 if total_cases > 0 else 0

        metrics = {'avg_diagnoses_considered': None, 'confidence_proxy': None, 'calibration_error': None}
        if 'num_diagnoses_considered' in group_data.columns:
            avg_diagnoses = group_data['num_diagnoses_considered'].mean()
            metrics['avg_diagnoses_considered'] = avg_diagnoses
            metrics['confidence_proxy'] = max(0, (10 - avg_diagnoses) / 10 * 100)
            expected = metrics['confidence_proxy']
            metrics['calibration_error'] = abs(expected - accuracy)

        results.append({
            'Group': group_name,
            'Total_Cases': total_cases,
            'Correct_Cases': correct_cases,
            'Accuracy_%': accuracy,
            **metrics
        })
    return pd.DataFrame(results).sort_values('Accuracy_%', ascending=False)

def calculate_demographic_parity(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    results = []
    overall_accuracy = df['is_correct'].mean() * 100
    for group_name in df[group_col].unique():
        if group_name == 'Unclassified':
            continue
        group_data = df[df[group_col] == group_name].copy()
        if len(group_data) == 0:
            continue
        total_cases = len(group_data)
        accuracy = (group_data['is_correct'].sum() / total_cases) * 100
        parity_gap = overall_accuracy - accuracy  # overall − group
        representation = (total_cases / len(df)) * 100

        if total_cases >= 5:
            group_correct = group_data['is_correct'].sum()
            overall_correct = df['is_correct'].sum()
            overall_total = len(df)
            p1 = group_correct / total_cases
            p2 = overall_correct / overall_total
            pooled_p = (group_correct + overall_correct) / (total_cases + overall_total)
            se = np.sqrt(pooled_p * (1 - pooled_p) * (1/total_cases + 1/overall_total)) if (total_cases + overall_total) > 0 else 0
            if se > 0:
                z_score = (p1 - p2) / se
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                is_significant = p_value < 0.05
            else:
                z_score = 0
                p_value = 1.0
                is_significant = False
        else:
            z_score = 0
            p_value = 1.0
            is_significant = False

        results.append({
            'Group': group_name,
            'Total_Cases': total_cases,
            'Accuracy_%': accuracy,
            'Parity_Gap_%': parity_gap,         # overall − group
            'Representation_%': representation,
            'Z_Score': z_score,
            'P_Value': p_value,
            'Statistically_Significant': is_significant
        })
    return pd.DataFrame(results).sort_values('Parity_Gap_%', ascending=False)

def print_combined_analysis(df: pd.DataFrame, group_col: str, expected_groups: List[str]):
    """Confidence proxy & demographic parity (overall − group)."""
    print("\n" + "="*100)
    print(f"SYMPTOM PRESENTATION - CONFIDENCE RATING & DEMOGRAPHIC PARITY")
    print("="*100)

    overall_accuracy_pct = df['is_correct'].mean() * 100
    if 'num_diagnoses_considered' in df.columns and len(df) > 0:
        overall_conf_proxy_pct = max(0.0, (10 - df['num_diagnoses_considered'].mean()) / 10 * 100)
    else:
        overall_conf_proxy_pct = 0.0

    print(f"\nOverall Baseline Accuracy: {overall_accuracy_pct:.1f}%")
    print(f"Total Cases: {len(df)}\n")

    print("-" * 100)
    print(f"{'Group':<25} {'Cases':<6} {'Accuracy':<9} {'Confidence':<11} {'Conf Diff':<11} {'Parity Gap':<11} {'Repr %':<7} {'Status':<15}")
    print("-" * 100)

    rows = []
    for group_name in expected_groups:
        group_data = df[df[group_col] == group_name]
        cases = len(group_data)

        if cases == 0:
            print(f"{group_name:<25} {0:<6} {'N/A':<9} {'N/A':<11} {'N/A':<11} {'N/A':<11} {'N/A':<7} {'No Data':<15}")
            continue

        acc_pct = group_data['is_correct'].mean() * 100
        conf_proxy_pct = max(0.0, (10 - group_data['num_diagnoses_considered'].mean()) / 10 * 100) if cases > 0 else 0.0

        parity_gap = overall_accuracy_pct - acc_pct        # overall − group
        conf_diff = overall_conf_proxy_pct - conf_proxy_pct # overall − group
        repr_pct = (cases / len(df)) * 100

        if cases < 5:
            status = 'Low Sample'
        elif abs(parity_gap) > 10:
            status = 'High Disparity'
        elif acc_pct > overall_accuracy_pct + 5:
            status = 'Above Average'
        elif acc_pct < overall_accuracy_pct - 5:
            status = 'Below Average'
        else:
            status = 'Balanced'

        print(f"{group_name:<25} {cases:<6} {acc_pct:>6.1f} % {conf_proxy_pct:>6.1f} % {conf_diff:+6.1f} % {parity_gap:+6.1f} % {repr_pct:>5.1f} % {status:<15}")

        rows.append({
            'Group': group_name,
            'Cases': cases,
            'Accuracy_%': acc_pct,
            'Confidence_Proxy_%': conf_proxy_pct,
            'Confidence_Diff_%': conf_diff,
            'Parity_Gap_%': parity_gap,
            'Representation_%': repr_pct,
            'Status': status
        })

    print("-" * 100)

    print("\n" + "="*60)
    print("DETAILED CONFIDENCE ANALYSIS")
    print("="*60)
    for r in rows:
        print(f"\n{r['Group']}:")
        print(f"  Cases: {r['Cases']}")
        print(f"  Accuracy: {r['Accuracy_%']:.1f}%")
        print(f"  Confidence Proxy: {r['Confidence_Proxy_%']:.1f}%")
        print(f"  Confidence Difference (overall − group): {r['Confidence_Diff_%']:+.1f}%")

    print("\n" + "="*60)
    print("DETAILED DEMOGRAPHIC PARITY ANALYSIS")
    print("="*60)
    for r in rows:
        print(f"\n{r['Group']}:")
        print(f"  Cases: {r['Cases']} ({r['Representation_%']:.1f}% of dataset)")
        print(f"  Accuracy: {r['Accuracy_%']:.1f}%")
        print(f"  Parity Gap (overall − group): {r['Parity_Gap_%']:+.1f}%")

    # Quick insights
    valid = [r for r in rows if r['Cases'] > 0]
    if valid:
        print("\n" + "="*60)
        print("KEY FINDINGS & RECOMMENDATIONS")
        print("="*60)
        highest_conf = max(valid, key=lambda x: x['Confidence_Proxy_%'])
        lowest_conf = min(valid, key=lambda x: x['Confidence_Proxy_%'])
        print("\nConfidence Insights:")
        print(f"  • Highest Confidence: {highest_conf['Group']} ({highest_conf['Confidence_Proxy_%']:.1f}%)")
        print(f"  • Lowest Confidence: {lowest_conf['Group']} ({lowest_conf['Confidence_Proxy_%']:.1f}%)")
        best_advantage = min(valid, key=lambda x: x['Parity_Gap_%'])  # most negative (group > overall)
        print("\nParity Insights:")
        print(f"  • Performance Advantage: {best_advantage['Group']} ({best_advantage['Parity_Gap_%']:+.1f}%)")
        underrep = [r for r in valid if r['Representation_%'] < 5]
        if underrep:
            print("\nRepresentation Concerns:")
            print(f"  • {len(underrep)} group(s) are underrepresented (< 5% of data)")
            for r in underrep:
                print(f"    - {r['Group']}: {r['Representation_%']:.1f}% ({r['Cases']} cases)")

    print("\nMethodology Notes:")
    print("  • Confidence Proxy: Inverse of diagnostic complexity (fewer diagnoses = higher confidence)")
    print("  • Parity Gap & Confidence Difference are (overall − group)")
    print("  • Statistical significance via two-proportion z-test (shown in prior section)")

# ================================
# Main
# ================================
if __name__ == "__main__":
    try:
        log_filepath = os.path.join(LOG_PATH, LOG_FILE)
        processed_df = load_and_process_log(log_filepath)

        print(f" Diagnoses considered data available (avg: {processed_df['num_diagnoses_considered'].mean():.1f})")

        # Overall summary
        total_cases = len(processed_df)
        correct_cases = processed_df['is_correct'].sum()
        overall_accuracy = (correct_cases / total_cases) * 100 if total_cases else 0
        avg_diagnoses = processed_df['num_diagnoses_considered'].mean() if total_cases else 0

        print("\n" + "="*60)
        print("ENHANCED BIAS-CORRECTED PERFORMANCE ANALYSIS")
        print("="*60)
        print(f" Dataset Overview:")
        print(f"   Total Cases: {total_cases}")
        print(f"   Correct Diagnoses: {correct_cases}")
        print(f"   Overall Accuracy: {overall_accuracy:.1f}%")
        print(f"   Average Diagnoses Considered: {avg_diagnoses:.1f}")

        # Symptom extraction + categorization
        print("\n Extracting and categorizing symptoms...")
        processed_df['symptoms_raw'] = processed_df.apply(enhanced_extract_symptoms_from_log, axis=1)
        cat_results = processed_df['symptoms_raw'].apply(enhanced_categorize_symptom_presentation)
        processed_df['symptom_presentation'] = [r[0] for r in cat_results]
        processed_df['categorization_details'] = [r[1] for r in cat_results]
        print(" Enhanced symptom categorization complete.")

        # Distribution
        print("\n" + "="*60)
        print("SYMPTOM PRESENTATION DISTRIBUTION")
        print("="*60)
        category_counts = processed_df['symptom_presentation'].value_counts()
        for category, count in category_counts.items():
            pct = (count / len(processed_df)) * 100
            avg_diag = processed_df[processed_df['symptom_presentation'] == category]['num_diagnoses_considered'].mean()
            avg_acc = processed_df[processed_df['symptom_presentation'] == category]['is_correct'].mean() * 100
            print(f"  {category}: {count} cases ({pct:.1f}%)")
            print(f"    └─ Avg accuracy: {avg_acc:.1f}%, Avg diagnoses: {avg_diag:.1f}")

        # Accuracy by category
        print("\n" + "="*60)
        print("ENHANCED ACCURACY ANALYSIS BY SYMPTOM PRESENTATION")
        print("="*60)
        perf_df = calculate_enhanced_performance_metrics(processed_df, 'symptom_presentation', 'is_correct')
        for _, row in perf_df.iterrows():
            category = row['symptom_presentation']
            acc = row['Accuracy_%']
            correct = row['Correct_Cases']
            total = row['Total_Cases']
            ci = row['Confidence_Interval_95%']
            avg_diag = row['Avg_Diagnoses_Considered']
            print(f"\n--- {category} ---")
            print(f"   Accuracy: {acc:.1f}% {ci}")
            print(f"   Cases: {correct}/{total} correct")
            if avg_diag > 0:
                print(f"   Avg Diagnoses Considered: {avg_diag:.1f}")

        # Complexity, insights
        analyze_diagnostic_complexity_enhanced(processed_df)
        generate_summary_insights(processed_df)

        # Confidence + parity (single, corrected table)
        presentation_groups = [
            "Classic Textbook",
            "Atypical/Vague Wording",
            "Multi-System Complex",
            "Single Symptom Only"
        ]
        print_combined_analysis(processed_df, group_col='symptom_presentation', expected_groups=presentation_groups)

        # Examples
        print("\n" + "="*60)
        print("SAMPLE DIAGNOSTIC REASONING")
        print("="*60)
        for category in processed_df['symptom_presentation'].unique():
            if category == 'Unclassified':
                continue
            examples = processed_df[processed_df['symptom_presentation'] == category].head(2)
            print(f"\n--- {category} Examples ---")
            for _, row in examples.iterrows():
                details = row['categorization_details']
                status = 'CORRECT' if row['is_correct'] else 'INCORRECT'
                print(f"  Scenario {row.get('scenario_id', 'N/A')} ({status}):")
                print(f"    Reason: {details.get('reason', 'N/A')}")
                if 'confidence' in details:
                    print(f"    Confidence: {details['confidence']}")

        print("\n" + "="*60)
        print(" ENHANCED ANALYSIS COMPLETE")
        print(f" Overall Performance: {overall_accuracy:.1f}% accuracy")
        print(f" Average Complexity: {avg_diagnoses:.1f} diagnoses per case")
        print(f" Total Categories: {len(category_counts)} symptom presentations identified")
        print("="*60)

    except Exception as e:
        print(f"\n An error occurred during processing: {e}")
        import traceback
        traceback.print_exc()

# Main execution block
if __name__ == "__main__":
    try:
        log_filepath = os.path.join(LOG_PATH, LOG_FILE)
        processed_df = load_and_process_log(log_filepath)

        # Enhanced data preparation
        if "consultation_analysis.diagnoses_considered_count" in processed_df.columns:
            processed_df['num_diagnoses_considered'] = processed_df["consultation_analysis.diagnoses_considered_count"]
            print(f" Diagnoses considered data available (avg: {processed_df['num_diagnoses_considered'].mean():.1f})")
        else:
            print("  Warning: No diagnoses considered count found")
            processed_df['num_diagnoses_considered'] = 0

        # Overall performance summary
        total_cases = len(processed_df)
        correct_cases = processed_df['is_correct'].sum()
        overall_accuracy = (correct_cases / total_cases) * 100
        avg_diagnoses = processed_df['num_diagnoses_considered'].mean()
        
        print("\n" + "="*60)
        print("ENHANCED BIAS-CORRECTED PERFORMANCE ANALYSIS")
        print("="*60)
        print(f" Dataset Overview:")
        print(f"   Total Cases: {total_cases}")
        print(f"   Correct Diagnoses: {correct_cases}")
        print(f"   Overall Accuracy: {overall_accuracy:.1f}%")
        print(f"   Average Diagnoses Considered: {avg_diagnoses:.1f}")

        # Enhanced symptom extraction and categorization
        print("\n Extracting and categorizing symptoms...")
        processed_df['symptoms_raw'] = processed_df.apply(enhanced_extract_symptoms_from_log, axis=1)
        
        # Apply enhanced categorization
        categorization_results = processed_df['symptoms_raw'].apply(enhanced_categorize_symptom_presentation)
        processed_df['symptom_presentation'] = [result[0] for result in categorization_results]
        processed_df['categorization_details'] = [result[1] for result in categorization_results]
        
        print(" Enhanced symptom categorization complete.")

        # Distribution analysis
        print("\n" + "="*60)
        print("SYMPTOM PRESENTATION DISTRIBUTION")
        print("="*60)
        category_counts = processed_df['symptom_presentation'].value_counts()
        for category, count in category_counts.items():
            percentage = (count / len(processed_df)) * 100
            avg_diag_for_category = processed_df[processed_df['symptom_presentation'] == category]['num_diagnoses_considered'].mean()
            avg_accuracy = processed_df[processed_df['symptom_presentation'] == category]['is_correct'].mean() * 100
            print(f"  {category}: {count} cases ({percentage:.1f}%)")
            print(f"    └─ Avg accuracy: {avg_accuracy:.1f}%, Avg diagnoses: {avg_diag_for_category:.1f}")
        
        # Enhanced performance metrics
        print("\n" + "="*60)
        print("ENHANCED ACCURACY ANALYSIS BY SYMPTOM PRESENTATION")
        print("="*60)
        
        performance_df = calculate_enhanced_performance_metrics(
            processed_df,
            group_col='symptom_presentation',
            actual_col='is_correct'
        )

        # Display results with confidence intervals
        for index, row in performance_df.iterrows():
            category = row['symptom_presentation']
            accuracy = row['Accuracy_%']
            correct = row['Correct_Cases']
            total = row['Total_Cases']
            ci = row['Confidence_Interval_95%']
            avg_diag = row['Avg_Diagnoses_Considered']

            print(f"\n--- {category} ---")
            print(f"   Accuracy: {accuracy:.1f}% {ci}")
            print(f"   Cases: {correct}/{total} correct")
            if avg_diag > 0:
                print(f"   Avg Diagnoses Considered: {avg_diag:.1f}")
        
        # Enhanced complexity analysis
        analyze_diagnostic_complexity_enhanced(processed_df)
        
        # Generate insights and recommendations
        generate_summary_insights(processed_df)

        # === SELF CONFIDENCE + DEMOGRAPHIC PARITY for Symptom Presentation ===
        if "symptom_presentation" in processed_df.columns:
            # Define all the expected presentation categories
            presentation_groups = [
                "Classic Textbook",
                "Atypical/Vague Wording",
                "Multi-System Complex", 
                "Single Symptom Only"
            ]

            # Ensure the column exists and is clean
            processed_df["symptom_presentation"] = processed_df["symptom_presentation"].astype(str)

            # Call the function to display confidence + parity
            print_combined_analysis(
                df=processed_df,
                group_col="symptom_presentation",
                expected_groups=presentation_groups
            )

        # Sample diagnostic reasoning (improved)
        print("\n" + "="*60)
        print("SAMPLE DIAGNOSTIC REASONING")
        print("="*60)
        
        # Show examples from each category
        for category in processed_df['symptom_presentation'].unique():
            if category == 'Unclassified':
                continue
            category_examples = processed_df[processed_df['symptom_presentation'] == category].head(2)
            
            print(f"\n--- {category} Examples ---")
            for _, row in category_examples.iterrows():
                details = row['categorization_details']
                status = 'CORRECT' if row['is_correct'] else 'INCORRECT'
                print(f"  Scenario {row.get('scenario_id', 'N/A')} ({status}):")
                print(f"    Reason: {details.get('reason', 'N/A')}")
                if 'confidence' in details:
                    print(f"    Confidence: {details['confidence']}")
            
        print("\n" + "="*60)
        print(" ENHANCED ANALYSIS COMPLETE")
        print(f" Overall Performance: {overall_accuracy:.1f}% accuracy")
        print(f" Average Complexity: {avg_diagnoses:.1f} diagnoses per case")
        print(f" Total Categories: {len(category_counts)} symptom presentations identified")
        print("="*60)

    except Exception as e:
        print(f"\n An error occurred during processing: {e}")
        import traceback
        traceback.print_exc()
