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

# --- Enhanced Constants ---
LOG_PATH = "base_files/logs"
LOG_FILE = "extracted_results_MedQA_Ext_corrected.json"

# Improved symptom categorization with more comprehensive keywords
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
    
    # GI/GU specific
    "hematuria", "melena", "hematemesis", "jaundice", "ascites", "hepatomegaly",
    "splenomegaly", "rebound tenderness", "murphy's sign",
    
    # Infectious/Constitutional
    "fever", "rigors", "night sweats", "lymphadenopathy", "petechiae",
    
    # Dermatological
    "rash", "erythema", "purpura", "cyanosis", "clubbing", "koilonychia",
    
    # MSK specific
    "joint swelling", "morning stiffness", "bone pain", "muscle weakness"
]

VAGUE_KEYWORDS = [
    "vague", "weird", "unclear", "confused", "nonspecific", "funny feeling", "off", 
    "just not right", "not myself", "weird sensation", "general malaise", "feeling unwell", 
    "uncomfortable", "tired", "fatigue", "malaise", "unwell", "poorly", "run down",
    "not feeling well", "under the weather", "out of sorts", "lethargic"
]

# More granular system classification
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

def load_and_process_log(filepath: str) -> pd.DataFrame:
    """
    Loads a JSON log file and normalizes into a DataFrame with enhanced error handling.
    """
    print(f"Loading and processing log file: {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Log file not found: {filepath}")

    try:
        with open(filepath, "r", encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {e}")

    # Handle both old and new log file formats
    if isinstance(data, dict) and "logs" in data:
        df = pd.json_normalize(data["logs"])
        print("Old format detected")
    elif isinstance(data, list):
        df = pd.json_normalize(data)
        print("New format detected (bias-corrected results)")
    else:
        raise ValueError("Unexpected log file format")

    print(f"Successfully loaded log. Shape: {df.shape}")
    
    # Validate required columns
    required_cols = ['is_correct']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing required columns: {missing_cols}")
    
    return df

def extract_text_content(obj: Any, max_length: int = 10000) -> str:
    """
    Recursively extracts text content from nested JSON structures.
    """
    if isinstance(obj, str):
        return obj[:max_length]
    elif isinstance(obj, dict):
        texts = []
        for key, value in obj.items():
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
    """
    Safely check if a value contains meaningful content.
    """
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

def enhanced_extract_symptoms_from_log(row: pd.Series) -> Dict[str, Any]:
    """
    Enhanced symptom extraction with better text processing and fallback strategies.
    """
    extracted_text = ""
    
    # Priority order for extraction
    extraction_strategies = [
        # Strategy 1: Dialogue history (most reliable for bias-corrected format)
        {
            'fields': ['dialogue_history'],
            'processor': lambda x: ' '.join([entry.get('text', '') for entry in x if isinstance(entry, dict) and 'text' in entry]) if isinstance(x, list) else ''
        },
        # Strategy 2: Direct diagnosis fields
        {
            'fields': ['correct_diagnosis', 'final_doctor_diagnosis', 'diagnosis'],
            'processor': lambda x: str(x) if x else ''
        },
        # Strategy 3: Symptom-specific fields
        {
            'fields': ['symptoms', 'Symptoms', 'symptom_data', 'chief_complaint', 'presenting_complaint'],
            'processor': lambda x: extract_text_content(x) if is_meaningful_content(x) else ''
        },
        # Strategy 4: Case description fields
        {
            'fields': ['scenario', 'case_description', 'patient_history', 'description', 'specialist_reason'],
            'processor': lambda x: str(x) if x else ''
        }
    ]
    
    # Try each strategy until we get meaningful content
    for strategy in extraction_strategies:
        for field in strategy['fields']:
            if field in row and is_meaningful_content(row[field]):
                try:
                    text = strategy['processor'](row[field])
                    if text and len(text.strip()) > 10:  # Meaningful content threshold
                        if isinstance(row[field], dict) and 'Primary_Symptom' in row[field]:
                            return row[field]  # Return structured format if available
                        extracted_text = text.strip()
                        break
                except Exception as e:
                    print(f"Warning: Error processing field {field}: {e}")
                    continue
        if extracted_text:
            break
    
    # Fallback: extract from entire row
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
    """
    Comprehensive keyword analysis with detailed breakdown.
    """
    text_lower = text.lower()
    
    # Find all matching keywords
    classic_matches = [kw for kw in CLASSIC_KEYWORDS if kw in text_lower]
    vague_matches = [kw for kw in VAGUE_KEYWORDS if kw in text_lower]
    
    # System-specific analysis
    system_matches = {}
    for system, keywords in MULTISYSTEM_KEYWORDS.items():
        matches = [kw for kw in keywords if kw in text_lower]
        if matches:
            system_matches[system] = matches
    
    return {
        'classic_keywords': classic_matches,
        'vague_keywords': vague_matches,
        'system_keywords': system_matches,
        'systems_involved': list(system_matches.keys()),
        'total_keywords': len(classic_matches) + len(vague_matches) + sum(len(matches) for matches in system_matches.values())
    }

def enhanced_categorize_symptom_presentation(symptom_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Enhanced categorization with detailed analysis and confidence scoring.
    """
    if not isinstance(symptom_data, dict) or not symptom_data:
        return 'Unclassified', {'reason': 'Invalid symptom data'}

    primary = symptom_data.get('Primary_Symptom', '')
    secondary = symptom_data.get('Secondary_Symptoms', [])

    if isinstance(secondary, str):
        secondary = [secondary]
    elif not isinstance(secondary, list):
        secondary = []

    all_symptoms = [primary] + secondary
    text = ' '.join([str(s) for s in all_symptoms if s]).strip()

    if not text or text == 'Unknown presentation':
        return 'Unclassified', {'reason': 'No symptom text available'}

    # Perform comprehensive keyword analysis
    keyword_analysis = analyze_symptom_keywords(text)
    
    # Decision logic with improved prioritization
    decision_info = {
        'text_analyzed': text[:200] + '...' if len(text) > 200 else text,
        'keyword_analysis': keyword_analysis
    }
    
    # 1. Classic presentation (highest priority)
    if keyword_analysis['classic_keywords']:
        return 'Classic Textbook', {
            **decision_info,
            'reason': f"Classic keywords found: {keyword_analysis['classic_keywords'][:3]}",
            'confidence': 'high'
        }
    
    # 2. Multi-system complex (2+ systems with symptoms)
    if len(keyword_analysis['systems_involved']) >= 2:
        return 'Multi-System Complex', {
            **decision_info,
            'reason': f"Multiple systems involved: {keyword_analysis['systems_involved']}",
            'confidence': 'high'
        }
    
    # 3. Explicit vague language
    if keyword_analysis['vague_keywords']:
        return 'Atypical/Vague Wording', {
            **decision_info,
            'reason': f"Vague language detected: {keyword_analysis['vague_keywords'][:3]}",
            'confidence': 'high'
        }
    
    # 4. Single clear symptom pattern
    if (len(keyword_analysis['systems_involved']) == 1 and 
        len(keyword_analysis['system_keywords'][keyword_analysis['systems_involved'][0]]) == 1):
        return 'Single Symptom Only', {
            **decision_info,
            'reason': f"Single symptom from {keyword_analysis['systems_involved'][0]} system",
            'confidence': 'medium'
        }
    
    # 5. Single system with multiple symptoms
    if len(keyword_analysis['systems_involved']) == 1:
        return 'Single System Multiple Symptoms', {
            **decision_info,
            'reason': f"Multiple symptoms from {keyword_analysis['systems_involved'][0]} system",
            'confidence': 'medium'
        }
    
    # 6. Fallback for unspecific presentations
    return 'Atypical/Vague Wording', {
        **decision_info,
        'reason': 'No clear pattern identified - defaulting to atypical',
        'confidence': 'low'
    }

def calculate_enhanced_performance_metrics(df: pd.DataFrame, group_col: str, actual_col: str = 'is_correct') -> pd.DataFrame:
    """
    Enhanced performance calculation with confidence intervals and additional metrics.
    """
    if group_col not in df.columns:
        raise KeyError(f"Grouping column '{group_col}' not found in the DataFrame.")
    
    if actual_col not in df.columns:
        raise KeyError(f"Actual outcome column '{actual_col}' not found in the DataFrame.")

    # Clean the data
    df_clean = df.copy()
    df_clean[actual_col] = pd.to_numeric(df_clean[actual_col], errors='coerce').astype('boolean')
    valid_df = df_clean.dropna(subset=[actual_col, group_col])

    def calculate_metrics(group_data):
        correct_cases = group_data[actual_col].sum()
        total_cases = len(group_data)
        incorrect_cases = total_cases - correct_cases
        
        if total_cases == 0:
            return pd.Series({
                'Accuracy_%': 0,
                'Total_Cases': 0,
                'Correct_Cases': 0,
                'Incorrect_Cases': 0,
                'Confidence_Interval_95%': 'N/A',
                'Avg_Diagnoses_Considered': 0
            })
        
        accuracy = (correct_cases / total_cases) * 100
        
        # Calculate 95% confidence interval for accuracy
        if total_cases > 0:
            p = correct_cases / total_cases
            margin_error = 1.96 * np.sqrt((p * (1 - p)) / total_cases)
            ci_lower = max(0, (p - margin_error) * 100)
            ci_upper = min(100, (p + margin_error) * 100)
            confidence_interval = f"({ci_lower:.1f}%-{ci_upper:.1f}%)"
        else:
            confidence_interval = "N/A"

        # Add average diagnoses considered if available
        avg_diagnoses = 0
        if 'num_diagnoses_considered' in group_data.columns:
            avg_diagnoses = group_data['num_diagnoses_considered'].mean()

        return pd.Series({
            'Accuracy_%': accuracy,
            'Total_Cases': total_cases,
            'Correct_Cases': correct_cases,
            'Incorrect_Cases': incorrect_cases,
            'Confidence_Interval_95%': confidence_interval,
            'Avg_Diagnoses_Considered': avg_diagnoses
        })

    grouped_metrics = valid_df.groupby(group_col, observed=True).apply(calculate_metrics, include_groups=False).reset_index()
    grouped_metrics = grouped_metrics.sort_values(by='Accuracy_%', ascending=False)
    
    return grouped_metrics

def analyze_diagnostic_complexity_enhanced(df: pd.DataFrame):
    """Enhanced diagnostic complexity analysis with statistical insights."""
    print("\n" + "="*70)
    print("ENHANCED DIAGNOSTIC COMPLEXITY ANALYSIS")
    print("="*70)
    
    if 'num_diagnoses_considered' not in df.columns or df['num_diagnoses_considered'].sum() == 0:
        print("No diagnoses considered data available for complexity analysis.")
        return
    
    # Overall statistics
    complexity_stats = df['num_diagnoses_considered'].describe()
    print(f"\nDiagnostic Complexity Statistics:")
    print(f"  Mean: {complexity_stats['mean']:.2f}")
    print(f"  Median: {complexity_stats['50%']:.1f}")
    print(f"  Std Dev: {complexity_stats['std']:.2f}")
    print(f"  Range: {complexity_stats['min']:.0f} - {complexity_stats['max']:.0f}")
    
    # Enhanced complexity categories
    df['diagnostic_complexity'] = pd.cut(
        df['num_diagnoses_considered'], 
        bins=[0, 2, 4, 6, float('inf')], 
        labels=["Simple (1-2)", "Moderate (3-4)", "Complex (5-6)", "Very Complex (7+)"],
        include_lowest=True
    )
    
    # Overall complexity distribution
    print(f"\nComplexity Distribution:")
    complexity_dist = df['diagnostic_complexity'].value_counts()
    for category, count in complexity_dist.items():
        percentage = (count / len(df)) * 100
        avg_accuracy = df[df['diagnostic_complexity'] == category]['is_correct'].mean() * 100
        print(f"  {category}: {count} cases ({percentage:.1f}%) - Avg Accuracy: {avg_accuracy:.1f}%")
    
    # Analysis by symptom presentation
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
    """Generate key insights and recommendations from the analysis."""
    print("\n" + "="*70)
    print("KEY INSIGHTS & RECOMMENDATIONS")
    print("="*70)
    
    total_cases = len(df)
    overall_accuracy = df['is_correct'].mean() * 100
    
    # Performance by category
    category_performance = df.groupby('symptom_presentation')['is_correct'].agg(['mean', 'count']).reset_index()
    category_performance['accuracy'] = category_performance['mean'] * 100
    category_performance = category_performance.sort_values('accuracy', ascending=False)
    
    print("\n Performance Rankings:")
    for _, row in category_performance.iterrows():
        if row['count'] >= 3:  # Only show categories with sufficient data
            print(f"  {row['symptom_presentation']}: {row['accuracy']:.1f}% ({row['count']} cases)")
    
    # Complexity insights
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
    
    # Recommendations
    print(f"\n Recommendations:")
    
    best_category = category_performance.iloc[0]
    worst_category = category_performance.iloc[-1]
    
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


# === SELF CONFIDENCE + DEMOGRAPHIC PARITY FUNCTIONS ===
def calculate_confidence_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Calculate self-confidence metrics for each group."""
    results = []
    
    for group_name in df[group_col].unique():
        if group_name == 'Unclassified':
            continue
            
        group_data = df[df[group_col] == group_name].copy()
        
        if len(group_data) == 0:
            continue
        
        # Basic metrics
        total_cases = len(group_data)
        correct_cases = group_data['is_correct'].sum()
        accuracy = (correct_cases / total_cases) * 100 if total_cases > 0 else 0
        
        # Self-confidence metrics using diagnostic complexity as proxy
        confidence_metrics = {}
        
        if 'num_diagnoses_considered' in group_data.columns:
            # Fewer diagnoses considered might indicate higher confidence
            avg_diagnoses = group_data['num_diagnoses_considered'].mean()
            confidence_metrics['avg_diagnoses_considered'] = avg_diagnoses
            # Inverse relationship: fewer diagnoses = higher confidence proxy
            confidence_metrics['confidence_proxy'] = max(0, (10 - avg_diagnoses) / 10 * 100)
            
            # Calibration metrics (confidence vs actual performance)
            expected_accuracy = confidence_metrics['confidence_proxy']
            actual_accuracy = accuracy
            calibration_error = abs(expected_accuracy - actual_accuracy)
            confidence_metrics['calibration_error'] = calibration_error
        
        results.append({
            'Group': group_name,
            'Total_Cases': total_cases,
            'Correct_Cases': correct_cases,
            'Accuracy_%': accuracy,
            **confidence_metrics
        })
    
    return pd.DataFrame(results).sort_values('Accuracy_%', ascending=False)

def calculate_demographic_parity(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Calculate demographic parity metrics across groups."""
    results = []
    
    # Overall baseline performance
    overall_accuracy = df['is_correct'].mean() * 100
    
    for group_name in df[group_col].unique():
        if group_name == 'Unclassified':
            continue
            
        group_data = df[df[group_col] == group_name].copy()
        
        if len(group_data) == 0:
            continue
        
        total_cases = len(group_data)
        accuracy = (group_data['is_correct'].sum() / total_cases) * 100
        
        # Calculate parity metrics
        parity_gap = accuracy - overall_accuracy
        representation = (total_cases / len(df)) * 100
        
        # Statistical significance test (if sample size allows)
        if total_cases >= 5:
            # Perform z-test comparing group accuracy to overall accuracy
            group_correct = group_data['is_correct'].sum()
            overall_correct = df['is_correct'].sum()
            overall_total = len(df)
            
            # Two-proportion z-test
            p1 = group_correct / total_cases
            p2 = overall_correct / overall_total
            
            pooled_p = (group_correct + overall_correct) / (total_cases + overall_total)
            se = np.sqrt(pooled_p * (1 - pooled_p) * (1/total_cases + 1/overall_total))
            
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
            'Parity_Gap_%': parity_gap,
            'Representation_%': representation,
            'Z_Score': z_score,
            'P_Value': p_value,
            'Statistically_Significant': is_significant
        })
    
    return pd.DataFrame(results).sort_values('Parity_Gap_%', ascending=False)

def print_combined_analysis(df: pd.DataFrame, group_col: str, expected_groups: List[str]):
    """Combined confidence and parity analysis in a comprehensive table format."""
    print("\n" + "="*100)
    print(f"SYMPTOM PRESENTATION - CONFIDENCE RATING & DEMOGRAPHIC PARITY")
    print("="*100)
    
    # Calculate metrics for all groups
    results = []
    overall_accuracy = df['is_correct'].mean() * 100
    
    for group_name in expected_groups:
        group_data = df[df[group_col] == group_name]
        
        if len(group_data) == 0:
            results.append({
                'Group': group_name,
                'Cases': 0,
                'Accuracy_%': 0,
                'Confidence_Proxy_%': 0,
                'Parity_Gap_%': 0,
                'Representation_%': 0,
                'Status': 'No Data'
            })
            continue
        
        total_cases = len(group_data)
        correct_cases = group_data['is_correct'].sum()
        accuracy = (correct_cases / total_cases) * 100
        
        # Confidence proxy (based on diagnostic complexity)
        avg_diagnoses = group_data['num_diagnoses_considered'].mean() if 'num_diagnoses_considered' in group_data.columns else 4.0
        confidence_proxy = max(0, (10 - avg_diagnoses) / 10 * 100)  # Inverse relationship
        
        # Parity metrics
        parity_gap = accuracy - overall_accuracy
        representation = (total_cases / len(df)) * 100
        
        # Status determination
        if total_cases < 5:
            status = 'Low Sample'
        elif abs(parity_gap) > 10:
            status = 'High Disparity'
        elif accuracy > overall_accuracy + 5:
            status = 'Above Average'
        elif accuracy < overall_accuracy - 5:
            status = 'Below Average'
        else:
            status = 'Balanced'
        
        results.append({
            'Group': group_name,
            'Cases': total_cases,
            'Accuracy_%': accuracy,
            'Confidence_Proxy_%': confidence_proxy,
            'Parity_Gap_%': parity_gap,
            'Representation_%': representation,
            'Status': status
        })
    
    # Create and display results table
    results_df = pd.DataFrame(results)
    
    print(f"\nOverall Baseline Accuracy: {overall_accuracy:.1f}%")
    print(f"Total Cases: {len(df)}")
    print("\n" + "-" * 100)
    print(f"{'Group':<25} {'Cases':<6} {'Accuracy':<9} {'Confidence':<11} {'Parity Gap':<11} {'Repr %':<7} {'Status':<15}")
    print("-" * 100)
    
    for _, row in results_df.iterrows():
        if row['Cases'] > 0:
            print(f"{row['Group']:<25} {row['Cases']:<6} {row['Accuracy_%']:<8.1f}% "
                  f"{row['Confidence_Proxy_%']:<10.1f}% {row['Parity_Gap_%']:<+10.1f}% "
                  f"{row['Representation_%']:<6.1f}% {row['Status']:<15}")
        else:
            print(f"{row['Group']:<25} {row['Cases']:<6} {'N/A':<8} {'N/A':<10} "
                  f"{'N/A':<10} {'N/A':<6} {row['Status']:<15}")
    
    print("-" * 100)
    
    # Detailed confidence analysis
    print("\n" + "="*60)
    print("DETAILED CONFIDENCE ANALYSIS")
    print("="*60)
    
    confidence_df = calculate_confidence_metrics(df, group_col)
    
    for _, row in confidence_df.iterrows():
        if row['Total_Cases'] > 0:
            print(f"\n{row['Group']}:")
            print(f"  Cases: {row['Total_Cases']}")
            print(f"  Accuracy: {row['Accuracy_%']:.1f}%")
            
            if 'confidence_proxy' in row:
                print(f"  Confidence Proxy: {row['confidence_proxy']:.1f}%")
                print(f"  Avg Diagnoses Considered: {row['avg_diagnoses_considered']:.1f}")
            
            if 'calibration_error' in row:
                calibration_status = "Well-calibrated" if row['calibration_error'] < 10 else "Poorly-calibrated"
                print(f"  Calibration Error: {row['calibration_error']:.1f}% ({calibration_status})")
    
    # Detailed parity analysis
    print("\n" + "="*60)
    print("DETAILED DEMOGRAPHIC PARITY ANALYSIS")
    print("="*60)
    
    parity_df = calculate_demographic_parity(df, group_col)
    
    for _, row in parity_df.iterrows():
        if row['Total_Cases'] > 0:
            print(f"\n{row['Group']}:")
            print(f"  Cases: {row['Total_Cases']} ({row['Representation_%']:.1f}% of dataset)")
            print(f"  Accuracy: {row['Accuracy_%']:.1f}%")
            print(f"  Parity Gap: {row['Parity_Gap_%']:+.1f}%")
            
            if row['Statistically_Significant']:
                significance = " *SIGNIFICANT*" if abs(row['Parity_Gap_%']) > 5 else " *significant*"
                print(f"  Statistical Test: Z={row['Z_Score']:.2f}, p={row['P_Value']:.3f}{significance}")
            else:
                print(f"  Statistical Test: Not significant (p={row['P_Value']:.3f})")
    
    # Summary insights
    print("\n" + "="*60)
    print("KEY FINDINGS & RECOMMENDATIONS")
    print("="*60)
    
    if results_df['Cases'].sum() > 0:
        valid_results = results_df[results_df['Cases'] > 0]
        
        if not valid_results.empty:
            print("\nConfidence Insights:")
            highest_conf = valid_results.loc[valid_results['Confidence_Proxy_%'].idxmax()]
            lowest_conf = valid_results.loc[valid_results['Confidence_Proxy_%'].idxmin()]
            
            print(f"  • Highest Confidence: {highest_conf['Group']} ({highest_conf['Confidence_Proxy_%']:.1f}%)")
            print(f"  • Lowest Confidence: {lowest_conf['Group']} ({lowest_conf['Confidence_Proxy_%']:.1f}%)")
            
            print("\nParity Insights:")
            largest_advantage = valid_results.loc[valid_results['Parity_Gap_%'].idxmax()]
            largest_disadvantage = valid_results.loc[valid_results['Parity_Gap_%'].idxmin()]
            
            if largest_advantage['Parity_Gap_%'] > 5:
                print(f"  • Performance Advantage: {largest_advantage['Group']} ({largest_advantage['Parity_Gap_%']:+.1f}%)")
            if largest_disadvantage['Parity_Gap_%'] < -5:
                print(f"  • Performance Disadvantage: {largest_disadvantage['Group']} ({largest_disadvantage['Parity_Gap_%']:+.1f}%)")
            
            # Representation insights
            underrep = valid_results[valid_results['Representation_%'] < 5]
            if not underrep.empty:
                print(f"\nRepresentation Concerns:")
                print(f"  • {len(underrep)} group(s) are underrepresented (< 5% of data)")
                for _, row in underrep.iterrows():
                    print(f"    - {row['Group']}: {row['Representation_%']:.1f}% ({row['Cases']} cases)")
    
    print("\nMethodology Notes:")
    print("  • Confidence Proxy: Inverse of diagnostic complexity (fewer diagnoses = higher confidence)")
    print("  • Parity Gap: Difference from overall accuracy baseline")
    print("  • Statistical significance tested using two-proportion z-test")

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
