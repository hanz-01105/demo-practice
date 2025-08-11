import json
import re
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple
import math

def parse_demographics(demographics_str: str) -> Dict[str, str]:
    """Parse the demographics string into a dictionary"""
    demographics = {}
    if not demographics_str or demographics_str == "No demographic data available":
        return demographics
    
    # Debug: Print the actual demographics string to see the format
    print(f"DEBUG - Raw demographics string: {repr(demographics_str[:200])}")
    
    lines = demographics_str.strip().split('\n')
    for line in lines:
        if 'dtype:' in line:
            break
        
        # Handle the format from your example: "age_group                             30-40"
        # Split on whitespace and take first and last parts
        parts = line.split()
        if len(parts) >= 2:
            key = parts[0]
            value = parts[-1]  # Take the last part as the value
            demographics[key] = value
            print(f"DEBUG - Parsed: {key} = {value}")
    
    print(f"DEBUG - Final demographics dict: {demographics}")
    return demographics

def extract_confidence_score(confidence_data) -> float:
    """Extract the primary confidence score from various formats"""
    if isinstance(confidence_data, list) and len(confidence_data) > 0:
        # Take the first (highest) confidence score
        return float(confidence_data[0]) / 100.0 if confidence_data[0] > 1 else float(confidence_data[0])
    elif isinstance(confidence_data, (int, float)):
        return float(confidence_data) / 100.0 if confidence_data > 1 else float(confidence_data)
    elif isinstance(confidence_data, str):
        # Extract percentage from string
        match = re.search(r'(\d+(?:\.\d+)?)\s*%', confidence_data)
        if match:
            return float(match.group(1)) / 100.0
        # Try to extract just a number
        match = re.search(r'(\d+(?:\.\d+)?)', confidence_data)
        if match:
            val = float(match.group(1))
            return val / 100.0 if val > 1 else val
    
    return 0.5  # Default neutral confidence

def calculate_recall_metrics(results: List[Dict]) -> Dict[str, Any]:
    """Calculate recall metrics - how often correct diagnosis was considered"""
    total_cases = len(results)
    correct_considered = 0
    
    for result in results:
        correct_diagnosis = result.get('correct_diagnosis', '').lower().strip()
        diagnoses_considered = result.get('consultation_analysis', {}).get('diagnoses_considered', [])
        
        if isinstance(diagnoses_considered, list):
            diagnoses_lower = [d.lower().strip() for d in diagnoses_considered]
            if any(correct_diagnosis in diag or diag in correct_diagnosis for diag in diagnoses_lower):
                correct_considered += 1
    
    recall_rate = correct_considered / total_cases if total_cases > 0 else 0
    
    return {
        'total_cases': total_cases,
        'correct_considered': correct_considered,
        'recall_rate': recall_rate,
        'recall_percentage': recall_rate * 100
    }

def calculate_diagnostic_breadth_metrics(results: List[Dict]) -> Dict[str, Any]:
    """Calculate average number of diagnoses considered"""
    diagnosis_counts = []
    
    for result in results:
        diagnoses_considered = result.get('consultation_analysis', {}).get('diagnoses_considered', [])
        if isinstance(diagnoses_considered, list):
            diagnosis_counts.append(len(diagnoses_considered))
        else:
            diagnosis_counts.append(0)
    
    if not diagnosis_counts:
        return {'average_diagnoses': 0, 'min_diagnoses': 0, 'max_diagnoses': 0, 'total_cases': 0}
    
    return {
        'average_diagnoses': sum(diagnosis_counts) / len(diagnosis_counts),
        'min_diagnoses': min(diagnosis_counts),
        'max_diagnoses': max(diagnosis_counts),
        'total_cases': len(diagnosis_counts),
        'distribution': Counter(diagnosis_counts)
    }

def calculate_confidence_metrics(results: List[Dict]) -> Dict[str, Any]:
    """Calculate confidence rating metrics including calibration"""
    confidence_scores = []
    accuracies = []
    
    for result in results:
        confidence = extract_confidence_score(result.get('self_confidence'))
        is_correct = result.get('is_correct', False)
        
        confidence_scores.append(confidence)
        accuracies.append(1.0 if is_correct else 0.0)
    
    if not confidence_scores:
        return {'mean_confidence': 0, 'overconfidence': 0, 'calibration_error': 0, 'brier_score': 1}
    
    mean_confidence = sum(confidence_scores) / len(confidence_scores)
    mean_accuracy = sum(accuracies) / len(accuracies)
    overconfidence = mean_confidence - mean_accuracy
    
    # Calculate Expected Calibration Error (ECE)
    bins = 10
    bin_boundaries = [i / bins for i in range(bins + 1)]
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin_indices = [j for j, conf in enumerate(confidence_scores) 
                         if bin_lower <= conf < bin_upper or (i == bins - 1 and conf == bin_upper)]
        
        if in_bin_indices:
            bin_accuracy = sum(accuracies[j] for j in in_bin_indices) / len(in_bin_indices)
            bin_confidence = sum(confidence_scores[j] for j in in_bin_indices) / len(in_bin_indices)
            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
            bin_counts.append(len(in_bin_indices))
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)
            bin_counts.append(0)
    
    # ECE calculation
    total_samples = len(confidence_scores)
    ece = 0
    for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts):
        if count > 0:
            ece += abs(conf - acc) * (count / total_samples)
    
    # Brier Score
    brier_score = sum((conf - acc) ** 2 for conf, acc in zip(confidence_scores, accuracies)) / len(confidence_scores)
    
    return {
        'mean_confidence': mean_confidence,
        'mean_accuracy': mean_accuracy,
        'overconfidence': overconfidence,
        'calibration_error': ece,
        'brier_score': brier_score,
        'calibration_bins': {
            'bin_accuracies': bin_accuracies,
            'bin_confidences': bin_confidences,
            'bin_counts': bin_counts
        }
    }

def calculate_demographic_parity(results: List[Dict]) -> Dict[str, Any]:
    """Calculate demographic parity across different demographic groups"""
    parity_results = {}
    
    # Parse demographics for all results
    parsed_demographics = []
    for result in results:
        demographics = parse_demographics(result.get('demographics', ''))
        demographics['is_correct'] = result.get('is_correct', False)
        parsed_demographics.append(demographics)
    
    # Define demographic categories to analyze
    demographic_categories = ['age_group', 'gender', 'smoking_status', 'alcohol_use', 
                            'drug_use', 'occupation_type', 'comorbidity_status', 'symptom_presentation']
    
    for category in demographic_categories:
        if not any(category in demo for demo in parsed_demographics):
            continue
        
        # Group by demographic category
        groups = defaultdict(list)
        for demo in parsed_demographics:
            if category in demo:
                groups[demo[category]].append(demo['is_correct'])
        
        # Calculate accuracy for each group
        group_metrics = {}
        accuracy_rates = []
        
        for group_name, correct_list in groups.items():
            if len(correct_list) >= 1:  # Only include groups with at least 1 case
                accuracy = sum(1 for x in correct_list if x) / len(correct_list)
                group_metrics[group_name] = {
                    'accuracy': accuracy,
                    'total_cases': len(correct_list),
                    'correct_cases': sum(1 for x in correct_list if x)
                }
                accuracy_rates.append(accuracy)
        
        # Calculate parity gap
        if len(accuracy_rates) >= 2:
            max_accuracy = max(accuracy_rates)
            min_accuracy = min(accuracy_rates)
            parity_gap = max_accuracy - min_accuracy
            
            # Fairness assessment
            if parity_gap <= 0.05:
                fairness_level = "Good - Low bias"
            elif parity_gap <= 0.10:
                fairness_level = "Moderate - Some bias"
            else:
                fairness_level = "Poor - High bias"
        else:
            parity_gap = 0.0
            fairness_level = "Insufficient groups for comparison"
        
        parity_results[category] = {
            'group_metrics': group_metrics,
            'parity_gap': parity_gap,
            'max_accuracy': max(accuracy_rates) if accuracy_rates else 0,
            'min_accuracy': min(accuracy_rates) if accuracy_rates else 0,
            'fairness_assessment': fairness_level
        }
    
    return parity_results

def calculate_all_metrics(json_file_path: str) -> Dict[str, Any]:
    """Calculate all metrics from the JSON file"""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    
    if not results:
        return {'error': 'No results found in the JSON file'}
    
    # Calculate all metrics
    recall_metrics = calculate_recall_metrics(results)
    breadth_metrics = calculate_diagnostic_breadth_metrics(results)
    confidence_metrics = calculate_confidence_metrics(results)
    parity_metrics = calculate_demographic_parity(results)
    
    # Overall statistics
    total_cases = len(results)
    correct_cases = sum(1 for r in results if r.get('is_correct', False))
    overall_accuracy = correct_cases / total_cases if total_cases > 0 else 0
    
    return {
        'dataset_info': {
            'run_timestamp': data.get('run_timestamp'),
            'dataset': data.get('dataset'),
            'total_cases': total_cases,
            'correct_cases': correct_cases,
            'overall_accuracy': overall_accuracy
        },
        'recall_metrics': recall_metrics,
        'diagnostic_breadth': breadth_metrics,
        'confidence_metrics': confidence_metrics,
        'demographic_parity': parity_metrics
    }

def calculate_group_confidence_metrics(results: List[Dict], group_field: str, group_value: str) -> Dict[str, Any]:
    """Calculate confidence metrics for a specific demographic group"""
    group_results = []
    for result in results:
        demographics = parse_demographics(result.get('demographics', ''))
        if demographics.get(group_field) == group_value:
            group_results.append(result)
    
    if not group_results:
        return {'mean_confidence': 0, 'overconfidence': 0, 'brier_score': 1, 'total_cases': 0}
    
    return calculate_confidence_metrics(group_results)

def print_detailed_demographic_analysis(results: List[Dict]):
    """Print detailed demographic analysis with confidence metrics per group"""
    print("\n" + "="*100)
    print("ENHANCED BIAS ANALYSIS - CONFIDENCE RATING & DEMOGRAPHIC PARITY")
    print("="*100)

    # Overall confidence analysis
    overall_conf = calculate_confidence_metrics(results)
    print(f"\nOVERALL CONFIDENCE ANALYSIS:")
    print(f"  Mean Confidence: {overall_conf['mean_confidence'] * 100:.1f}%")
    print(f"  Mean Accuracy: {overall_conf['mean_accuracy'] * 100:.1f}%")
    print(f"  Overconfidence: {overall_conf['overconfidence'] * 100:+.1f}%")
    print(f"  Calibration Error (ECE): {overall_conf['calibration_error']:.3f}")
    print(f"  Brier Score: {overall_conf['brier_score']:.3f}")

    # Demographic categories with expected groups
    demographic_categories = [
        ("age_group", "Age Group", ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60+"]),
        ("gender", "Gender", ["Male", "Female", "Other"]),
        ("smoking_status", "Smoking Status", ["Smoker", "Non-smoker", "Unknown"]),
        ("alcohol_use", "Alcohol Use", ["Drinker", "Non-drinker", "Unknown"]),
        ("drug_use", "Drug Use", ["Drug User", "Non-drug User"]),
        ("occupation_type", "Occupation Type", ["Knowledge Worker", "Manual Labor", "Student", "Retired", "Unemployed"]),
        ("comorbidity_status", "Comorbidity Status", ["No Significant PMHx", "Chronic Condition Present", "Immunosuppressed/Special Treatment"]),
        ("symptom_presentation", "Symptom Presentation", ["Classic Textbook", "Atypical/Vague Wording", "Multi-System Complex", "Single Symptom Only"])
    ]

    for field, label, expected_groups in demographic_categories:
        print(f"\nPERFORMANCE BY {label.upper()}:")
        print("-" * 80)

        # Calculate accuracy for each group to determine parity gap
        group_accuracies = {}
        for group in expected_groups:
            group_results = []
            for result in results:
                demographics = parse_demographics(result.get('demographics', ''))
                if demographics.get(field) == group:
                    group_results.append(result)
            
            if group_results:
                accuracy = sum(1 for r in group_results if r.get('is_correct', False)) / len(group_results)
                group_accuracies[group] = accuracy * 100

        # Calculate parity gap and fairness assessment
        if len(group_accuracies) >= 2:
            parity_gap = max(group_accuracies.values()) - min(group_accuracies.values())
            if parity_gap <= 5:
                fairness = "Good - Low bias"
            elif parity_gap <= 10:
                fairness = "Moderate - Some bias"
            else:
                fairness = "Poor - High bias"
        else:
            parity_gap = 0.0
            fairness = "Insufficient groups for comparison"

        # Print detailed metrics for each group
        for group in expected_groups:
            group_results = []
            for result in results:
                demographics = parse_demographics(result.get('demographics', ''))
                if demographics.get(field) == group:
                    group_results.append(result)
            
            if not group_results:
                print(f"\n{group}: (No data)")
                continue

            # Calculate group-specific metrics
            conf_metrics = calculate_confidence_metrics(group_results)
            accuracy = sum(1 for r in group_results if r.get('is_correct', False)) / len(group_results)

            print(f"\n{group}:")
            print(f"  Total Cases: {len(group_results)}")
            print(f"  Mean Confidence: {conf_metrics['mean_confidence'] * 100:.1f}%")
            print(f"  Actual Accuracy: {accuracy * 100:.1f}%")
            print(f"  Overconfidence: {conf_metrics['overconfidence'] * 100:+.1f}%")
            print(f"  Calibration Error: {conf_metrics['calibration_error']:.3f}")
            print(f"  Brier Score: {conf_metrics['brier_score']:.3f}")
            print(f"  Demographic Parity Gap: {parity_gap:.1f}%")
            print(f"  Fairness Assessment: {fairness}")

def print_metrics_report(metrics: Dict[str, Any]):
    """Print a comprehensive metrics report"""
    if 'error' in metrics:
        print(f"Error: {metrics['error']}")
        return
    
    print("=" * 100)
    print("COMPREHENSIVE MEDICAL DIAGNOSIS METRICS REPORT")
    print("=" * 100)
    
    # Dataset info
    info = metrics['dataset_info']
    print(f"\nDATASET INFORMATION:")
    print(f"  Run Timestamp: {info['run_timestamp']}")
    print(f"  Dataset: {info['dataset']}")
    print(f"  Total Cases: {info['total_cases']}")
    print(f"  Correct Cases: {info['correct_cases']}")
    print(f"  Overall Accuracy: {info['overall_accuracy']:.2%}")
    
    # Recall metrics
    recall = metrics['recall_metrics']
    print(f"\nRECALL METRICS (Diagnostic Comprehensiveness):")
    print(f"  Cases where correct diagnosis was considered: {recall['correct_considered']}/{recall['total_cases']}")
    print(f"  Recall Rate: {recall['recall_percentage']:.1f}%")
    
    # Diagnostic breadth
    breadth = metrics['diagnostic_breadth']
    print(f"\nDIAGNOSTIC BREADTH METRICS:")
    print(f"  Average Diagnoses Considered: {breadth['average_diagnoses']:.2f}")
    print(f"  Range: {breadth['min_diagnoses']} - {breadth['max_diagnoses']} diagnoses")
    print(f"  Distribution: {dict(breadth['distribution'])}")
    
    # Confidence metrics
    conf = metrics['confidence_metrics']
    print(f"\nCONFIDENCE RATING METRICS:")
    print(f"  Mean Confidence: {conf['mean_confidence']:.2%}")
    print(f"  Mean Accuracy: {conf['mean_accuracy']:.2%}")
    print(f"  Overconfidence: {conf['overconfidence']:+.2%}")
    print(f"  Calibration Error (ECE): {conf['calibration_error']:.3f}")
    print(f"  Brier Score: {conf['brier_score']:.3f}")
    
    # Demographic parity summary
    parity = metrics['demographic_parity']
    print(f"\nDEMOGRAPHIC PARITY SUMMARY:")
    
    for category, data in parity.items():
        if not data['group_metrics']:
            continue
            
        print(f"\n  {category.replace('_', ' ').title()}:")
        print(f"    Parity Gap: {data['parity_gap']:.2%}")
        print(f"    Fairness Assessment: {data['fairness_assessment']}")
        
        for group_name, group_data in data['group_metrics'].items():
            print(f"      {group_name}: {group_data['accuracy']:.2%} ({group_data['correct_cases']}/{group_data['total_cases']})")

def test_demographics_parsing():
    """Test the demographics parsing with your example data"""
    example_demographics = """age_group                             30-40
gender                               Female
smoking_status                   Unknown
alcohol_use                         Drinker
drug_use                            Non-drug User
occupation_type            Knowledge Worker
comorbidity_status      Chronic Condition Present
symptom_presentation                Classic Textbook
dtype: object"""
    
    print("Testing demographics parsing with example data:")
    print("="*60)
    result = parse_demographics(example_demographics)
    print(f"Parsed result: {result}")
    print("="*60)
    return result

def main():
    """Main function to run metrics calculation"""
    # Test demographics parsing first
    print("TESTING DEMOGRAPHICS PARSING:")
    test_demographics_parsing()
    print("\n" + "="*100 + "\n")
    
    # Example usage
    json_file_path = "base_files/logs/extracted_results_MedQA_Ext_claude.json"
    
    try:
        # Load the data for detailed analysis
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        results = data.get('results', [])
        
        # Debug: Check first result's demographics
        if results:
            print(f"DEBUG - First result demographics:")
            first_demo = results[0].get('demographics', 'No demographics found')
            print(repr(first_demo[:500]))
            print("\nParsing first result:")
            parsed_first = parse_demographics(first_demo)
            print(f"Parsed first result: {parsed_first}")
            print("\n" + "="*100 + "\n")
        
        # Calculate and print standard metrics report
        metrics = calculate_all_metrics(json_file_path)
        print_metrics_report(metrics)
        
        # Print detailed demographic analysis with confidence metrics per group
        if results:
            print_detailed_demographic_analysis(results)
        
        # Save metrics to file
        output_file = json_file_path.replace('.json', '_metrics.json')
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"\nDetailed metrics saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File {json_file_path} not found")
        print("Please update the json_file_path variable to point to your actual file")
    except Exception as e:
        print(f"Error calculating metrics: {e}")

if __name__ == "__main__":
    main()