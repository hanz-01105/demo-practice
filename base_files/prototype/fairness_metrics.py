import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error
from sklearn.calibration import calibration_curve
from typing import Dict, List, Tuple, Any
import warnings
import os
import json
import re
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class HealthcareFairnessMetrics:
    """
    Comprehensive fairness metrics implementation for healthcare applications
    """

    def __init__(self):
        self.results = {}

    def demographic_parity(self, y_true: np.array, y_pred: np.array,
                          sensitive_attr: np.array, threshold: float = 0.5) -> Dict:
        """
        Calculate demographic parity across groups
        """
        results = {}
        groups = np.unique(sensitive_attr)

        y_pred_binary = (y_pred >= threshold).astype(int)

        positive_rates = {}
        for group in groups:
            mask = sensitive_attr == group
            if np.sum(mask) > 0:
                positive_rate = np.mean(y_pred_binary[mask])
                positive_rates[group] = positive_rate

        if positive_rates:
            parity_diff = max(positive_rates.values()) - min(positive_rates.values())
            results['is_fair'] = parity_diff <= 0.1
        else:
            parity_diff = 0
            results['is_fair'] = True

        results['positive_rates'] = positive_rates
        results['parity_difference'] = parity_diff

        return results

    def equalized_odds(self, y_true: np.array, y_pred: np.array,
                      sensitive_attr: np.array, threshold: float = 0.5) -> Dict:
        """
        Calculate equalized odds (TPR and FPR equality across groups)
        """
        results = {}
        groups = np.unique(sensitive_attr)

        y_pred_binary = (y_pred >= threshold).astype(int)

        tpr_by_group = {}
        fpr_by_group = {}

        for group in groups:
            mask = sensitive_attr == group
            if np.sum(mask) == 0:
                continue

            # THE FIX: Add labels=[0, 1] to ensure a 2x2 matrix is always returned
            tn, fp, fn, tp = confusion_matrix(y_true[mask], y_pred_binary[mask], labels=[0, 1]).ravel()

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            tpr_by_group[group] = tpr
            fpr_by_group[group] = fpr

        if tpr_by_group and fpr_by_group:
            tpr_diff = max(tpr_by_group.values()) - min(tpr_by_group.values())
            fpr_diff = max(fpr_by_group.values()) - min(fpr_by_group.values())
            is_fair = (tpr_diff <= 0.1) and (fpr_diff <= 0.1)
        else:
            tpr_diff = 0
            fpr_diff = 0
            is_fair = True

        results['tpr_by_group'] = tpr_by_group
        results['fpr_by_group'] = fpr_by_group
        results['tpr_difference'] = tpr_diff
        results['fpr_difference'] = fpr_diff
        results['is_fair'] = is_fair

        return results

    def equal_opportunity(self, y_true: np.array, y_pred: np.array,
                         sensitive_attr: np.array, threshold: float = 0.5) -> Dict:
        """
        Calculate equal opportunity (TPR equality across groups)
        """
        results = {}
        groups = np.unique(sensitive_attr)

        y_pred_binary = (y_pred >= threshold).astype(int)

        tpr_by_group = {}

        for group in groups:
            mask = sensitive_attr == group
            if np.sum(mask) == 0:
                continue

            # THE FIX: Add labels=[0, 1] here as well
            tn, fp, fn, tp = confusion_matrix(y_true[mask], y_pred_binary[mask], labels=[0, 1]).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tpr_by_group[group] = tpr

        if tpr_by_group:
            tpr_diff = max(tpr_by_group.values()) - min(tpr_by_group.values())
            is_fair = tpr_diff <= 0.1
        else:
            tpr_diff = 0
            is_fair = True

        results['tpr_by_group'] = tpr_by_group
        results['tpr_difference'] = tpr_diff
        results['is_fair'] = is_fair

        return results

    def calibration_by_group(self, y_true: np.array, y_pred: np.array,
                            sensitive_attr: np.array, n_bins: int = 5) -> Dict:
        """
        Calculate calibration metrics by group
        """
        results = {}
        groups = np.unique(sensitive_attr)

        calibration_errors = {}

        for group in groups:
            mask = sensitive_attr == group
            if np.sum(y_true[mask]) < 1 or len(np.unique(y_true[mask])) < 2:
                # Cannot calculate calibration if only one class is present
                calibration_errors[group] = np.nan
                continue

            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true[mask], y_pred[mask], n_bins=n_bins
                )
                ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
                calibration_errors[group] = ece
            except ValueError:
                calibration_errors[group] = np.nan

        valid_errors = [e for e in calibration_errors.values() if not np.isnan(e)]
        if valid_errors:
            calibration_disparity = max(valid_errors) - min(valid_errors)
            is_fair = calibration_disparity <= 0.1
        else:
            calibration_disparity = np.nan
            is_fair = True

        results['calibration_errors'] = calibration_errors
        results['calibration_disparity'] = calibration_disparity
        results['is_fair'] = is_fair

        return results

    def statistical_parity(self, y_pred: np.array, sensitive_attr: np.array) -> Dict:
        """
        Calculate statistical parity (equal mean predictions across groups)
        """
        results = {}
        groups = np.unique(sensitive_attr)

        mean_predictions = {}
        for group in groups:
            mask = sensitive_attr == group
            if np.sum(mask) > 0:
                mean_predictions[group] = np.mean(y_pred[mask])

        if mean_predictions:
            parity_diff = max(mean_predictions.values()) - min(mean_predictions.values())
            is_fair = parity_diff <= 0.1
        else:
            parity_diff = 0
            is_fair = True

        results['mean_predictions'] = mean_predictions
        results['parity_difference'] = parity_diff
        results['is_fair'] = is_fair

        return results

    def comprehensive_fairness_report(self, y_true: np.array, y_pred: np.array,
                                    demographics: Dict[str, np.array],
                                    lifestyle: Dict[str, np.array] = None,
                                    threshold: float = 0.5) -> Dict:
        """
        Generate comprehensive fairness report for all categories
        """
        report = {}
        all_attributes = {**demographics, **(lifestyle or {})}

        for attr_name, attr_values in all_attributes.items():
            report[attr_name] = {
                'demographic_parity': self.demographic_parity(y_true, y_pred, attr_values, threshold),
                'equalized_odds': self.equalized_odds(y_true, y_pred, attr_values, threshold),
                'equal_opportunity': self.equal_opportunity(y_true, y_pred, attr_values, threshold),
                'calibration': self.calibration_by_group(y_true, y_pred, attr_values),
                'statistical_parity': self.statistical_parity(y_pred, attr_values),
            }
        return report

    def generate_fairness_summary(self, report: Dict) -> Dict:
        """
        Generate summary statistics from comprehensive report
        """
        summary = {
            'total_metrics': 0,
            'fair_metrics': 0,
            'unfair_metrics': 0,
            'fairness_rate': 0.0,
            'critical_issues': []
        }

        for category, metrics in report.items():
            for metric_name, metric_result in metrics.items():
                summary['total_metrics'] += 1

                is_fair = metric_result.get('is_fair', False)
                if pd.isna(is_fair): is_fair = True # Treat NaN as fair (not enough data to prove unfair)

                if is_fair:
                    summary['fair_metrics'] += 1
                else:
                    summary['unfair_metrics'] += 1
                    issue_text = f"{category.replace('_', ' ').title()} {metric_name.replace('_', ' ')} violation"
                    summary['critical_issues'].append(issue_text)

        if summary['total_metrics'] > 0:
            summary['fairness_rate'] = summary['fair_metrics'] / summary['total_metrics']

        return summary

# ===================================================================
# Main script execution block - FINAL VERSION
# ===================================================================
if __name__ == "__main__":

    # --- Step 1: Load and Parse Your JSON Log File ---
    LOG_PATH = "base_files/logs"
    log_file = "agentclinic_run_latest.json"
    log_filepath = os.path.join(LOG_PATH, log_file)

    if not os.path.exists(log_filepath):
        raise FileNotFoundError(f"Log file not found: {log_filepath}")

    with open(log_filepath, "r") as f:
        data = json.load(f)

    df = pd.json_normalize(data["logs"])

    def parse_demographics(demo_string):
        if not isinstance(demo_string, str):
            return {}
        pattern = re.compile(r"([\w_]+)\s+([\w\s-]+?)(?=\n\w+|\n$|dtype)")
        matches = pattern.findall(demo_string)
        return {key: value.strip() for key, value in matches}

    demo_df = df['demographics'].apply(parse_demographics).apply(pd.Series)
    df = pd.concat([df.drop(columns=['demographics']), demo_df], axis=1)

    print(f"Successfully loaded and processed {log_filepath}. Shape: {df.shape}")

    # --- Step 2: Data Cleaning ---
    cols_to_clean = ['gender', 'age_group', 'comorbidity_status',
                       'occupation_type', 'smoking_status', 'alcohol_use', 'drug_use']

    for col in cols_to_clean:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)
        else:
            df[col] = 'Unknown'

    print("Cleaned all demographic and lifestyle columns.")

    # --- Step 3: Map Your Columns ---
    try:
        y_true = df['is_correct'].astype(int).values
        y_pred_series = df['self_confidence'].fillna('0%')
        y_pred = y_pred_series.str.replace('%', '').astype(float) / 100.0
        y_pred = y_pred.values

        demographics = {
            'gender': df['gender'].values,
            'age_group': df['age_group'].values,
            'comorbidity': df['comorbidity_status'].values,
            'occupation': df['occupation_type'].values
        }

        lifestyle = {
            'smoking_status': df['smoking_status'].values,
            'alcohol_use': df['alcohol_use'].values,
            'drug_use': df['drug_use'].values
        }

    except KeyError as e:
        print(f"\n--- ERROR ---")
        print(f"A required column was not found: {e}.")
        print("Stopping analysis.")
        exit()

    # --- Step 4: Run Analysis ---
    fairness_analyzer = HealthcareFairnessMetrics()

    print("\n=== RUNNING COMPREHENSIVE FAIRNESS ANALYSIS ===")
    report = fairness_analyzer.comprehensive_fairness_report(
        y_true, y_pred, demographics, lifestyle
    )
    summary = fairness_analyzer.generate_fairness_summary(report)

    # --- Step 5: Display Summary ---
    print("\n=== FAIRNESS ANALYSIS SUMMARY ===")
    print(f"Total metrics evaluated: {summary['total_metrics']}")
    print(f"Fair metrics: {summary['fair_metrics']}")
    print(f"Unfair metrics: {summary['unfair_metrics']}")
    print(f"Overall fairness rate: {summary['fairness_rate']:.2%}")

    if summary['critical_issues']:
        print("\n=== CRITICAL ISSUES ===")
        for issue in summary['critical_issues']:
            print(f"- {issue}")

    # --- Step 6: Display Detailed Metrics ---
    print("\n\n" + "="*25)
    print(" DETAILED FAIRNESS REPORT ")
    print("="*25)

    for category, metrics in report.items():
        category_title = category.replace('_', ' ').upper()
        print(f"\n\n--- {category_title} FAIRNESS DETAILS ---")

        for metric_name, result in metrics.items():
            metric_title = metric_name.replace('_', ' ').title()
            print(f"\n▶ {metric_title}:")

            if not result:
                print("  - Not enough data to compute.")
                continue

            for key, value in result.items():
                if isinstance(value, float):
                    print(f"  - {key}: {value:.3f}")
                else:
                    print(f"  - {key}: {value}")


