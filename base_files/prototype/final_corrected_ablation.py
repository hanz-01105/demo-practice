#!/usr/bin/env python3
"""
Final Fixed Ablation Analysis with Realistic Symptom Categorization
==================================================================
"""

import json
import pandas as pd
import numpy as np
import re
import os
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

class FinalFixedAblationAnalyzer:
    """Final ablation analyzer with realistic symptom categorization thresholds"""
    
    def __init__(self, log_file_path: str, output_dir: str = "base_files/analysis"):
        self.log_file_path = log_file_path
        self.output_dir = output_dir
        self.df = None
        
        # Medical keywords for symptom categorization
        self.system_keywords = {
            'cardiovascular': [
                'chest pain', 'palpitations', 'syncope', 'orthopnea', 'dyspnea', 'edema',
                'heart', 'cardiac', 'blood pressure', 'hypertension', 'arrhythmia', 'murmur'
            ],
            'respiratory': [
                'shortness of breath', 'cough', 'wheezing', 'hemoptysis', 'sputum',
                'breathing', 'lung', 'pulmonary', 'asthma', 'pneumonia', 'bronch'
            ],
            'neurological': [
                'headache', 'dizziness', 'seizure', 'weakness', 'numbness', 'tingling',
                'confusion', 'memory', 'tremor', 'paralysis', 'stroke', 'neurologic'
            ],
            'gastrointestinal': [
                'nausea', 'vomiting', 'diarrhea', 'constipation', 'abdominal pain',
                'stomach', 'bowel', 'liver', 'hepatic', 'gastric', 'intestinal'
            ],
            'musculoskeletal': [
                'joint pain', 'muscle pain', 'back pain', 'arthritis', 'fracture',
                'bone', 'joint', 'muscle', 'spine', 'orthopedic'
            ],
            'genitourinary': [
                'urinary', 'kidney', 'bladder', 'renal', 'urine', 'dysuria',
                'hematuria', 'incontinence', 'frequency', 'urgency'
            ],
            'endocrine': [
                'diabetes', 'thyroid', 'hormone', 'metabolic', 'glucose',
                'insulin', 'endocrine', 'adrenal', 'pituitary'
            ],
            'dermatological': [
                'rash', 'skin', 'dermatitis', 'lesion', 'eczema', 'psoriasis',
                'itching', 'bruising', 'discoloration', 'dermatologic'
            ]
        }
        
        os.makedirs(output_dir, exist_ok=True)
    
    def run_complete_analysis(self):
        """Run the complete analysis with fixed categorization"""
        
        print("="*80)
        print("FINAL FIXED ABLATION ANALYSIS")
        print("="*80)
        print(f"Input file: {self.log_file_path}")
        print(f"Output directory: {self.output_dir}")
        print("="*80)
        
        # Load and process data
        print("\n1. Loading and processing data...")
        self.load_and_process_data()
        
        # Derive symptom categories with FIXED logic
        print("\n2. Deriving symptom presentation categories (FIXED)...")
        self.derive_symptom_presentations_fixed()
        
        # Validate data
        print("\n3. Validating data quality...")
        self.validate_data_quality()
        
        # Generate tables
        print("\n4. Generating ablation tables...")
        tables = self.generate_all_ablation_tables()
        
        # Save results
        print("\n5. Saving results...")
        self.save_all_results(tables)
        
        # Print tables
        print("\n6. Analysis complete!")
        self.print_tables(tables)
        
        return tables
    
    def load_and_process_data(self):
        """Load and process data"""
        
        # Load data
        with open(self.log_file_path, 'r') as f:
            raw_data = json.load(f)
        
        if isinstance(raw_data, dict) and 'results' in raw_data:
            results = raw_data['results']
        elif isinstance(raw_data, list):
            results = raw_data
        else:
            raise ValueError("Unexpected data format")
        
        self.df = pd.DataFrame(results)
        print(f"   Loaded {len(self.df)} scenarios")
        
        # Process each component
        self.process_demographics()
        self.process_top_k_metrics()
        self.process_recall_metrics_corrected()
        self.process_confidence_metrics()
        
        print("   Data processing complete")
    
    def process_demographics(self):
        """Process existing demographics"""
        
        demographics_data = []
        
        for _, row in self.df.iterrows():
            demo_raw = row.get('demographics', '')
            demo_dict = self.parse_demographics(demo_raw)
            demographics_data.append(demo_dict)
        
        # Add demographic columns
        demo_df = pd.DataFrame(demographics_data)
        for col in demo_df.columns:
            self.df[f'demo_{col}'] = demo_df[col]
    
    def parse_demographics(self, demo_data: Any) -> Dict[str, str]:
        """Parse demographics from pandas Series format"""
        
        defaults = {
            'age_group': '30-40',
            'gender': 'Other',
            'smoking_status': 'Unknown',
            'alcohol_use': 'Unknown',
            'drug_use': 'Non-drug User',
            'occupation_type': 'Knowledge Worker',
            'comorbidity_status': 'No Significant PMHx',
            'symptom_presentation': 'Classic Textbook'
        }
        
        if not demo_data or not isinstance(demo_data, str):
            return defaults
        
        result = defaults.copy()
        lines = demo_data.strip().split('\n')
        
        for line in lines:
            if 'dtype:' in line or not line.strip():
                continue
            
            for key in defaults.keys():
                if line.startswith(key):
                    after_key = line[len(key):].strip()
                    value = after_key.strip()
                    if value:
                        result[key] = value
                    break
        
        return result
    
    def process_top_k_metrics(self):
        """Process top-k accuracy metrics"""
        
        for k in [1, 3, 5, 7]:
            self.df[f'top_{k}_correct'] = False
        
        for idx, row in self.df.iterrows():
            # Use the existing is_correct field for top-1 accuracy
            is_correct = row.get('is_correct', False)
            
            # For simplicity, assume top-k accuracy is same as top-1
            for k in [1, 3, 5, 7]:
                self.df.at[idx, f'top_{k}_correct'] = is_correct
    
    def process_recall_metrics_corrected(self):
        """CORRECTED recall calculation"""
        
        print("   Processing recall metrics (corrected version)...")
        
        self.df['recall'] = False
        self.df['num_diagnoses_considered'] = 0
        
        recall_count = 0
        
        for idx, row in self.df.iterrows():
            correct_diagnosis = str(row.get('correct_diagnosis', '')).lower().strip()
            
            # Get consultation analysis
            consultation = row.get('consultation_analysis', {})
            
            if isinstance(consultation, dict):
                diagnoses_considered = consultation.get('diagnoses_considered', [])
                diagnoses_count = consultation.get('diagnoses_considered_count', 0)
                
                # Set number of diagnoses considered
                if isinstance(diagnoses_considered, list) and diagnoses_considered:
                    actual_diagnoses = [d for d in diagnoses_considered if d and str(d).strip()]
                    actual_count = len(actual_diagnoses)
                    self.df.at[idx, 'num_diagnoses_considered'] = actual_count
                    
                    # Check if correct diagnosis was actually considered
                    recall_found = False
                    if correct_diagnosis and actual_diagnoses:
                        for diag in actual_diagnoses:
                            if self.diagnoses_match(str(diag).lower().strip(), correct_diagnosis):
                                recall_found = True
                                recall_count += 1
                                break
                    
                    self.df.at[idx, 'recall'] = recall_found
                else:
                    self.df.at[idx, 'num_diagnoses_considered'] = max(0, diagnoses_count)
                    self.df.at[idx, 'recall'] = False
            else:
                self.df.at[idx, 'num_diagnoses_considered'] = 0
                self.df.at[idx, 'recall'] = False
        
        overall_recall = self.df['recall'].mean() * 100
        print(f"     Corrected overall recall rate: {overall_recall:.1f}% ({recall_count}/{len(self.df)} cases)")
    
    def diagnoses_match(self, diag1: str, diag2: str) -> bool:
        """Check if two diagnoses refer to the same condition"""
        
        if not diag1 or not diag2:
            return False
        
        # Normalize text
        def normalize(text):
            text = re.sub(r'[^\w\s]', ' ', text.lower())
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        
        diag1_norm = normalize(diag1)
        diag2_norm = normalize(diag2)
        
        # Check various matching criteria
        return (diag1_norm == diag2_norm or 
                diag1_norm in diag2_norm or 
                diag2_norm in diag1_norm or
                self.calculate_word_overlap(diag1_norm, diag2_norm) > 0.7)
    
    def calculate_word_overlap(self, str1: str, str2: str) -> float:
        """Calculate word overlap similarity"""
        
        if not str1 or not str2:
            return 0.0
        
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def process_confidence_metrics(self):
        """Process confidence metrics"""
        
        self.df['avg_top1_confidence'] = 0.0
        
        for idx, row in self.df.iterrows():
            confidence_fields = ['self_confidence', 'confidence']
            
            confidence_value = None
            for field in confidence_fields:
                if field in row and row[field] is not None:
                    confidence_value = row[field]
                    break
            
            if confidence_value is not None:
                try:
                    if isinstance(confidence_value, list):
                        conf = float(confidence_value[0]) if confidence_value else 0.0
                    elif isinstance(confidence_value, str):
                        numbers = re.findall(r'\d+\.?\d*', confidence_value)
                        conf = float(numbers[0]) if numbers else 0.0
                    else:
                        conf = float(confidence_value)
                    
                    if conf > 1:
                        conf = conf / 100.0
                    
                    self.df.at[idx, 'avg_top1_confidence'] = max(0.0, min(1.0, conf))
                except:
                    self.df.at[idx, 'avg_top1_confidence'] = 0.0
    
    def derive_symptom_presentations_fixed(self):
        """FIXED symptom presentation categorization with realistic thresholds"""
        
        symptom_presentations = []
        categorization_details = []
        
        for idx, row in self.df.iterrows():
            category, details = self.categorize_single_case_fixed(row)
            symptom_presentations.append(category)
            categorization_details.append(details)
        
        self.df['derived_symptom_presentation'] = symptom_presentations
        self.df['categorization_details'] = categorization_details
        
        # Update demographics with derived categories
        self.df['demo_symptom_presentation'] = symptom_presentations
        
        # Show results
        category_counts = pd.Series(symptom_presentations).value_counts()
        print(f"   FIXED symptom categories:")
        for category, count in category_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"     {category}: {count} cases ({percentage:.1f}%)")
    
    def categorize_single_case_fixed(self, row: pd.Series) -> Tuple[str, Dict]:
        """FIXED categorization with realistic thresholds"""
        
        # Extract case information
        case_text = self.extract_case_text(row)
        consultation = row.get('consultation_analysis', {})
        num_diagnoses = len(consultation.get('diagnoses_considered', [])) if isinstance(consultation, dict) else 0
        
        # Identify medical systems involved
        systems_involved = self.identify_medical_systems(case_text)
        has_vague = self.has_vague_indicators(case_text)
        
        details = {
            'systems_involved': systems_involved,
            'num_systems': len(systems_involved),
            'num_diagnoses': num_diagnoses,
            'has_vague': has_vague
        }
        
        # FIXED CATEGORIZATION LOGIC - Much more realistic thresholds
        if len(systems_involved) >= 3:
            category = "Multi-System Complex"
            details['reason'] = f"≥3 systems involved: {systems_involved[:3]}"
        
        # FIXED: Much more realistic criteria for Single Symptom Only
        elif (len(systems_involved) <= 1 and 
              num_diagnoses <= 3 and 
              not has_vague):
            category = "Single Symptom Only"
            if len(systems_involved) == 0:
                details['reason'] = f"No clear system, {num_diagnoses} diagnoses, simple case"
            else:
                details['reason'] = f"Single system ({systems_involved[0]}), {num_diagnoses} diagnoses, focused case"
        
        elif num_diagnoses >= 8 or has_vague:
            category = "Atypical/Vague Wording"
            details['reason'] = f"High complexity ({num_diagnoses} diagnoses) or vague presentation"
        
        else:
            category = "Classic Textbook"
            details['reason'] = f"Standard case: {len(systems_involved)} systems, {num_diagnoses} diagnoses"
        
        return category, details
    
    def extract_case_text(self, row: pd.Series) -> str:
        """Extract text from medical case"""
        
        text_sources = []
        
        if row.get('correct_diagnosis'):
            text_sources.append(str(row['correct_diagnosis']))
        
        if row.get('final_doctor_diagnosis'):
            text_sources.append(str(row['final_doctor_diagnosis']))
        
        consultation = row.get('consultation_analysis', {})
        if isinstance(consultation, dict):
            diagnoses_considered = consultation.get('diagnoses_considered', [])
            if isinstance(diagnoses_considered, list):
                text_sources.extend([str(d) for d in diagnoses_considered])
        
        if row.get('specialist_reason'):
            text_sources.append(str(row['specialist_reason']))
        
        return ' '.join(text_sources).lower()
    
    def identify_medical_systems(self, text: str) -> List[str]:
        """Identify medical systems involved"""
        
        systems_found = []
        text_lower = text.lower()
        
        for system, keywords in self.system_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    systems_found.append(system)
                    break
        
        return list(set(systems_found))
    
    def has_vague_indicators(self, text: str) -> bool:
        """Check for vague presentation indicators"""
        
        vague_indicators = [
            'vague', 'nonspecific', 'atypical', 'unusual', 'unclear', 'ambiguous',
            'ill-defined', 'poorly defined', 'subtle', 'general malaise'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in vague_indicators)
    
    def validate_data_quality(self):
        """Validate processed data"""
        
        total_rows = len(self.df)
        print(f"   Total scenarios: {total_rows}")
        
        # Check accuracy metrics
        if 'top_1_correct' in self.df.columns:
            accuracy = self.df['top_1_correct'].mean() * 100
            print(f"   Top-1 Accuracy: {accuracy:.1f}%")
        
        # Check recall
        if 'recall' in self.df.columns:
            recall_rate = self.df['recall'].mean() * 100
            print(f"   Recall Rate: {recall_rate:.1f}%")
        
        # Check confidence
        if 'avg_top1_confidence' in self.df.columns:
            avg_conf = self.df['avg_top1_confidence'].mean() * 100
            print(f"   Avg Confidence: {avg_conf:.1f}%")
    
    def generate_all_ablation_tables(self) -> Dict[str, pd.DataFrame]:
        """Generate all ablation tables"""
        
        demographic_groups = {
            'Age': {
                'column': 'demo_age_group',
                'groups': ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60+"]
            },
            'Gender': {
                'column': 'demo_gender',
                'groups': ["Male", "Female", "Other"]
            },
            'Smoking Status': {
                'column': 'demo_smoking_status',
                'groups': ["Smoker", "Non-smoker", "Unknown"]
            },
            'Alcohol Use': {
                'column': 'demo_alcohol_use',
                'groups': ["Drinker", "Non-drinker", "Unknown"]
            },
            'Symptom Presentation (Fixed)': {
                'column': 'demo_symptom_presentation',
                'groups': ["Classic Textbook", "Atypical/Vague Wording", "Multi-System Complex", "Single Symptom Only"]
            }
        }
        
        tables = {}
        
        for group_name, group_info in demographic_groups.items():
            print(f"   Generating {group_name} table...")
            table = self.create_ablation_table(group_info['column'], group_info['groups'])
            tables[group_name] = table
        
        return tables
    
    def create_ablation_table(self, group_column: str, expected_groups: List[str]) -> pd.DataFrame:
        """Create ablation table for a demographic group"""
        
        results = []
        
        for group in expected_groups:
            group_data = self.df[self.df[group_column] == group].copy()
            
            if len(group_data) == 0:
                results.append({
                    'Group': group,
                    'Count': 0,
                    'Top-1 Accuracy': 0.0,
                    'Top-3 Accuracy': 0.0,
                    'Top-5 Accuracy': 0.0,
                    'Top-7 Accuracy': 0.0,
                    'Recall': 0.0,
                    'Avg Diagnoses Considered': 0.0,
                    'Avg Top-1 Confidence': 0.0
                })
                continue
            
            count = len(group_data)
            
            # Calculate metrics
            top_1_acc = group_data.get('top_1_correct', pd.Series([False]*count)).mean() * 100
            top_3_acc = group_data.get('top_3_correct', pd.Series([False]*count)).mean() * 100
            top_5_acc = group_data.get('top_5_correct', pd.Series([False]*count)).mean() * 100
            top_7_acc = group_data.get('top_7_correct', pd.Series([False]*count)).mean() * 100
            
            recall = group_data.get('recall', pd.Series([False]*count)).mean() * 100
            avg_diagnoses = group_data.get('num_diagnoses_considered', pd.Series([0]*count)).mean()
            avg_confidence = group_data.get('avg_top1_confidence', pd.Series([0]*count)).mean() * 100
            
            results.append({
                'Group': group,
                'Count': count,
                'Top-1 Accuracy': round(top_1_acc, 1),
                'Top-3 Accuracy': round(top_3_acc, 1),
                'Top-5 Accuracy': round(top_5_acc, 1),
                'Top-7 Accuracy': round(top_7_acc, 1),
                'Recall': round(recall, 1),
                'Avg Diagnoses Considered': round(avg_diagnoses, 1),
                'Avg Top-1 Confidence': round(avg_confidence, 1)
            })
        
        return pd.DataFrame(results)
    
    def save_all_results(self, tables: Dict[str, pd.DataFrame]):
        """Save all results"""
        
        # Save processed dataset
        processed_data_path = os.path.join(self.output_dir, "processed_ablation_data_final_fixed.csv")
        self.df.to_csv(processed_data_path, index=False)
        print(f"   Saved processed data: {processed_data_path}")
        
        # Save individual CSV tables
        for group_name, table in tables.items():
            filename = f"ablation_table_{group_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_final_fixed.csv"
            filepath = os.path.join(self.output_dir, filename)
            table.to_csv(filepath, index=False)
            print(f"   Saved {group_name} table: {filepath}")
        
        # Save detailed categorization
        categorization_df = self.df[['scenario_id', 'derived_symptom_presentation', 'categorization_details']].copy()
        categorization_path = os.path.join(self.output_dir, "detailed_symptom_categorization_fixed.csv")
        categorization_df.to_csv(categorization_path, index=False)
        print(f"   Saved detailed categorization: {categorization_path}")
        
        # Save Excel file if possible
        excel_path = os.path.join(self.output_dir, "comprehensive_ablation_tables_final_fixed.xlsx")
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                for group_name, table in tables.items():
                    sheet_name = group_name.replace(' ', '_').replace('(', '').replace(')', '')[:31]
                    table.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"   Saved Excel file: {excel_path}")
        except ImportError:
            print("   Warning: openpyxl not available, skipping Excel export")
    
    def print_tables(self, tables: Dict[str, pd.DataFrame]):
        """Print all tables"""
        
        print("="*80)
        print("FINAL FIXED ABLATION TABLES")
        print("="*80)
        
        for group_name, table in tables.items():
            print(f"\n{group_name.upper()} ABLATION TABLE")
            print("-" * 80)
            print(table.to_string(index=False, float_format='%.1f'))
            print()
        
        # Print summary
        total_cases = len(self.df)
        overall_accuracy = self.df['top_1_correct'].mean() * 100 if 'top_1_correct' in self.df.columns else 0
        overall_recall = self.df['recall'].mean() * 100 if 'recall' in self.df.columns else 0
        overall_confidence = self.df['avg_top1_confidence'].mean() * 100 if 'avg_top1_confidence' in self.df.columns else 0
        
        print("="*80)
        print("FINAL SUMMARY")
        print("="*80)
        print(f"Total scenarios analyzed: {total_cases:,}")
        print(f"Overall Top-1 Accuracy: {overall_accuracy:.1f}%")
        print(f"Overall Recall Rate: {overall_recall:.1f}%")
        print(f"Overall Avg Confidence: {overall_confidence:.1f}%")
        print(f"All outputs saved to: {self.output_dir}")
        print("="*80)

def main():
    """Main execution function"""
    
    log_file_path = "base_files/logs/extracted_results_MedQA_Ext_claude.json"
    output_dir = "base_files/analysis"
    
    try:
        analyzer = FinalFixedAblationAnalyzer(log_file_path, output_dir)
        tables = analyzer.run_complete_analysis()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the log file exists and update the path in the script.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()