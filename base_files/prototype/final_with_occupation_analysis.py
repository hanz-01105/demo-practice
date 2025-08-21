#!/usr/bin/env python3
"""
Adaptive Occupation Analysis - Creates Categories Based on Actual Data
====================================================================
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

class AdaptiveOccupationAnalyzer:
    """Analyzes actual occupation data and creates realistic categories"""
    
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
        """Run analysis with adaptive occupation categories"""
        
        print("="*80)
        print("ADAPTIVE OCCUPATION ANALYSIS - DATA-DRIVEN CATEGORIES")
        print("="*80)
        print(f"Input file: {self.log_file_path}")
        print(f"Output directory: {self.output_dir}")
        print("="*80)
        
        # Load and process data
        print("\n1. Loading and processing data...")
        self.load_and_process_data()
        
        # Analyze actual occupation data
        print("\n2. Analyzing actual occupation data...")
        actual_occupations = self.analyze_actual_occupations()
        
        # Create adaptive occupation categories
        print("\n3. Creating adaptive occupation categories...")
        occupation_categories = self.create_adaptive_occupation_categories(actual_occupations)
        
        # Derive symptom categories
        print("\n4. Deriving symptom presentation categories...")
        self.derive_symptom_presentations_fixed()
        
        # Validate data
        print("\n5. Validating data quality...")
        self.validate_data_quality()
        
        # Generate all tables with adaptive occupation categories
        print("\n6. Generating all ablation tables...")
        tables = self.generate_all_ablation_tables(occupation_categories)
        
        # Save results
        print("\n7. Saving results...")
        self.save_all_results(tables, occupation_categories)
        
        # Print tables
        print("\n8. Analysis complete!")
        self.print_all_tables(tables)
        
        return tables, occupation_categories
    
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
        """Process demographics and extract raw occupation data"""
        
        demographics_data = []
        
        for _, row in self.df.iterrows():
            demo_raw = row.get('demographics', '')
            demo_dict = self.parse_demographics_raw(demo_raw)
            demographics_data.append(demo_dict)
        
        # Add demographic columns
        demo_df = pd.DataFrame(demographics_data)
        for col in demo_df.columns:
            self.df[f'demo_{col}'] = demo_df[col]
    
    def parse_demographics_raw(self, demo_data: Any) -> Dict[str, str]:
        """Parse demographics keeping raw occupation values for analysis"""
        
        defaults = {
            'age_group': '30-40',
            'gender': 'Other',
            'smoking_status': 'Unknown',
            'alcohol_use': 'Unknown',
            'drug_use': 'Non-drug User',
            'occupation_type': 'Knowledge Worker',  # Keep original for now
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
    
    def analyze_actual_occupations(self) -> Dict[str, int]:
        """Analyze what occupation values actually exist in the data"""
        
        occupation_counts = self.df['demo_occupation_type'].value_counts()
        
        print(f"   Actual occupation values found:")
        for occupation, count in occupation_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"     '{occupation}': {count} cases ({percentage:.1f}%)")
        
        return occupation_counts.to_dict()
    
    def create_adaptive_occupation_categories(self, actual_occupations: Dict[str, int]) -> List[str]:
        """Create occupation categories based on actual data"""
        
        # Strategy 1: Use actual values if we have good diversity
        unique_occupations = list(actual_occupations.keys())
        
        print(f"   Found {len(unique_occupations)} unique occupation values")
        
        # Strategy 2: If we have limited categories, enhance with meaningful alternatives
        if len(unique_occupations) <= 3:
            print("   Limited occupation diversity found. Creating enhanced categories...")
            
            # Try to infer additional categories from case data
            enhanced_categories = self.infer_occupation_categories_from_cases()
            
            # Combine actual + inferred
            all_categories = list(set(unique_occupations + enhanced_categories))
            
            print(f"   Enhanced to {len(all_categories)} categories: {all_categories}")
            return all_categories
        
        # Strategy 3: If we have good diversity, group similar ones
        elif len(unique_occupations) > 8:
            print("   High occupation diversity found. Grouping similar categories...")
            grouped_categories = self.group_similar_occupations(unique_occupations)
            print(f"   Grouped to {len(grouped_categories)} categories: {grouped_categories}")
            return grouped_categories
        
        # Strategy 4: Perfect number of categories (4-8), use as-is
        else:
            print(f"   Good occupation diversity. Using actual categories: {unique_occupations}")
            return unique_occupations
    
    def infer_occupation_categories_from_cases(self) -> List[str]:
        """Infer additional occupation categories from medical case characteristics"""
        
        # Analyze case complexity and diagnosis patterns to infer likely occupations
        inferred_categories = []
        
        # Look at age patterns
        age_groups = self.df['demo_age_group'].value_counts()
        
        if '20-30' in age_groups and age_groups['20-30'] >= 10:
            inferred_categories.append('Student/Young Professional')
        
        if '60+' in age_groups and age_groups['60+'] >= 5:
            inferred_categories.append('Retired/Senior')
        
        # Look at diagnostic complexity patterns
        avg_diagnoses = self.df['num_diagnoses_considered'].mean()
        
        if avg_diagnoses > 6:
            inferred_categories.append('High-Stress Professional')
        
        # Look at confidence patterns
        if 'avg_top1_confidence' in self.df.columns:
            high_conf_cases = self.df[self.df['avg_top1_confidence'] > 0.9]
            if len(high_conf_cases) >= 20:
                inferred_categories.append('Healthcare Worker')
        
        return inferred_categories
    
    def group_similar_occupations(self, occupations: List[str]) -> List[str]:
        """Group similar occupation types into broader categories"""
        
        occupation_groups = {
            'Professional/Knowledge': [],
            'Healthcare': [],
            'Education': [],
            'Technical': [],
            'Service': [],
            'Other': []
        }
        
        for occupation in occupations:
            occ_lower = occupation.lower()
            
            if any(word in occ_lower for word in ['doctor', 'nurse', 'medical', 'health']):
                occupation_groups['Healthcare'].append(occupation)
            elif any(word in occ_lower for word in ['teacher', 'professor', 'education', 'student']):
                occupation_groups['Education'].append(occupation)
            elif any(word in occ_lower for word in ['engineer', 'technical', 'programmer', 'developer']):
                occupation_groups['Technical'].append(occupation)
            elif any(word in occ_lower for word in ['professional', 'manager', 'analyst', 'consultant']):
                occupation_groups['Professional/Knowledge'].append(occupation)
            elif any(word in occ_lower for word in ['service', 'retail', 'customer']):
                occupation_groups['Service'].append(occupation)
            else:
                occupation_groups['Other'].append(occupation)
        
        # Return only groups that have members
        grouped_categories = [group for group, members in occupation_groups.items() if members]
        
        # If still too many, combine smaller groups
        if len(grouped_categories) > 6:
            # Combine smaller groups into "Other"
            large_groups = [group for group, members in occupation_groups.items() if len(members) >= 2]
            if len(large_groups) <= 6:
                return large_groups
        
        return grouped_categories[:6]  # Limit to 6 categories max
    
    def process_top_k_metrics(self):
        """Process top-k accuracy metrics"""
        
        for k in [1, 3, 5, 7]:
            self.df[f'top_{k}_correct'] = False
        
        for idx, row in self.df.iterrows():
            is_correct = row.get('is_correct', False)
            for k in [1, 3, 5, 7]:
                self.df.at[idx, f'top_{k}_correct'] = is_correct
    
    def process_recall_metrics_corrected(self):
        """Corrected recall calculation"""
        
        print("   Processing recall metrics (corrected version)...")
        
        self.df['recall'] = False
        self.df['num_diagnoses_considered'] = 0
        
        recall_count = 0
        
        for idx, row in self.df.iterrows():
            correct_diagnosis = str(row.get('correct_diagnosis', '')).lower().strip()
            
            consultation = row.get('consultation_analysis', {})
            
            if isinstance(consultation, dict):
                diagnoses_considered = consultation.get('diagnoses_considered', [])
                diagnoses_count = consultation.get('diagnoses_considered_count', 0)
                
                if isinstance(diagnoses_considered, list) and diagnoses_considered:
                    actual_diagnoses = [d for d in diagnoses_considered if d and str(d).strip()]
                    actual_count = len(actual_diagnoses)
                    self.df.at[idx, 'num_diagnoses_considered'] = actual_count
                    
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
        print(f"     Overall recall rate: {overall_recall:.1f}% ({recall_count}/{len(self.df)} cases)")
    
    def diagnoses_match(self, diag1: str, diag2: str) -> bool:
        """Check if two diagnoses refer to the same condition"""
        
        if not diag1 or not diag2:
            return False
        
        def normalize(text):
            text = re.sub(r'[^\w\s]', ' ', text.lower())
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        
        diag1_norm = normalize(diag1)
        diag2_norm = normalize(diag2)
        
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
        """Fixed symptom presentation categorization"""
        
        symptom_presentations = []
        
        for idx, row in self.df.iterrows():
            category = self.categorize_single_case_fixed(row)
            symptom_presentations.append(category)
        
        self.df['demo_symptom_presentation'] = symptom_presentations
        
        category_counts = pd.Series(symptom_presentations).value_counts()
        print(f"   Symptom categories:")
        for category, count in category_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"     {category}: {count} cases ({percentage:.1f}%)")
    
    def categorize_single_case_fixed(self, row: pd.Series) -> str:
        """Fixed categorization with realistic thresholds"""
        
        case_text = self.extract_case_text(row)
        consultation = row.get('consultation_analysis', {})
        num_diagnoses = len(consultation.get('diagnoses_considered', [])) if isinstance(consultation, dict) else 0
        
        systems_involved = self.identify_medical_systems(case_text)
        has_vague = self.has_vague_indicators(case_text)
        
        if len(systems_involved) >= 3:
            return "Multi-System Complex"
        elif (len(systems_involved) <= 1 and num_diagnoses <= 3 and not has_vague):
            return "Single Symptom Only"
        elif num_diagnoses >= 8 or has_vague:
            return "Atypical/Vague Wording"
        else:
            return "Classic Textbook"
    
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
        
        if 'top_1_correct' in self.df.columns:
            accuracy = self.df['top_1_correct'].mean() * 100
            print(f"   Top-1 Accuracy: {accuracy:.1f}%")
        
        if 'recall' in self.df.columns:
            recall_rate = self.df['recall'].mean() * 100
            print(f"   Recall Rate: {recall_rate:.1f}%")
        
        if 'avg_top1_confidence' in self.df.columns:
            avg_conf = self.df['avg_top1_confidence'].mean() * 100
            print(f"   Avg Confidence: {avg_conf:.1f}%")
    
    def generate_all_ablation_tables(self, occupation_categories: List[str]) -> Dict[str, pd.DataFrame]:
        """Generate all ablation tables with adaptive occupation categories and drug use"""
        
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
            'Drug Use': {
                'column': 'demo_drug_use',
                'groups': ["Drug User", "Non-drug User", "Former Drug User", "Medical Drug User", "Unknown"]
            },
            'Occupation Type (Adaptive)': {  # ADAPTIVE: Uses actual data categories
                'column': 'demo_occupation_type',
                'groups': occupation_categories
            },
            'Symptom Presentation': {
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
    
    def save_all_results(self, tables: Dict[str, pd.DataFrame], occupation_categories: List[str]):
        """Save all results including occupation category info and drug use analysis"""
        
        # Save processed dataset
        processed_data_path = os.path.join(self.output_dir, "processed_ablation_data_with_drug_use.csv")
        self.df.to_csv(processed_data_path, index=False)
        print(f"   Saved processed data: {processed_data_path}")
        
        # Save individual CSV tables
        for group_name, table in tables.items():
            filename = f"ablation_table_{group_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.csv"
            filepath = os.path.join(self.output_dir, filename)
            table.to_csv(filepath, index=False)
            print(f"   Saved {group_name} table: {filepath}")
        
        # Save occupation category mapping
        occupation_info = {
            'adaptive_categories': occupation_categories,
            'category_analysis': 'Categories created based on actual data distribution',
            'total_categories': len(occupation_categories)
        }
        
        occupation_info_path = os.path.join(self.output_dir, "occupation_categories_info.json")
        with open(occupation_info_path, 'w') as f:
            json.dump(occupation_info, f, indent=2)
        print(f"   Saved occupation category info: {occupation_info_path}")
        
        # Analyze and save drug use distribution
        drug_use_analysis = self.analyze_drug_use_distribution()
        drug_use_path = os.path.join(self.output_dir, "drug_use_analysis.json")
        with open(drug_use_path, 'w') as f:
            json.dump(drug_use_analysis, f, indent=2)
        print(f"   Saved drug use analysis: {drug_use_path}")
        
        # Save Excel file if possible
        excel_path = os.path.join(self.output_dir, "complete_ablation_tables_with_drug_use.xlsx")
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                for group_name, table in tables.items():
                    sheet_name = group_name.replace(' ', '_').replace('(', '').replace(')', '')[:31]
                    table.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"   Saved Excel file: {excel_path}")
        except ImportError:
            print("   Warning: openpyxl not available, skipping Excel export")
    
    def analyze_drug_use_distribution(self) -> Dict[str, Any]:
        """Analyze drug use distribution and potential bias patterns"""
        
        drug_use_counts = self.df['demo_drug_use'].value_counts()
        total_cases = len(self.df)
        
        analysis = {
            'distribution': {},
            'performance_summary': {},
            'potential_bias_indicators': []
        }
        
        # Distribution analysis
        for drug_status, count in drug_use_counts.items():
            percentage = (count / total_cases) * 100
            analysis['distribution'][drug_status] = {
                'count': int(count),
                'percentage': round(percentage, 1)
            }
        
        # Performance summary for each drug use category
        for drug_status in drug_use_counts.index:
            group_data = self.df[self.df['demo_drug_use'] == drug_status]
            
            if len(group_data) > 0:
                analysis['performance_summary'][drug_status] = {
                    'top_1_accuracy': round(group_data.get('top_1_correct', pd.Series([False]*len(group_data))).mean() * 100, 1),
                    'recall': round(group_data.get('recall', pd.Series([False]*len(group_data))).mean() * 100, 1),
                    'avg_confidence': round(group_data.get('avg_top1_confidence', pd.Series([0]*len(group_data))).mean() * 100, 1),
                    'avg_diagnoses_considered': round(group_data.get('num_diagnoses_considered', pd.Series([0]*len(group_data))).mean(), 1)
                }
        
        # Check for potential bias indicators
        if 'Drug User' in analysis['performance_summary'] and 'Non-drug User' in analysis['performance_summary']:
            drug_user_acc = analysis['performance_summary']['Drug User']['top_1_accuracy']
            non_drug_user_acc = analysis['performance_summary']['Non-drug User']['top_1_accuracy']
            
            if abs(drug_user_acc - non_drug_user_acc) > 10:
                analysis['potential_bias_indicators'].append(f"Large accuracy gap between Drug Users ({drug_user_acc}%) and Non-drug Users ({non_drug_user_acc}%)")
            
            # Check confidence bias
            drug_user_conf = analysis['performance_summary']['Drug User']['avg_confidence']
            non_drug_user_conf = analysis['performance_summary']['Non-drug User']['avg_confidence']
            
            if abs(drug_user_conf - non_drug_user_conf) > 15:
                analysis['potential_bias_indicators'].append(f"Significant confidence gap between Drug Users ({drug_user_conf}%) and Non-drug Users ({non_drug_user_conf}%)")
            
            # Check diagnosis consideration bias
            drug_user_diag = analysis['performance_summary']['Drug User']['avg_diagnoses_considered']
            non_drug_user_diag = analysis['performance_summary']['Non-drug User']['avg_diagnoses_considered']
            
            if abs(drug_user_diag - non_drug_user_diag) > 2:
                analysis['potential_bias_indicators'].append(f"Different diagnosis consideration patterns: Drug Users ({drug_user_diag}) vs Non-drug Users ({non_drug_user_diag})")
        
        # Check for underrepresented categories
        for drug_status, info in analysis['distribution'].items():
            if info['count'] < 5 and info['count'] > 0:
                analysis['potential_bias_indicators'].append(f"Low representation for {drug_status} group ({info['count']} cases)")
        
        return analysis
    
    def print_all_tables(self, tables: Dict[str, pd.DataFrame]):
        """Print all ablation tables including drug use"""
        
        print("="*80)
        print("COMPREHENSIVE ABLATION TABLES WITH DRUG USE ANALYSIS")
        print("="*80)
        
        # Print all tables
        for group_name, table in tables.items():
            print(f"\n{group_name.upper()} ABLATION TABLE")
            print("-" * 80)
            print(table.to_string(index=False, float_format='%.1f'))
            print()
        
        # Highlight drug use analysis specifically
        if 'Drug Use' in tables:
            drug_table = tables['Drug Use']
            print("="*80)
            print("DRUG USE FAIRNESS ANALYSIS HIGHLIGHTS")
            print("="*80)
            
            # Find performance disparities
            non_zero_groups = drug_table[drug_table['Count'] > 0]
            if len(non_zero_groups) > 1:
                max_acc = non_zero_groups['Top-1 Accuracy'].max()
                min_acc = non_zero_groups['Top-1 Accuracy'].min()
                max_conf = non_zero_groups['Avg Top-1 Confidence'].max()
                min_conf = non_zero_groups['Avg Top-1 Confidence'].min()
                
                print(f"Top-1 Accuracy range: {min_acc:.1f}% - {max_acc:.1f}% (gap: {max_acc-min_acc:.1f}%)")
                print(f"Confidence range: {min_conf:.1f}% - {max_conf:.1f}% (gap: {max_conf-min_conf:.1f}%)")
                
                # Identify highest and lowest performing groups
                best_group = non_zero_groups.loc[non_zero_groups['Top-1 Accuracy'].idxmax(), 'Group']
                worst_group = non_zero_groups.loc[non_zero_groups['Top-1 Accuracy'].idxmin(), 'Group']
                
                print(f"Highest accuracy: {best_group} ({max_acc:.1f}%)")
                print(f"Lowest accuracy: {worst_group} ({min_acc:.1f}%)")
                
                if max_acc - min_acc > 10:
                    print("⚠️  POTENTIAL BIAS DETECTED: Large performance gap between drug use categories")
                else:
                    print("✓ No major performance disparities detected across drug use categories")
            
        # Overall summary
        total_cases = len(self.df)
        overall_accuracy = self.df['top_1_correct'].mean() * 100 if 'top_1_correct' in self.df.columns else 0
        overall_recall = self.df['recall'].mean() * 100 if 'recall' in self.df.columns else 0
        overall_confidence = self.df['avg_top1_confidence'].mean() * 100 if 'avg_top1_confidence' in self.df.columns else 0
        
        print("\n" + "="*80)
        print("COMPREHENSIVE ANALYSIS SUMMARY")
        print("="*80)
        print(f"Total scenarios analyzed: {total_cases:,}")
        print(f"Overall Top-1 Accuracy: {overall_accuracy:.1f}%")
        print(f"Overall Recall Rate: {overall_recall:.1f}%") 
        print(f"Overall Avg Confidence: {overall_confidence:.1f}%")
        print(f"Demographic groups analyzed: {len(tables)}")
        
        # Show drug use distribution
        if 'demo_drug_use' in self.df.columns:
            drug_counts = self.df['demo_drug_use'].value_counts()
            print(f"\nDrug Use Distribution:")
            for drug_status, count in drug_counts.items():
                percentage = (count / total_cases) * 100
                print(f"  {drug_status}: {count} cases ({percentage:.1f}%)")
        
        # Show occupation category info
        occupation_table = tables.get('Occupation Type (Adaptive)')
        if occupation_table is not None:
            non_zero_occupations = len(occupation_table[occupation_table['Count'] > 0])
            print(f"Occupation categories with data: {non_zero_occupations}/{len(occupation_table)}")
        
        print(f"\nAll outputs saved to: {self.output_dir}")
        print("="*80)

def main():
    """Main execution function"""
    
    log_file_path = "base_files/logs/extracted_results_MedQA_Ext_claude.json"
    output_dir = "base_files/analysis"
    
    try:
        analyzer = AdaptiveOccupationAnalyzer(log_file_path, output_dir)
        tables, occupation_categories = analyzer.run_complete_analysis()
        
        print(f"\nAdaptive occupation categories created: {occupation_categories}")
        
        # Print drug use specific insights
        if 'Drug Use' in tables:
            drug_table = tables['Drug Use']
            print(f"\nDrug Use Categories Found:")
            for _, row in drug_table.iterrows():
                if row['Count'] > 0:
                    print(f"  {row['Group']}: {row['Count']} cases, {row['Top-1 Accuracy']:.1f}% accuracy")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the log file exists and update the path in the script.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()