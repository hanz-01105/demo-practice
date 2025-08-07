import anthropic, re, random, time, json, os
from datetime import datetime
import argparse
import glob
from .util import query_model
import pandas as pd

# --- FIX: ROBUST PATH SETUP ---
# Get the absolute path of the directory containing this script (e.g., '.../prototype/')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The 'base_files' directory is one level up from the script's directory
BASE_FILES_DIR = os.path.dirname(SCRIPT_DIR)
# --- END FIX ---


# --- Base Scenario Class ---
class BaseScenario:
    def __init__(self, scenario_dict):
        self.scenario_dict = scenario_dict
        self._init_data()
    
    def _init_data(self):
        pass
    
    def patient_information(self):
        return self.patient_info
    
    def examiner_information(self):
        return self.examiner_info
    
    def exam_information(self):
        return self.exam_info
    
    def diagnosis_information(self):
        return str(self.diagnosis)

# --- Concrete Scenario Classes ---
class ScenarioMedQA(BaseScenario):
    def _init_data(self):
        self.tests = self.scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = self.scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info = self.scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info = self.scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = self.scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]
        self.exam_info = {**self.physical_exams, "tests": self.tests}
    
    def get_available_tests(self):
        return list(self.tests.keys())

class ScenarioNEJM(BaseScenario):
    def _init_data(self):
        self.question = self.scenario_dict["question"]
        self.diagnosis = [_sd["text"] for _sd in self.scenario_dict["answers"] if _sd["correct"]][0]
        self.patient_info = self.scenario_dict["patient_info"]
        self.examiner_info = "What is the most likely diagnosis?"
        self.physical_exams = self.scenario_dict["physical_exams"]
        self.exam_info = self.physical_exams
    
    def get_available_tests(self):
        # Use LLM to extract test names
        prompt = f"Extract the list of medical tests mentioned in the following text:\n\n{self.physical_exams}\n\nRespond with a comma-separated list of test names."
        system_prompt = "You are a medical assistant. Extract test names from the provided text."
        response = query_model(prompt, system_prompt)
        return [test.strip() for test in response.split(",") if test.strip()]

# --- Scenario Loader ---
class ScenarioLoader:
    def __init__(self, dataset):
        self.dataset = dataset
        
        # --- FIX: Build absolute paths to data files ---
        data_dir = os.path.join(BASE_FILES_DIR, 'data')
        
        self.scenario_map = {
            "MedQA": (ScenarioMedQA, os.path.join(data_dir, "templates/agentclinic_medqa.jsonl")),
            "MedQA_Ext": (ScenarioMedQA, os.path.join(data_dir, "agentclinic_medqa_extended.jsonl")),
            "NEJM": (ScenarioNEJM, os.path.join(data_dir, "templates/agentclinic_nejm.jsonl")),
            "NEJM_Ext": (ScenarioNEJM, os.path.join(data_dir, "templates/agentclinic_nejm_extended.jsonl")),
        }
        # --- END FIX ---
        
        if dataset not in self.scenario_map:
            raise ValueError(f"Dataset '{dataset}' not recognized. Choices are: {list(self.scenario_map.keys())}")
        
        self._load_scenarios()
    
    def _load_scenarios(self):
        scenario_class, filename = self.scenario_map[self.dataset]
        
        # This 'open' call will now use the full, correct path
        with open(filename, "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [scenario_class(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def get_scenario(self, id=None):
        if not self.scenarios:
            return None
        if id is None:
            return random.choice(self.scenarios)
        if 0 <= id < self.num_scenarios:
            return self.scenarios[id]
        else:
            print(f"Warning: Scenario ID {id} out of range (0-{self.num_scenarios-1}). Returning None.")
            return None