from .util import query_model
from .prompts import ALL_BIASES

# --- Agent Classes ---
class Agent:
    def __init__(self, scenario=None):
        self.scenario = scenario
        self.agent_hist = ""
        self.reset()
    
    def reset(self):
        self.agent_hist = ""
        self._init_data()
    
    def _init_data(self):
        # To be implemented by subclasses
        pass
    
    def add_hist(self, hist_str):
        self.agent_hist += hist_str + "\n\n"
    
    def system_prompt(self):
        # To be implemented by subclasses
        return ""

class PatientAgent(Agent):
    def _init_data(self):
        self.symptoms = self.scenario.patient_information()
    
    def system_prompt(self):
        base = """You are a patient in a clinic who only responds in the form of dialogue. You are being inspected by a doctor who will ask you questions and will perform exams on you in order to understand your disease. Your answer will only be 1-3 sentences in length."""
        symptoms = f"\n\nBelow is all of your information. {self.symptoms}. \n\n Remember, you must not reveal your disease explicitly but may only convey the symptoms you have in the form of dialogue if you are asked. You must ONLY draw from the information provided to you and must not make up any information or infer symptoms that are not provided."
        return base + symptoms
    
    def inference_patient(self, question):
        prompt = f"\nHere is a history of your dialogue:\n{self.agent_hist}\nHere was the doctor response:\n{question}\nNow please continue your dialogue\nPatient: "
        answer = query_model(prompt, self.system_prompt())
        print("Prompt: ", prompt)
        return answer

class DoctorAgent(Agent):
    def __init__(self, scenario=None, max_infs=20, bias=None):
        self.MAX_INFS = max_infs
        self.infs = 0
        self.specialist_type = None
        self.consultation_turns = 0
        self.bias = bias  # New bias parameter
        super().__init__(scenario)
    
    def _init_data(self):
        self.presentation = self.scenario.examiner_information()
    
    def system_prompt(self):
        base = f"You are a doctor named Dr. Agent who only responds in the form of dialogue. You are inspecting a patient who you will ask questions in order to understand their disease. You are only allowed to ask {self.MAX_INFS} questions total before you must make a decision. You have asked {self.infs} questions so far. You can request test results using the format \"REQUEST TEST: [test]\". For example, \"REQUEST TEST: Chest_X-Ray\". Your dialogue will only be 1-3 sentences in length. Once you have decided to make a diagnosis please type \"DIAGNOSIS READY: [diagnosis here]\""
        presentation = f"\n\nBelow is all of the information you have. {self.presentation}. \n\n Remember, you must accomplish your objective by asking them questions. You are also able to provide exams."
        
        # Add bias prompt if specified
        bias_text = ""
        if self.bias and self.bias in ALL_BIASES:
            bias_text = f"\n\nIMPORTANT: {ALL_BIASES[self.bias]['prompt']}"
        
        return base + presentation + bias_text
    
    def determine_specialist(self):
        """Queries the LLM to determine the best specialist based on dialogue history."""
        prompt = f"Based on the following patient interaction history, what type of medical specialist (e.g., Cardiologist, Neurologist, Pulmonologist, Gastroenterologist, Endocrinologist, Infectious Disease Specialist, Oncologist, etc.) would be most appropriate to consult for a potential diagnosis? Please respond with only the specialist type.\n\nHistory:\n{self.agent_hist}"
        system_prompt = "You are a helpful medical assistant. Analyze the dialogue and suggest the single most relevant medical specialist type."
        specialist = query_model(prompt, system_prompt)
        self.specialist_type = specialist.replace("Specialist", "").strip()
        explanation_prompt = f"Explain why a {self.specialist_type} is the most appropriate specialist based on the following dialogue history:\n\n{self.agent_hist}"
        explanation = query_model(explanation_prompt, system_prompt)
        print(f"Doctor decided to consult: {self.specialist_type}")
        print(f"Reason for choice: {explanation}")
        return self.specialist_type, explanation

    def inference_doctor(self, last_response, mode="patient"):
        """Generates the doctor's response, adapting to patient interaction or specialist consultation."""
        if mode == "patient":
             if self.infs > 0 or "Patient presents with initial information." not in last_response:
                 self.add_hist(f"Patient: {last_response}")
        elif mode == "consultation":
             self.add_hist(f"Specialist ({self.specialist_type}): {last_response}")

        if mode == "patient":
            if self.infs >= self.MAX_INFS:
                 return "Okay, I have gathered enough information from the patient. I need to analyze this and potentially consult a specialist.", "consultation_needed"

            prompt = f"\nHere is a history of your dialogue with the patient:\n{self.agent_hist}\nHere was the patient response:\n{last_response}\nNow please continue your dialogue with the patient. You have {self.MAX_INFS - self.infs} questions remaining for the patient. Remember you can REQUEST TEST: [test].\nDoctor: "
            system_prompt = f"You are a doctor named Dr. Agent interacting with a patient. You have {self.MAX_INFS - self.infs} questions left. Your goal is to gather information. {self.presentation}"
            answer = query_model(prompt, system_prompt)
            self.add_hist(f"Doctor: {answer}")
            self.infs += 1
            if "DIAGNOSIS READY:" in answer:
                answer = "Let me gather a bit more information first."
            return answer, "patient_interaction"

        elif mode == "consultation":
            prompt = f"\nHere is the full history (Patient interaction followed by consultation):\n{self.agent_hist}\nYou are consulting with a {self.specialist_type}.\nHere was the specialist's latest response:\n{last_response}\nContinue the consultation. Ask questions or share your thoughts to refine the diagnosis.\nDoctor: "
            system_prompt = f"You are Dr. Agent, consulting with a {self.specialist_type} about a patient case. Discuss the findings and differential diagnoses based on the history provided. Aim to reach a conclusion."
            answer = query_model(prompt, system_prompt)
            self.add_hist(f"Doctor: {answer}")
            self.consultation_turns += 1
            if "DIAGNOSIS READY:" in answer:
                 pass
            return answer, "consultation"

    def get_final_diagnosis(self):
        """Generates the final diagnosis prompt after all interactions."""
        prompt = f"\nHere is the complete history of your dialogue with the patient and the specialist ({self.specialist_type}):\n{self.agent_hist}\nBased on this entire consultation, please provide your final diagnosis and your confidence level in percent now in the format 'DIAGNOSIS READY: [Your Diagnosis Here] \n CONFIDENCE: [Your Confidence % here]'."
        system_prompt = f"You are Dr. Agent. You have finished interviewing the patient and consulting with a {self.specialist_type}. Review the entire history and provide your single, most likely final diagnosis in the required format."
        response = query_model(prompt, system_prompt)

        diagnosis_text = ""
        confidence = ""

        if "DIAGNOSIS READY:" in response:
            diagnosis_text = response.split("DIAGNOSIS READY:")[1].split("CONFIDENCE:")[0].strip()

        if "CONFIDENCE:" in response:
            confidence = response.split("CONFIDENCE:")[1].strip()

        return {
            "diagnosis": diagnosis_text,
            "confidence": confidence
        }


class MeasurementAgent(Agent):
    def _init_data(self):
        self.information = self.scenario.exam_information()
    
    def system_prompt(self):
        base = "You are an measurement reader who responds with medical test results. Please respond in the format \"RESULTS: [results here]\""
        presentation = f"\n\nBelow is all of the information you have. {self.information}. \n\n If the requested results are not in your data then you can respond with NORMAL READINGS."
        return base + presentation
    
    def inference_measurement(self, doctor_request):
        prompt = f"\nHere is a history of the dialogue:\n{self.agent_hist}\nHere was the doctor measurement request:\n{doctor_request}"
        answer = query_model(prompt, self.system_prompt())
        return answer

# --- Specialist Agent Class ---
class SpecialistAgent(Agent):
    def __init__(self, scenario=None, specialty="General Medicine"):
        self.specialty = specialty
        super().__init__(scenario)
        self.information = scenario.exam_information()

    def _init_data(self):
        pass

    def system_prompt(self):
        base = f"You are a consulting specialist in {self.specialty}. You are discussing a case with the primary doctor (Dr. Agent). Review the provided dialogue history and the doctor's latest message. Provide your expert opinion, ask clarifying questions, or suggest next steps/differential diagnoses. Respond concisely (1-3 sentences) as dialogue."
        return base

    def inference_specialist(self, doctor_consult_message):
        self.add_hist(f"Doctor: {doctor_consult_message}")

        prompt = f"\nHere is the history of the case discussion:\n{self.agent_hist}\nHere was the primary doctor's latest message:\n{doctor_consult_message}\nPlease provide your specialist input.\nSpecialist ({self.specialty}): "
        answer = query_model(prompt, self.system_prompt())

        self.add_hist(f"Specialist ({self.specialty}): {answer}")
        return answer