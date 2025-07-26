from util import query_model
from prompts import ALL_BIASES

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
        base = """You are a patient in a clinic who only responds in the form of dialogue. You are being inspected by a doctor who will ask you questions and will perform exams on you in order to understand your condition. Your answer will only be 1-3 sentences in length."""
        symptoms = f"\n\nBelow is all of your information. {self.symptoms}. \n\n Remember, you must not reveal your diagnosis explicitly but may only convey the symptoms you have in the form of dialogue if you are asked. You must ONLY draw from the information provided to you and must not make up any information or infer symptoms that are not provided."
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
        # FIXED: Bias-corrected base prompt
        base = f"""You are Dr. Agent, an experienced physician conducting a diagnostic evaluation. You are allowed to ask {self.MAX_INFS} questions total before making a diagnosis. You have asked {self.infs} questions so far.

CRITICAL DIAGNOSTIC PRINCIPLES:
- Common conditions are more likely than rare conditions (Hickam's Razor)
- Consider statistical likelihood based on patient demographics and presentation
- Benign conditions are more common than malignant conditions
- Base your reasoning on clinical evidence, not worst-case scenarios
- Request appropriate diagnostic tests when clinically indicated

PROCESS:
1. Gather relevant history and symptoms through focused questions
2. Request tests using format "REQUEST TEST: [test]" when needed for diagnosis
3. When ready, provide diagnosis using "DIAGNOSIS READY: [diagnosis]"

Your dialogue responses should be 1-3 sentences in length."""

        presentation = f"\n\nPatient presentation: {self.presentation}\n\nConduct a systematic evaluation focusing on the most statistically likely diagnoses first."
        
        # Add bias prompt if specified (for research purposes)
        bias_text = ""
        if self.bias and self.bias in ALL_BIASES:
            bias_text = f"\n\nIMPORTANT: {ALL_BIASES[self.bias]['prompt']}"
        
        return base + presentation + bias_text
    
    def determine_specialist(self):
        """Queries the LLM to determine the best specialist based on dialogue history."""
        # FIXED: Added statistical likelihood guidance
        prompt = f"""Based on the following patient interaction history, what type of medical specialist would be most appropriate to consult? Consider the most statistically likely diagnoses first.

History:\n{self.agent_hist}

Please respond with only the specialist type (e.g., Cardiologist, Neurologist, Pulmonologist, Gastroenterologist, Endocrinologist, Infectious Disease Specialist, Oncologist, etc.)."""
        
        system_prompt = "You are a helpful medical assistant. Analyze the dialogue and suggest the single most relevant medical specialist type based on statistical likelihood of conditions presented."
        specialist = query_model(prompt, system_prompt)
        self.specialist_type = specialist.replace("Specialist", "").strip()
        
        explanation_prompt = f"Briefly explain why a {self.specialist_type} is the most appropriate specialist based on the statistical likelihood of conditions suggested by this dialogue:\n\n{self.agent_hist}"
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
                 return "I have gathered sufficient information from the patient. Let me now consult with an appropriate specialist to refine the differential diagnosis.", "consultation_needed"

            # FIXED: Added statistical thinking to patient interaction prompt
            prompt = f"""Here is your dialogue history with the patient:
{self.agent_hist}

Patient's latest response: {last_response}

Continue your systematic evaluation. You have {self.MAX_INFS - self.infs} questions remaining. 
- Focus on the most statistically likely conditions based on the presentation
- Request tests when clinically indicated: "REQUEST TEST: [test]"
- Consider common conditions before rare ones

Doctor: """

            system_prompt = f"""You are Dr. Agent conducting a focused diagnostic evaluation. You have {self.MAX_INFS - self.infs} questions left. 

APPROACH:
- Ask targeted questions to differentiate between likely diagnoses
- Consider patient demographics and common conditions in this population
- Request appropriate diagnostic tests when needed
- Avoid premature focus on rare or serious conditions without evidence

Patient presentation: {self.presentation}"""

            answer = query_model(prompt, system_prompt)
            self.add_hist(f"Doctor: {answer}")
            self.infs += 1
            
            # REMOVED: The anti-diagnosis logic that forced overthinking
            return answer, "patient_interaction"

        elif mode == "consultation":
            # FIXED: Added evidence-based consultation guidance
            prompt = f"""Complete case history:
{self.agent_hist}

Specialist's latest input: {last_response}

Continue the consultation with the {self.specialist_type}. Focus on:
- Evidence-based differential diagnosis
- Statistical likelihood of conditions
- Appropriate next steps or diagnostic confirmation

Doctor: """

            system_prompt = f"""You are Dr. Agent, consulting with a {self.specialist_type} about a patient case. 

CONSULTATION GOALS:
- Discuss findings and develop evidence-based differential diagnoses
- Consider statistical likelihood and patient demographics  
- Reach a conclusion based on clinical evidence, not worst-case scenarios
- Focus on the most probable diagnoses first

Aim to reach a diagnosis when sufficient evidence is available."""

            answer = query_model(prompt, system_prompt)
            self.add_hist(f"Doctor: {answer}")
            self.consultation_turns += 1
            return answer, "consultation"

    def get_final_diagnosis(self):
        """Generates the final diagnosis using evidence-based decision making."""
        # FIXED: Complete rewrite with bias-corrected final diagnosis prompt
        prompt = f"""Complete patient case history:
{self.agent_hist}

FINAL DIAGNOSIS TASK:
Based on the complete consultation history above, provide your final diagnosis using evidence-based clinical reasoning.

DECISION FRAMEWORK:
1. Consider all diagnoses discussed during the consultation
2. Weight each diagnosis by:
   - Statistical likelihood in this patient demographic
   - Strength of supporting clinical evidence
   - How well it explains all the patient's symptoms
3. Choose the diagnosis with the highest evidence-weighted probability
4. Remember: Common conditions are more likely than rare conditions

CRITICAL INSTRUCTIONS:
- Do NOT automatically assume malignancy without strong evidence
- Do NOT choose diagnoses based on "ruling out" serious conditions
- DO choose based on positive evidence and statistical likelihood
- Consider Occam's Razor: the simplest explanation that fits the evidence

Provide your response in this exact format:
DIAGNOSIS READY: [Your single most likely diagnosis based on evidence and statistical probability]
CONFIDENCE: [Your confidence percentage based on strength of evidence]

Briefly explain your reasoning focusing on why this diagnosis is more statistically likely than the alternatives."""

        system_prompt = f"""You are Dr. Agent making a final diagnosis after a complete evaluation with specialist consultation.

DIAGNOSTIC PRINCIPLES:
- Base decisions on evidence and statistical likelihood
- Common conditions are more probable than rare conditions  
- Avoid malignancy bias unless strong evidence supports it
- Confidence should reflect evidence strength, not fear of missing serious conditions

Specialty consulted: {self.specialist_type}
Your goal: Provide the most statistically likely diagnosis based on all available evidence."""

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
        base = "You are a measurement reader who responds with medical test results. Please respond in the format \"RESULTS: [results here]\""
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
        # FIXED: Added evidence-based specialist guidance
        base = f"""You are a consulting specialist in {self.specialty}. You are discussing a case with the primary doctor (Dr. Agent). 

CONSULTATION APPROACH:
- Provide expert opinion based on evidence and statistical likelihood
- Consider common conditions in your specialty first
- Suggest appropriate diagnostic tests or next steps
- Focus on differential diagnoses that fit the clinical evidence
- Avoid malignancy bias unless strong evidence supports serious conditions

Review the dialogue history and provide concise expert input (1-3 sentences)."""
        return base

    def inference_specialist(self, doctor_consult_message):
        self.add_hist(f"Doctor: {doctor_consult_message}")

        # FIXED: Added evidence-based specialist prompt
        prompt = f"""Case discussion history:
{self.agent_hist}

Primary doctor's latest message: {doctor_consult_message}

As a {self.specialty} specialist, provide your expert input focusing on:
- Most statistically likely diagnoses in your specialty
- Evidence-based differential diagnosis
- Appropriate next steps or confirmatory tests
- Clinical reasoning based on probability and evidence

Specialist ({self.specialty}): """

        system_prompt = f"""You are an experienced {self.specialty} specialist providing consultation. 

EXPERT PRINCIPLES:
- Base recommendations on clinical evidence and statistical likelihood
- Consider common conditions in your specialty before rare ones
- Provide practical, evidence-based guidance
- Avoid unnecessary bias toward serious conditions without supporting evidence"""

        answer = query_model(prompt, system_prompt)
        self.add_hist(f"Specialist ({self.specialty}): {answer}")
        return answer