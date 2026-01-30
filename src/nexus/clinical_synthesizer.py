"""
Clinical Synthesizer Module

Uses MedGemma from Google HAI-DEF for clinical reasoning and synthesis.
Combines findings from MedSigLIP (images) and HeAR (audio) into actionable recommendations.

HAI-DEF Model: MedGemma 4B (google/medgemma-4b-it)
"""

import torch
from typing import Dict, Optional, List
from datetime import datetime

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class ClinicalSynthesizer:
    """
    Synthesizes clinical findings using MedGemma.

    HAI-DEF Model: MedGemma 4B (google/medgemma-4b-it)
    Method: Prompt engineering (no fine-tuning required)

    Output:
    - Integrated diagnosis suggestions
    - Severity assessment (GREEN/YELLOW/RED)
    - Treatment recommendations (WHO IMNCI)
    - Referral decision with urgency
    - CHW-friendly explanations
    """

    # WHO IMNCI severity colors
    SEVERITY_LEVELS = {
        "GREEN": "Routine care - no immediate concern",
        "YELLOW": "Close monitoring - may need referral",
        "RED": "Urgent referral - immediate action required",
    }

    def __init__(
        self,
        model_name: str = "google/medgemma-4b-it",
        device: Optional[str] = None,
        use_medgemma: bool = True,
    ):
        """
        Initialize the Clinical Synthesizer with MedGemma.

        Args:
            model_name: HuggingFace model name for MedGemma
            device: Device to run model on
            use_medgemma: Whether to use MedGemma (True) or rule-based (False)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.use_medgemma = use_medgemma
        self._medgemma_available = False

        if use_medgemma and HAS_TRANSFORMERS:
            self._load_medgemma()
        else:
            print("MedGemma not available. Using rule-based clinical synthesis.")
            self.use_medgemma = False

        print(f"Clinical Synthesizer (HAI-DEF MedGemma) initialized")

    def _load_medgemma(self) -> None:
        """Load MedGemma model from HuggingFace."""
        import os
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("Warning: HF_TOKEN not set. MedGemma is a gated model and requires authentication.")
            print("Set HF_TOKEN environment variable with your HuggingFace token.")

        try:
            print(f"Loading MedGemma model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, token=hf_token
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=hf_token,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            self._medgemma_available = True
            print("MedGemma loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load MedGemma: {e}")
            print("Falling back to rule-based synthesis")
            self.model = None
            self.tokenizer = None
            self.use_medgemma = False
            self._medgemma_available = False

    def _build_prompt(self, findings: Dict) -> str:
        """
        Build clinical synthesis prompt for MedGemma.

        Args:
            findings: Dictionary with anemia, jaundice, cry analysis results

        Returns:
            Formatted prompt for MedGemma
        """
        # Extract findings with safe defaults
        anemia = findings.get("anemia", {})
        jaundice = findings.get("jaundice", {})
        cry = findings.get("cry", {})
        symptoms = findings.get("symptoms", "None reported")
        patient_info = findings.get("patient_info", {})

        prompt = f"""You are a pediatric health assistant helping community health workers in low-resource settings.

PATIENT INFORMATION:
- Age: {patient_info.get("age", "Not specified")}
- Weight: {patient_info.get("weight", "Not specified")}
- Location: {patient_info.get("location", "Rural health post")}

ASSESSMENT FINDINGS:

1. ANEMIA SCREENING (Conjunctiva Analysis):
   - Result: {"Anemia detected" if anemia.get("is_anemic") else "No anemia detected"}
   - Confidence: {anemia.get("confidence", "N/A")}
   - Severity: {anemia.get("severity", anemia.get("risk_level", "N/A"))}
   - Estimated Hemoglobin: {anemia.get("estimated_hemoglobin", "N/A")} g/dL

2. JAUNDICE SCREENING (Skin Analysis):
   - Result: {"Jaundice detected" if jaundice.get("has_jaundice") else "No jaundice detected"}
   - Confidence: {jaundice.get("confidence", "N/A")}
   - Severity: {jaundice.get("severity", "N/A")}
   - Estimated Bilirubin: {jaundice.get("estimated_bilirubin", "N/A")} mg/dL
   - Needs Phototherapy: {jaundice.get("needs_phototherapy", "N/A")}

3. CRY ANALYSIS (Audio):
   - Result: {"Abnormal cry pattern" if cry.get("is_abnormal") else "Normal cry pattern"}
   - Asphyxia Risk: {cry.get("asphyxia_risk", "N/A")}
   - Cry Type: {cry.get("cry_type", "N/A")}

4. REPORTED SYMPTOMS:
   {symptoms}

Based on these findings, provide a clinical assessment following WHO IMNCI protocols:

1. ASSESSMENT SUMMARY (2-3 sentences in simple language)
2. SEVERITY LEVEL (GREEN = routine care, YELLOW = close monitoring, RED = urgent referral)
3. IMMEDIATE ACTIONS for the CHW (bullet points, simple steps)
4. REFERRAL RECOMMENDATION (Yes/No, and if yes, urgency level)
5. FOLLOW-UP PLAN (when to reassess)

Use simple language appropriate for a community health worker with basic training.
Focus on actionable steps they can take immediately.
"""
        return prompt

    def synthesize(self, findings: Dict) -> Dict:
        """
        Synthesize all findings into clinical recommendations.

        Args:
            findings: Dictionary with anemia, jaundice, cry analysis results

        Returns:
            Clinical summary and recommendations
        """
        if self.use_medgemma and self.model is not None:
            return self._synthesize_with_medgemma(findings)
        else:
            return self._synthesize_rule_based(findings)

    def _synthesize_with_medgemma(self, findings: Dict) -> Dict:
        """Synthesize using MedGemma model."""
        prompt = self._build_prompt(findings)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated part (after the prompt)
        response = response[len(prompt):].strip()

        return {
            "summary": response,
            "model": "MedGemma 4B",
            "generated_at": datetime.now().isoformat(),
            "findings_used": list(findings.keys()),
        }

    def _synthesize_rule_based(self, findings: Dict) -> Dict:
        """
        Rule-based clinical synthesis (fallback when MedGemma unavailable).

        Follows WHO IMNCI protocols for maternal and neonatal care.
        """
        # Extract findings
        anemia = findings.get("anemia", {})
        jaundice = findings.get("jaundice", {})
        cry = findings.get("cry", {})

        # Determine overall severity
        severity_score = 0
        urgent_conditions = []
        actions = []
        referral_needed = False
        referral_urgency = "none"

        # Assess anemia
        if anemia.get("is_anemic"):
            if anemia.get("risk_level") == "high":
                severity_score += 3
                urgent_conditions.append("Severe anemia")
                actions.append("Refer for blood transfusion if Hb < 7 g/dL")
                referral_needed = True
                referral_urgency = "urgent"
            elif anemia.get("risk_level") == "medium":
                severity_score += 2
                urgent_conditions.append("Moderate anemia")
                actions.append("Start iron supplementation")
                actions.append("Schedule blood test within 48 hours")
            else:
                severity_score += 1
                actions.append("Monitor hemoglobin levels")
                actions.append("Encourage iron-rich foods")

        # Assess jaundice
        if jaundice.get("has_jaundice"):
            if jaundice.get("needs_phototherapy"):
                severity_score += 3
                urgent_conditions.append("Severe jaundice requiring phototherapy")
                actions.append("URGENT: Start phototherapy immediately")
                actions.append("Refer to hospital if phototherapy unavailable")
                referral_needed = True
                referral_urgency = "immediate"
            elif jaundice.get("severity") in ["moderate", "severe"]:
                severity_score += 2
                urgent_conditions.append("Moderate jaundice")
                actions.append("Expose baby to indirect sunlight")
                actions.append("Ensure frequent breastfeeding")
                actions.append("Recheck in 12-24 hours")
            else:
                severity_score += 1
                actions.append("Continue breastfeeding")
                actions.append("Monitor skin color")

        # Assess cry analysis
        if cry.get("is_abnormal"):
            if cry.get("asphyxia_risk", 0) > 0.6:
                severity_score += 3
                urgent_conditions.append("Signs of birth asphyxia")
                actions.append("URGENT: Check airway, breathing, circulation")
                actions.append("Provide warmth and stimulation")
                actions.append("Immediate referral for evaluation")
                referral_needed = True
                referral_urgency = "immediate"
            else:
                severity_score += 1
                actions.append("Monitor cry patterns")
                actions.append("Assess feeding and alertness")

        # Determine overall severity level
        if severity_score >= 5 or referral_urgency == "immediate":
            severity_level = "RED"
            summary = f"URGENT ATTENTION NEEDED. {', '.join(urgent_conditions)}. Immediate medical intervention required."
        elif severity_score >= 2:
            severity_level = "YELLOW"
            summary = f"Close monitoring required. {', '.join(urgent_conditions) if urgent_conditions else 'Some abnormal findings detected'}. Follow recommended actions."
        else:
            severity_level = "GREEN"
            summary = "Routine care. No immediate concerns detected. Continue standard monitoring."

        # Default actions if none specified
        if not actions:
            actions = [
                "Continue routine care",
                "Ensure adequate nutrition",
                "Schedule follow-up in 1 week",
            ]

        # Follow-up plan
        if severity_level == "RED":
            follow_up = "Immediate referral. Follow up after hospital evaluation."
        elif severity_level == "YELLOW":
            follow_up = "Reassess in 24-48 hours. Refer if condition worsens."
        else:
            follow_up = "Routine follow-up in 1-2 weeks."

        return {
            "summary": summary,
            "severity_level": severity_level,
            "severity_description": self.SEVERITY_LEVELS[severity_level],
            "immediate_actions": actions,
            "referral_needed": referral_needed,
            "referral_urgency": referral_urgency,
            "follow_up": follow_up,
            "urgent_conditions": urgent_conditions,
            "model": "Rule-based (WHO IMNCI)",
            "generated_at": datetime.now().isoformat(),
        }

    def get_who_protocol(self, condition: str) -> Dict:
        """
        Get WHO IMNCI protocol for a specific condition.

        Args:
            condition: Condition name (anemia, jaundice, asphyxia)

        Returns:
            Protocol details
        """
        protocols = {
            "anemia": {
                "name": "Maternal Anemia Management",
                "source": "WHO IMNCI Guidelines",
                "steps": [
                    "Assess pallor of conjunctiva, palms, and nail beds",
                    "If severe pallor: Urgent referral",
                    "If some pallor: Iron supplementation + folic acid",
                    "Counsel on iron-rich foods",
                    "Follow up in 4 weeks",
                ],
                "referral_criteria": "Hb < 7 g/dL or severe pallor with symptoms",
            },
            "jaundice": {
                "name": "Neonatal Jaundice Management",
                "source": "WHO IMNCI Guidelines",
                "steps": [
                    "Check for yellow skin/eyes within first 24 hours",
                    "If jaundice in first 24 hours: URGENT referral",
                    "If moderate jaundice: Frequent breastfeeding, sun exposure",
                    "If bilirubin > 15 mg/dL: Phototherapy",
                    "If bilirubin > 25 mg/dL: Exchange transfusion",
                ],
                "referral_criteria": "Jaundice < 24 hours old, bilirubin > 20 mg/dL",
            },
            "asphyxia": {
                "name": "Birth Asphyxia Management",
                "source": "WHO Neonatal Resuscitation Guidelines",
                "steps": [
                    "Assess APGAR score at 1 and 5 minutes",
                    "Clear airway if needed",
                    "Provide warmth and stimulation",
                    "If not breathing: Begin resuscitation",
                    "Refer for evaluation if abnormal cry or poor feeding",
                ],
                "referral_criteria": "APGAR < 7, abnormal cry, seizures, poor feeding",
            },
        }
        return protocols.get(condition.lower(), {"error": "Protocol not found"})


def test_synthesizer():
    """Test the clinical synthesizer."""
    print("Testing Clinical Synthesizer...")

    synthesizer = ClinicalSynthesizer(use_medgemma=False)  # Use rule-based for testing

    # Test case: Multiple findings
    findings = {
        "anemia": {
            "is_anemic": True,
            "confidence": 0.85,
            "risk_level": "medium",
            "estimated_hemoglobin": 9.5,
        },
        "jaundice": {
            "has_jaundice": True,
            "confidence": 0.75,
            "severity": "mild",
            "estimated_bilirubin": 8.5,
            "needs_phototherapy": False,
        },
        "cry": {
            "is_abnormal": False,
            "asphyxia_risk": 0.2,
            "cry_type": "hunger",
        },
        "symptoms": "Mother reports baby seems tired after feeding",
    }

    result = synthesizer.synthesize(findings)

    print("\n=== Clinical Synthesis Result ===")
    print(f"Summary: {result['summary']}")
    print(f"Severity: {result.get('severity_level', 'N/A')}")
    print(f"Referral Needed: {result.get('referral_needed', 'N/A')}")
    print(f"Actions: {result.get('immediate_actions', [])}")
    print(f"Follow-up: {result.get('follow_up', 'N/A')}")


if __name__ == "__main__":
    test_synthesizer()
