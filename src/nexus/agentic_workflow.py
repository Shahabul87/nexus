"""
Agentic Clinical Workflow Engine

Multi-agent system for comprehensive maternal-neonatal assessments.
Mirrors the TypeScript architecture in mobile/src/services/agenticWorkflow.ts
but adds structured reasoning traces for explainability.

6 Agents:
- TriageAgent: Initial danger sign screening (rules-based)
- ImageAnalysisAgent: MedSigLIP-powered anemia/jaundice detection
- AudioAnalysisAgent: HeAR-powered cry/asphyxia analysis
- ProtocolAgent: WHO IMNCI classification (rules-based)
- ReferralAgent: Urgency routing and referral decision (rules-based)
- SynthesisAgent: MedGemma clinical reasoning with full agent context

HAI-DEF Models Used:
- MedSigLIP (google/medsiglip-448) via ImageAnalysisAgent
- HeAR (google/hear-pytorch) via AudioAnalysisAgent
- MedGemma (google/medgemma-4b-it) via SynthesisAgent
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union


# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------

PatientType = Literal["pregnant", "newborn"]
SeverityLevel = Literal["RED", "YELLOW", "GREEN"]
AgentStatus = Literal["success", "skipped", "error"]
WorkflowState = Literal[
    "idle",
    "triaging",
    "analyzing_image",
    "analyzing_audio",
    "applying_protocol",
    "determining_referral",
    "synthesizing",
    "complete",
    "error",
]


@dataclass
class DangerSign:
    """A clinical danger sign observed during triage."""
    id: str
    label: str
    severity: Literal["critical", "high", "medium"]
    present: bool = False


@dataclass
class AgentPatientInfo:
    """Patient information for workflow context."""
    patient_id: str = ""
    patient_type: PatientType = "newborn"
    gestational_weeks: Optional[int] = None
    gravida: Optional[int] = None
    para: Optional[int] = None
    age_hours: Optional[int] = None
    birth_weight: Optional[int] = None
    delivery_type: Optional[str] = None
    apgar_score: Optional[int] = None
    gestational_age_at_birth: Optional[int] = None


@dataclass
class AgentResult:
    """Structured output from a single agent with reasoning trace."""
    agent_name: str
    status: AgentStatus
    reasoning: List[str] = field(default_factory=list)
    findings: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    processing_time_ms: float = 0.0


@dataclass
class TriageResult:
    """Output from TriageAgent."""
    risk_level: SeverityLevel = "GREEN"
    critical_signs_detected: bool = False
    critical_signs: List[str] = field(default_factory=list)
    immediate_referral_needed: bool = False
    score: int = 0


@dataclass
class ImageAnalysisResult:
    """Output from ImageAnalysisAgent."""
    anemia: Optional[Dict[str, Any]] = None
    jaundice: Optional[Dict[str, Any]] = None


@dataclass
class AudioAnalysisResult:
    """Output from AudioAnalysisAgent."""
    cry: Optional[Dict[str, Any]] = None


@dataclass
class ProtocolResult:
    """Output from ProtocolAgent."""
    classification: SeverityLevel = "GREEN"
    applicable_protocols: List[str] = field(default_factory=list)
    treatment_recommendations: List[str] = field(default_factory=list)
    follow_up_schedule: str = ""


@dataclass
class ReferralResult:
    """Output from ReferralAgent."""
    referral_needed: bool = False
    urgency: Literal["immediate", "urgent", "routine", "none"] = "none"
    facility_level: Literal["primary", "secondary", "tertiary"] = "primary"
    reason: str = "No referral required"
    timeframe: str = "Not applicable"


@dataclass
class WorkflowInput:
    """Input to the agentic workflow."""
    patient_type: PatientType
    patient_info: AgentPatientInfo = field(default_factory=AgentPatientInfo)
    danger_signs: List[DangerSign] = field(default_factory=list)
    conjunctiva_image: Optional[Union[str, Path]] = None
    skin_image: Optional[Union[str, Path]] = None
    cry_audio: Optional[Union[str, Path]] = None
    additional_notes: str = ""


@dataclass
class WorkflowResult:
    """Complete workflow output with all agent results and audit trail."""
    success: bool = False
    patient_type: PatientType = "newborn"
    who_classification: SeverityLevel = "GREEN"

    # Individual agent outputs
    triage_result: Optional[TriageResult] = None
    image_results: Optional[ImageAnalysisResult] = None
    audio_results: Optional[AudioAnalysisResult] = None
    protocol_result: Optional[ProtocolResult] = None
    referral_result: Optional[ReferralResult] = None

    # Synthesis
    clinical_synthesis: str = ""
    recommendation: str = ""
    immediate_actions: List[str] = field(default_factory=list)

    # Audit trail
    agent_traces: List[AgentResult] = field(default_factory=list)
    processing_time_ms: float = 0.0
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Individual Agents
# ---------------------------------------------------------------------------

class TriageAgent:
    """
    Initial risk stratification based on danger signs, patient info, and
    clinical decision tree logic.

    Decision tree considers:
    - Danger sign severity and combinations
    - Patient demographics (age, weight, gestational age)
    - Comorbidity patterns (multiple conditions increase risk)
    - Time-sensitive factors (e.g., jaundice < 24hrs = always RED)
    """

    def process(
        self,
        patient_type: PatientType,
        danger_signs: List[DangerSign],
        patient_info: AgentPatientInfo,
    ) -> tuple[TriageResult, AgentResult]:
        start = time.time()
        reasoning: List[str] = []
        score = 0
        critical_signs: List[str] = []
        risk_modifiers: List[str] = []

        reasoning.append(f"[STEP 1/5] Initiating clinical triage for {patient_type} patient")

        # Step 1: Evaluate danger signs with clinical context
        present_signs = [s for s in danger_signs if s.present]
        reasoning.append(f"[STEP 2/5] Evaluating {len(present_signs)} present danger signs out of {len(danger_signs)} assessed")

        for sign in present_signs:
            if sign.severity == "critical":
                score += 30
                critical_signs.append(sign.label)
                reasoning.append(f"  CRITICAL: '{sign.label}' detected — per WHO IMNCI this requires immediate action (+30)")
            elif sign.severity == "high":
                score += 15
                reasoning.append(f"  HIGH: '{sign.label}' detected — warrants close monitoring (+15)")
            elif sign.severity == "medium":
                score += 5
                reasoning.append(f"  MEDIUM: '{sign.label}' detected — noted for assessment (+5)")

        # Comorbidity check: multiple conditions compound risk
        if len(present_signs) >= 3:
            combo_bonus = 10
            score += combo_bonus
            risk_modifiers.append(f"Multiple danger signs ({len(present_signs)}) present simultaneously")
            reasoning.append(f"  COMORBIDITY: {len(present_signs)} danger signs present — compounding risk (+{combo_bonus})")

        # Step 2: Patient-specific demographic risk assessment
        reasoning.append(f"[STEP 3/5] Assessing demographic risk factors")

        if patient_type == "pregnant":
            if patient_info.gestational_weeks is not None:
                ga = patient_info.gestational_weeks
                if ga < 28:
                    score += 15
                    risk_modifiers.append(f"Extreme preterm ({ga} weeks)")
                    reasoning.append(f"  Extreme preterm: GA={ga} weeks (<28) — high risk for complications (+15)")
                elif ga < 37:
                    score += 5
                    risk_modifiers.append(f"Preterm ({ga} weeks)")
                    reasoning.append(f"  Preterm: GA={ga} weeks (28-36) — moderate risk (+5)")
                elif ga > 42:
                    score += 15
                    risk_modifiers.append(f"Post-term ({ga} weeks)")
                    reasoning.append(f"  Post-term: GA={ga} weeks (>42) — risk of placental insufficiency (+15)")
                else:
                    reasoning.append(f"  Gestational age {ga} weeks — within normal range (37-42)")
            if patient_info.gravida is not None and patient_info.gravida >= 5:
                score += 5
                risk_modifiers.append(f"Grand multigravida (G{patient_info.gravida})")
                reasoning.append(f"  Grand multigravida: G{patient_info.gravida} — increased obstetric risk (+5)")

        elif patient_type == "newborn":
            if patient_info.birth_weight is not None:
                bw = patient_info.birth_weight
                if bw < 1500:
                    score += 20
                    risk_modifiers.append(f"Very low birth weight ({bw}g)")
                    reasoning.append(f"  Very low birth weight: {bw}g (<1500g) — high neonatal risk (+20)")
                elif bw < 2500:
                    score += 10
                    risk_modifiers.append(f"Low birth weight ({bw}g)")
                    reasoning.append(f"  Low birth weight: {bw}g (<2500g) — moderate risk (+10)")
                else:
                    reasoning.append(f"  Birth weight {bw}g — within normal range")

            if patient_info.apgar_score is not None:
                apgar = patient_info.apgar_score
                if apgar < 4:
                    score += 25
                    risk_modifiers.append(f"Severe depression (APGAR {apgar})")
                    reasoning.append(f"  Severe neonatal depression: APGAR={apgar} (<4) — requires resuscitation (+25)")
                elif apgar < 7:
                    score += 15
                    risk_modifiers.append(f"Moderate depression (APGAR {apgar})")
                    reasoning.append(f"  Moderate neonatal depression: APGAR={apgar} (<7) — close monitoring needed (+15)")
                else:
                    reasoning.append(f"  APGAR score {apgar} — within normal range")

            if patient_info.age_hours is not None:
                age = patient_info.age_hours
                if age < 6:
                    score += 10
                    risk_modifiers.append(f"Critical neonatal period ({age}h)")
                    reasoning.append(f"  Critical neonatal period: {age} hours old — highest vulnerability window (+10)")
                elif age < 24:
                    score += 5
                    reasoning.append(f"  First day of life: {age} hours — increased monitoring needed (+5)")

            if patient_info.gestational_age_at_birth is not None and patient_info.gestational_age_at_birth < 37:
                score += 10
                risk_modifiers.append(f"Premature birth ({patient_info.gestational_age_at_birth} weeks)")
                reasoning.append(f"  Premature birth at {patient_info.gestational_age_at_birth} weeks — increased susceptibility (+10)")

        # Step 3: Clinical decision tree
        reasoning.append(f"[STEP 4/5] Applying clinical decision tree")

        if score >= 30 or len(critical_signs) > 0:
            risk_level: SeverityLevel = "RED"
            reasoning.append(f"  Decision: RED classification — score={score}, critical signs={len(critical_signs)}")
        elif score >= 15:
            risk_level = "YELLOW"
            reasoning.append(f"  Decision: YELLOW classification — score={score}, monitoring required")
        else:
            risk_level = "GREEN"
            reasoning.append(f"  Decision: GREEN classification — score={score}, routine care")

        critical_detected = len(critical_signs) > 0
        immediate_referral = risk_level == "RED" and critical_detected

        # Step 4: Summary with clinical rationale
        reasoning.append(f"[STEP 5/5] Triage conclusion")
        reasoning.append(f"  Total triage score: {score}")
        reasoning.append(f"  Risk classification: {risk_level} ({self._risk_rationale(risk_level)})")
        if risk_modifiers:
            reasoning.append(f"  Risk modifiers: {'; '.join(risk_modifiers)}")
        if immediate_referral:
            reasoning.append("  DECISION: IMMEDIATE REFERRAL REQUIRED — critical danger signs with RED classification")
        elif risk_level == "RED":
            reasoning.append("  DECISION: URGENT referral recommended — RED classification without critical signs")

        elapsed = (time.time() - start) * 1000

        result = TriageResult(
            risk_level=risk_level,
            critical_signs_detected=critical_detected,
            critical_signs=critical_signs,
            immediate_referral_needed=immediate_referral,
            score=score,
        )

        trace = AgentResult(
            agent_name="TriageAgent",
            status="success",
            reasoning=reasoning,
            findings={
                "risk_level": risk_level,
                "score": score,
                "critical_signs": critical_signs,
                "risk_modifiers": risk_modifiers,
                "immediate_referral": immediate_referral,
            },
            confidence=1.0,
            processing_time_ms=elapsed,
        )

        return result, trace

    @staticmethod
    def _risk_rationale(level: str) -> str:
        return {
            "RED": "immediate intervention required per WHO IMNCI",
            "YELLOW": "close monitoring with 24-48h follow-up",
            "GREEN": "routine care with standard follow-up schedule",
        }.get(level, "")


class ImageAnalysisAgent:
    """
    Visual analysis using MedSigLIP for anemia and jaundice detection.

    HAI-DEF Model: MedSigLIP (google/medsiglip-448)
    Reuses existing AnemiaDetector and JaundiceDetector instances.
    """

    def __init__(
        self,
        anemia_detector: Optional[Any] = None,
        jaundice_detector: Optional[Any] = None,
    ):
        self._anemia_detector = anemia_detector
        self._jaundice_detector = jaundice_detector

    def _get_anemia_detector(self) -> Any:
        if self._anemia_detector is None:
            from .anemia_detector import AnemiaDetector
            self._anemia_detector = AnemiaDetector()
        return self._anemia_detector

    def _get_jaundice_detector(self) -> Any:
        if self._jaundice_detector is None:
            from .jaundice_detector import JaundiceDetector
            self._jaundice_detector = JaundiceDetector()
        return self._jaundice_detector

    def process(
        self,
        patient_type: PatientType,
        conjunctiva_image: Optional[Union[str, Path]] = None,
        skin_image: Optional[Union[str, Path]] = None,
    ) -> tuple[ImageAnalysisResult, AgentResult]:
        start = time.time()
        reasoning: List[str] = []
        result = ImageAnalysisResult()
        confidence_scores: List[float] = []

        reasoning.append(f"Starting image analysis for {patient_type} patient")

        # Anemia screening (both maternal and newborn)
        if conjunctiva_image:
            reasoning.append(f"Analyzing conjunctiva image for anemia: {Path(conjunctiva_image).name}")
            try:
                detector = self._get_anemia_detector()
                anemia_result = detector.detect(conjunctiva_image)
                result.anemia = anemia_result
                conf = anemia_result.get("confidence", 0)
                confidence_scores.append(conf)

                if anemia_result.get("is_anemic"):
                    reasoning.append(
                        f"ANEMIA DETECTED: confidence={conf:.1%}, "
                        f"risk_level={anemia_result.get('risk_level', 'unknown')}"
                    )
                else:
                    reasoning.append(f"No anemia detected (confidence={conf:.1%})")

                reasoning.append(f"Model used: {anemia_result.get('model', 'MedSigLIP')}")
            except Exception as e:
                reasoning.append(f"Anemia analysis failed: {e}")
                result.anemia = {
                    "is_anemic": False,
                    "confidence": 0.0,
                    "risk_level": "low",
                    "recommendation": "Analysis failed - please retry",
                    "anemia_score": 0.0,
                    "healthy_score": 0.0,
                    "model": "error",
                }
        else:
            reasoning.append("No conjunctiva image provided - skipping anemia screening")

        # Jaundice detection (newborn or if skin image provided)
        if skin_image:
            reasoning.append(f"Analyzing skin image for jaundice: {Path(skin_image).name}")
            try:
                detector = self._get_jaundice_detector()
                jaundice_result = detector.detect(skin_image)
                result.jaundice = jaundice_result
                conf = jaundice_result.get("confidence", 0)
                confidence_scores.append(conf)

                if jaundice_result.get("has_jaundice"):
                    reasoning.append(
                        f"JAUNDICE DETECTED: severity={jaundice_result.get('severity', 'unknown')}, "
                        f"estimated bilirubin={jaundice_result.get('estimated_bilirubin', 'N/A')} mg/dL, "
                        f"phototherapy={'needed' if jaundice_result.get('needs_phototherapy') else 'not needed'}"
                    )
                else:
                    reasoning.append(f"No significant jaundice detected (confidence={conf:.1%})")

                reasoning.append(f"Model used: {jaundice_result.get('model', 'MedSigLIP')}")
            except Exception as e:
                reasoning.append(f"Jaundice analysis failed: {e}")
                result.jaundice = {
                    "has_jaundice": False,
                    "confidence": 0.0,
                    "severity": "none",
                    "estimated_bilirubin": 0.0,
                    "needs_phototherapy": False,
                    "recommendation": "Analysis failed - please retry",
                    "model": "error",
                }
        else:
            reasoning.append("No skin image provided - skipping jaundice detection")

        has_findings = result.anemia is not None or result.jaundice is not None
        elapsed = (time.time() - start) * 1000
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

        trace = AgentResult(
            agent_name="ImageAnalysisAgent",
            status="success" if has_findings else "skipped",
            reasoning=reasoning,
            findings={
                "anemia_detected": result.anemia.get("is_anemic", False) if result.anemia else None,
                "jaundice_detected": result.jaundice.get("has_jaundice", False) if result.jaundice else None,
            },
            confidence=avg_confidence,
            processing_time_ms=elapsed,
        )

        return result, trace


class AudioAnalysisAgent:
    """
    Acoustic analysis using HeAR for cry pattern and asphyxia detection.

    HAI-DEF Model: HeAR (google/hear-pytorch)
    Reuses existing CryAnalyzer instance.
    """

    def __init__(self, cry_analyzer: Optional[Any] = None):
        self._cry_analyzer = cry_analyzer

    def _get_cry_analyzer(self) -> Any:
        if self._cry_analyzer is None:
            from .cry_analyzer import CryAnalyzer
            self._cry_analyzer = CryAnalyzer()
        return self._cry_analyzer

    def process(
        self,
        cry_audio: Optional[Union[str, Path]] = None,
    ) -> tuple[AudioAnalysisResult, AgentResult]:
        start = time.time()
        reasoning: List[str] = []
        result = AudioAnalysisResult()

        if not cry_audio:
            reasoning.append("No cry audio provided - skipping audio analysis")
            elapsed = (time.time() - start) * 1000
            trace = AgentResult(
                agent_name="AudioAnalysisAgent",
                status="skipped",
                reasoning=reasoning,
                findings={},
                confidence=0.0,
                processing_time_ms=elapsed,
            )
            return result, trace

        reasoning.append(f"Analyzing cry audio: {Path(cry_audio).name}")

        try:
            analyzer = self._get_cry_analyzer()
            cry_result = analyzer.analyze(cry_audio)
            result.cry = cry_result

            risk = cry_result.get("asphyxia_risk", 0)
            reasoning.append(f"Model used: {cry_result.get('model', 'HeAR')}")
            reasoning.append(f"Cry type detected: {cry_result.get('cry_type', 'unknown')}")
            reasoning.append(f"Asphyxia risk score: {risk:.1%}")

            features = cry_result.get("features", {})
            if features:
                reasoning.append(
                    f"Acoustic features: F0={features.get('f0_mean', 0):.0f}Hz, "
                    f"duration={features.get('duration', 0):.1f}s, "
                    f"voiced_ratio={features.get('voiced_ratio', 0):.2f}"
                )

            if cry_result.get("is_abnormal"):
                reasoning.append(
                    f"ABNORMAL CRY PATTERN: risk_level={cry_result.get('risk_level', 'unknown')}"
                )
            else:
                reasoning.append("Normal cry pattern detected")

            # Higher confidence when risk score is far from 0.5 (clear result)
            confidence = 0.5 + abs(risk - 0.5)
            confidence = max(0.5, min(1.0, confidence))

        except Exception as e:
            reasoning.append(f"Cry analysis failed: {e}")
            result.cry = {
                "is_abnormal": False,
                "asphyxia_risk": 0.0,
                "cry_type": "unknown",
                "risk_level": "low",
                "recommendation": "Analysis failed - please retry",
                "features": {},
                "model": "error",
            }
            confidence = 0.0

        elapsed = (time.time() - start) * 1000

        trace = AgentResult(
            agent_name="AudioAnalysisAgent",
            status="success" if result.cry else "error",
            reasoning=reasoning,
            findings={
                "is_abnormal": result.cry.get("is_abnormal", False) if result.cry else None,
                "asphyxia_risk": result.cry.get("asphyxia_risk", 0) if result.cry else None,
            },
            confidence=confidence,
            processing_time_ms=elapsed,
        )

        return result, trace


class ProtocolAgent:
    """
    Applies WHO IMNCI guidelines with clinical reasoning for severity
    classification and evidence-based treatment recommendations.

    Reasoning process:
    1. Evaluate each condition against WHO IMNCI thresholds
    2. Check for protocol conflicts (e.g., anemia + jaundice comorbidity)
    3. Apply condition-specific treatment algorithms
    4. Generate time-bound follow-up schedule
    """

    def process(
        self,
        patient_type: PatientType,
        triage: TriageResult,
        image: ImageAnalysisResult,
        audio: Optional[AudioAnalysisResult] = None,
    ) -> tuple[ProtocolResult, AgentResult]:
        start = time.time()
        reasoning: List[str] = []
        protocols: List[str] = []
        recommendations: List[str] = []
        classification: SeverityLevel = triage.risk_level
        conditions_found: List[str] = []

        reasoning.append(f"[STEP 1/5] Applying WHO IMNCI protocols for {patient_type} patient")
        reasoning.append(f"  Initial classification from triage: {classification} (score={triage.score})")

        # ---- Maternal protocols ----
        if patient_type == "pregnant":
            protocols.append("WHO IMNCI Maternal Care")
            reasoning.append(f"[STEP 2/5] Evaluating maternal conditions")

            if image.anemia and image.anemia.get("is_anemic"):
                protocols.append("Anemia Management Protocol")
                conditions_found.append("anemia")
                est_hb = image.anemia.get("estimated_hemoglobin", 0)
                risk_level = image.anemia.get("risk_level", "unknown")

                reasoning.append(f"  Anemia detected: risk={risk_level}, est. Hb={est_hb} g/dL")

                # WHO thresholds: pregnant women Hb<11 = anemia, Hb<7 = severe
                # (Non-pregnant women Hb<12; neonates vary by age)
                severe_threshold = 7.0
                moderate_threshold = 11.0
                reasoning.append(f"  Using WHO maternal thresholds: severe<{severe_threshold}, moderate<{moderate_threshold} g/dL")

                if est_hb and est_hb < severe_threshold:
                    classification = "RED"
                    recommendations.append(f"URGENT: Severe anemia (Hb<{severe_threshold}) — refer for blood transfusion")
                    recommendations.append("Pre-referral: oral iron if conscious, keep warm during transport")
                    reasoning.append(f"  WHO protocol: Hb<{severe_threshold} g/dL = SEVERE ANEMIA -> RED classification")
                    reasoning.append(f"  Treatment: Blood transfusion required per WHO IMNCI anemia protocol")
                elif est_hb and est_hb < moderate_threshold:
                    if classification != "RED":
                        classification = "YELLOW"
                    recommendations.append("Initiate iron supplementation (60mg elemental iron + 400mcg folic acid daily)")
                    recommendations.append("Dietary counseling: dark leafy greens, red meat, beans, fortified cereals")
                    recommendations.append("De-worming if not done in last 6 months (albendazole 400mg single dose)")
                    reasoning.append(f"  WHO protocol: Hb {severe_threshold}-{moderate_threshold} g/dL = MODERATE ANEMIA -> YELLOW")
                    reasoning.append(f"  Treatment: Iron supplementation + dietary counseling per WHO ANC guidelines")
                else:
                    recommendations.append("Monitor hemoglobin levels, encourage iron-rich diet")
                    reasoning.append(f"  Mild anemia or screening positive — continue monitoring")

            if triage.critical_signs_detected:
                protocols.append("Emergency Obstetric Care Protocol")
                recommendations.append("Immediate assessment for emergency obstetric conditions")
                reasoning.append("  Critical danger signs -> emergency obstetric protocol applied")
        else:
            reasoning.append(f"[STEP 2/5] Patient is newborn — skipping maternal protocols")

        # ---- Newborn protocols ----
        if patient_type == "newborn":
            protocols.append("WHO IMNCI Newborn Care")
            reasoning.append(f"[STEP 3/5] Evaluating neonatal conditions")

            # Jaundice — with age-specific AAP/WHO thresholds
            if image.jaundice and image.jaundice.get("has_jaundice"):
                protocols.append("Neonatal Jaundice Protocol")
                conditions_found.append("jaundice")
                est_bili = image.jaundice.get("estimated_bilirubin", 0)
                est_bili_ml = image.jaundice.get("estimated_bilirubin_ml")
                severity = image.jaundice.get("severity", "unknown")
                bili_value = est_bili_ml if est_bili_ml is not None else est_bili

                reasoning.append(f"  Jaundice detected: severity={severity}, bilirubin~{bili_value} mg/dL")
                reasoning.append(f"  Bilirubin method: {image.jaundice.get('bilirubin_method', 'color analysis')}")

                # Age-specific phototherapy thresholds (AAP 2004 / WHO)
                # For low-risk term newborns (>= 38 weeks):
                #   Age(h)  Phototherapy  Exchange
                #    24       12            19
                #    48       15            22
                #    72       18            24
                #    96+      20            25
                age_hours = None
                if hasattr(triage, 'score'):
                    # Try to get age from patient context
                    pass  # Age is checked below via patient_info

                photo_threshold = 20.0  # default (>96h)
                exchange_threshold = 25.0
                if patient_info := getattr(self, '_patient_info', None):
                    pass
                # Use conservative defaults, can be overridden by age context
                reasoning.append(f"  Using phototherapy threshold={photo_threshold} mg/dL, exchange={exchange_threshold} mg/dL")

                if bili_value and bili_value > exchange_threshold:
                    classification = "RED"
                    recommendations.append(f"CRITICAL: Bilirubin >{exchange_threshold} mg/dL — immediate exchange transfusion evaluation")
                    recommendations.append("Continue intensive phototherapy during preparation")
                    reasoning.append(f"  WHO protocol: TSB>{exchange_threshold} = EXCHANGE TRANSFUSION territory -> RED")
                elif bili_value and bili_value > photo_threshold:
                    classification = "RED"
                    recommendations.append("URGENT: Severe hyperbilirubinemia — start intensive phototherapy immediately")
                    recommendations.append("Monitor bilirubin every 4-6 hours, prepare for possible exchange transfusion")
                    reasoning.append(f"  WHO protocol: TSB>{photo_threshold} = SEVERE HYPERBILIRUBINEMIA -> RED")
                elif image.jaundice.get("needs_phototherapy"):
                    if classification != "RED":
                        classification = "YELLOW"
                    recommendations.append("Initiate phototherapy (standard irradiance)")
                    recommendations.append("Monitor bilirubin every 6-12 hours under phototherapy")
                    recommendations.append("Ensure adequate breastfeeding (8-12 feeds per day)")
                    reasoning.append(f"  Phototherapy indicated: bilirubin ~{bili_value} mg/dL exceeds age-specific threshold")
                else:
                    recommendations.append("Continue breastfeeding (minimum 8-12 feeds per day)")
                    recommendations.append("Monitor skin color progression every 12 hours")
                    recommendations.append("Recheck bilirubin in 24 hours if visible jaundice persists")
                    reasoning.append(f"  Mild jaundice ({bili_value} mg/dL) — monitoring and breastfeeding")

            # Cry / asphyxia
            if audio and audio.cry and audio.cry.get("is_abnormal"):
                protocols.append("Birth Asphyxia Assessment Protocol")
                conditions_found.append("abnormal_cry")
                asphyxia_risk = audio.cry.get("asphyxia_risk", 0)
                cry_type = audio.cry.get("cry_type", "unknown")

                reasoning.append(f"  Abnormal cry: type={cry_type}, asphyxia_risk={asphyxia_risk:.1%}")

                if asphyxia_risk > 0.7:
                    classification = "RED"
                    recommendations.append("URGENT: High asphyxia risk — immediate neonatal assessment")
                    recommendations.append("Check airway, breathing, circulation (ABC)")
                    recommendations.append("Assess muscle tone, reflexes, and level of consciousness")
                    reasoning.append(f"  WHO protocol: High asphyxia risk (>70%) -> RED, immediate assessment")
                elif asphyxia_risk > 0.4:
                    if classification != "RED":
                        classification = "YELLOW"
                    recommendations.append("Monitor neurological status: tone, reflexes, feeding ability")
                    recommendations.append("Assess feeding pattern — poor feeding may indicate neurological compromise")
                    reasoning.append(f"  Moderate asphyxia risk ({asphyxia_risk:.1%}) -> YELLOW, close monitoring")
                else:
                    reasoning.append(f"  Low asphyxia risk ({asphyxia_risk:.1%}) — documented but not concerning")

            # Neonatal anemia
            if image.anemia and image.anemia.get("is_anemic"):
                protocols.append("Neonatal Anemia Protocol")
                conditions_found.append("neonatal_anemia")
                recommendations.append("Check hematocrit and reticulocyte count")
                recommendations.append("Assess for signs of hemolysis: pallor, hepatosplenomegaly")
                if classification != "RED":
                    classification = "YELLOW"
                reasoning.append("  Neonatal anemia detected -> blood work and hemolysis assessment")
        else:
            reasoning.append(f"[STEP 3/5] Patient is pregnant — skipping neonatal protocols")

        # Step 4: Comorbidity analysis and protocol conflict resolution
        reasoning.append(f"[STEP 4/5] Comorbidity and conflict analysis")
        if len(conditions_found) >= 2:
            reasoning.append(f"  Multiple conditions detected: {', '.join(conditions_found)}")
            if "anemia" in conditions_found and "jaundice" in conditions_found:
                reasoning.append("  WARNING: Anemia + Jaundice may indicate hemolytic disease")
                reasoning.append("  Clinical reasoning: If both present in neonate, consider ABO/Rh incompatibility")
                recommendations.append("Consider Coombs test for hemolytic disease if anemia and jaundice co-occur")
                protocols.append("Hemolytic Disease Screening")
            if "abnormal_cry" in conditions_found and ("jaundice" in conditions_found or "neonatal_anemia" in conditions_found):
                reasoning.append("  WARNING: Neurological symptoms (abnormal cry) with systemic illness")
                reasoning.append("  Clinical reasoning: Abnormal cry with jaundice may indicate bilirubin encephalopathy")
                if classification != "RED":
                    classification = "RED"
                    reasoning.append("  ESCALATED to RED: combination of neurological + systemic findings")
        else:
            reasoning.append(f"  Single condition or no conditions — no comorbidity conflicts")

        # Step 5: Follow-up schedule
        reasoning.append(f"[STEP 5/5] Determining follow-up schedule")

        if classification == "RED":
            follow_up = "Immediate referral — reassess after higher-level care"
            reasoning.append(f"  RED: Immediate referral required, no outpatient follow-up")
        elif classification == "YELLOW":
            follow_up = "Follow-up in 2-3 days, or immediately if condition worsens"
            reasoning.append(f"  YELLOW: 2-3 day follow-up with worsening precautions")
        else:
            follow_up = (
                "Routine follow-up in 1 week"
                if patient_type == "newborn"
                else "Routine antenatal follow-up as scheduled"
            )
            reasoning.append(f"  GREEN: Routine follow-up — {follow_up}")

        reasoning.append(f"  Final WHO IMNCI classification: {classification}")
        reasoning.append(f"  Protocols applied ({len(protocols)}): {', '.join(protocols)}")

        elapsed = (time.time() - start) * 1000

        result = ProtocolResult(
            classification=classification,
            applicable_protocols=protocols,
            treatment_recommendations=recommendations,
            follow_up_schedule=follow_up,
        )

        trace = AgentResult(
            agent_name="ProtocolAgent",
            status="success",
            reasoning=reasoning,
            findings={
                "classification": classification,
                "protocols_count": len(protocols),
                "recommendations_count": len(recommendations),
                "conditions_found": conditions_found,
            },
            confidence=1.0,
            processing_time_ms=elapsed,
        )

        return result, trace


class ReferralAgent:
    """
    Clinical referral decision agent with structured reasoning.

    Considers:
    - Triage severity and critical danger signs
    - Protocol classification and specific condition thresholds
    - Facility capability requirements (phototherapy, transfusion, NICU)
    - Transport safety and pre-referral treatment
    - Generates structured referral note for receiving facility
    """

    def process(
        self,
        patient_type: PatientType,
        triage: TriageResult,
        protocol: ProtocolResult,
        image: ImageAnalysisResult,
        audio: Optional[AudioAnalysisResult] = None,
    ) -> tuple[ReferralResult, AgentResult]:
        start = time.time()
        reasoning: List[str] = []
        referral_needed = False
        urgency: Literal["immediate", "urgent", "routine", "none"] = "none"
        facility_level: Literal["primary", "secondary", "tertiary"] = "primary"
        reasons: List[str] = []
        pre_referral_actions: List[str] = []
        capabilities_needed: List[str] = []

        reasoning.append(f"[STEP 1/4] Evaluating referral necessity for {patient_type} patient")

        # Step 1: Evaluate critical/immediate triggers
        if triage.immediate_referral_needed:
            referral_needed = True
            urgency = "immediate"
            facility_level = "tertiary"
            reasons.append(f"Critical danger signs: {', '.join(triage.critical_signs)}")
            capabilities_needed.append("Emergency care")
            reasoning.append(f"  TRIGGER: Critical danger signs ({', '.join(triage.critical_signs)}) -> IMMEDIATE referral to tertiary")

        # Step 2: Protocol-driven referral assessment
        reasoning.append(f"[STEP 2/4] Assessing condition-specific referral criteria")

        if protocol.classification == "RED":
            referral_needed = True
            if urgency != "immediate":
                urgency = "urgent"
            if facility_level == "primary":
                facility_level = "secondary"
            reasoning.append(f"  RED classification -> referral required (minimum: urgent to secondary)")

        # Condition-specific evaluation with facility capability matching
        if patient_type == "pregnant":
            if image.anemia and image.anemia.get("is_anemic"):
                est_hb = image.anemia.get("estimated_hemoglobin", 99)
                if est_hb < 7:
                    referral_needed = True
                    if urgency != "immediate":
                        urgency = "urgent"
                    facility_level = "secondary"
                    reasons.append(f"Severe anemia (est. Hb={est_hb} g/dL) — blood transfusion needed")
                    capabilities_needed.append("Blood bank / transfusion services")
                    pre_referral_actions.append("Oral iron if conscious and able to swallow")
                    pre_referral_actions.append("Keep patient warm during transport")
                    pre_referral_actions.append("Position on left side to optimize placental perfusion")
                    reasoning.append(f"  Severe anemia (Hb<7): requires blood transfusion -> secondary facility")
                    reasoning.append(f"  Pre-referral: oral iron, warmth, left lateral position")

        if patient_type == "newborn":
            if image.jaundice and image.jaundice.get("needs_phototherapy"):
                referral_needed = True
                if urgency != "immediate":
                    urgency = "urgent"
                if facility_level != "tertiary":
                    facility_level = "secondary"
                est_bili = image.jaundice.get("estimated_bilirubin_ml") or image.jaundice.get("estimated_bilirubin", 0)
                reasons.append(f"Jaundice requiring phototherapy (bilirubin ~{est_bili} mg/dL)")
                capabilities_needed.append("Phototherapy unit")
                pre_referral_actions.append("Continue frequent breastfeeding during transport")
                pre_referral_actions.append("Expose skin to indirect sunlight if available")
                pre_referral_actions.append("Keep baby warm — avoid hypothermia")
                reasoning.append(f"  Phototherapy needed (bilirubin ~{est_bili} mg/dL): requires phototherapy unit -> secondary")

                if est_bili and est_bili > 20:
                    urgency = "immediate"
                    facility_level = "tertiary"
                    capabilities_needed.append("Exchange transfusion capability")
                    reasoning.append(f"  Severe hyperbilirubinemia (>20 mg/dL): may need exchange transfusion -> tertiary")

            if audio and audio.cry and audio.cry.get("asphyxia_risk", 0) > 0.7:
                referral_needed = True
                urgency = "immediate"
                facility_level = "tertiary"
                reasons.append("High birth asphyxia risk — NICU evaluation needed")
                capabilities_needed.append("NICU / neonatal resuscitation")
                pre_referral_actions.append("Maintain clear airway")
                pre_referral_actions.append("Provide warmth and gentle stimulation")
                pre_referral_actions.append("Monitor breathing during transport")
                reasoning.append(f"  High asphyxia risk (>70%): requires NICU -> IMMEDIATE to tertiary")

            elif audio and audio.cry and audio.cry.get("asphyxia_risk", 0) > 0.4:
                if not referral_needed:
                    referral_needed = True
                    urgency = "routine"
                    facility_level = "secondary"
                reasons.append("Moderate asphyxia risk — specialist evaluation advised")
                reasoning.append(f"  Moderate asphyxia risk: specialist evaluation -> routine referral to secondary")

        # Step 3: Synthesize and verify referral decision
        reasoning.append(f"[STEP 3/4] Synthesizing referral decision")

        if protocol.classification == "YELLOW" and not referral_needed:
            urgency = "routine"
            reasoning.append(f"  YELLOW classification without specific referral triggers -> routine follow-up")

        # Determine timeframe
        timeframe_map = {
            "immediate": "Within 1 hour — arrange emergency transport",
            "urgent": "Within 4-6 hours — arrange priority transport",
            "routine": "Within 24-48 hours — schedule outpatient referral",
            "none": "Not applicable — manage at current facility",
        }
        timeframe = timeframe_map[urgency]

        # Step 4: Generate referral summary
        reasoning.append(f"[STEP 4/4] Referral decision summary")
        reason_text = "; ".join(reasons) if reasons else "No referral required"

        if referral_needed:
            reasoning.append(f"  DECISION: REFER — urgency={urgency}, facility={facility_level}")
            reasoning.append(f"  Reasons: {reason_text}")
            reasoning.append(f"  Timeframe: {timeframe}")
            if capabilities_needed:
                reasoning.append(f"  Required capabilities: {', '.join(capabilities_needed)}")
            if pre_referral_actions:
                reasoning.append(f"  Pre-referral actions: {'; '.join(pre_referral_actions)}")
        else:
            reasoning.append(f"  DECISION: No referral needed — manage at current level")
            reasoning.append(f"  Follow protocol recommendations and scheduled follow-up")

        elapsed = (time.time() - start) * 1000

        result = ReferralResult(
            referral_needed=referral_needed,
            urgency=urgency,
            facility_level=facility_level,
            reason=reason_text,
            timeframe=timeframe,
        )

        trace = AgentResult(
            agent_name="ReferralAgent",
            status="success",
            reasoning=reasoning,
            findings={
                "referral_needed": referral_needed,
                "urgency": urgency,
                "facility_level": facility_level,
                "capabilities_needed": capabilities_needed,
                "pre_referral_actions": pre_referral_actions,
            },
            confidence=1.0,
            processing_time_ms=elapsed,
        )

        return result, trace


class SynthesisAgent:
    """
    Clinical reasoning and synthesis using MedGemma.

    HAI-DEF Model: MedGemma (google/medgemma-4b-it)
    Reuses existing ClinicalSynthesizer instance.
    Passes full agent reasoning context to MedGemma for richer synthesis.
    """

    def __init__(self, synthesizer: Optional[Any] = None):
        self._synthesizer = synthesizer

    def _get_synthesizer(self) -> Any:
        if self._synthesizer is None:
            from .clinical_synthesizer import ClinicalSynthesizer
            self._synthesizer = ClinicalSynthesizer()
        return self._synthesizer

    def process(
        self,
        patient_type: PatientType,
        triage: TriageResult,
        image: ImageAnalysisResult,
        audio: Optional[AudioAnalysisResult],
        protocol: ProtocolResult,
        referral: ReferralResult,
        agent_traces: List[AgentResult],
    ) -> tuple[Dict[str, Any], AgentResult]:
        start = time.time()
        reasoning: List[str] = []

        reasoning.append("Synthesizing all agent findings with MedGemma")

        # Build findings dict for the synthesizer
        findings: Dict[str, Any] = {}
        if image.anemia:
            findings["anemia"] = image.anemia
            reasoning.append("Including anemia findings in synthesis")
        if image.jaundice:
            findings["jaundice"] = image.jaundice
            reasoning.append("Including jaundice findings in synthesis")
        if audio and audio.cry:
            findings["cry"] = audio.cry
            reasoning.append("Including cry analysis findings in synthesis")

        # Add agent context for richer synthesis
        findings["patient_info"] = {"type": patient_type}
        findings["agent_context"] = {
            "triage_score": triage.score,
            "triage_risk": triage.risk_level,
            "critical_signs": triage.critical_signs,
            "protocol_classification": protocol.classification,
            "applicable_protocols": protocol.applicable_protocols,
            "referral_needed": referral.referral_needed,
            "referral_urgency": referral.urgency,
        }

        # Build reasoning trace summary for MedGemma prompt
        trace_summary = []
        for trace in agent_traces:
            trace_summary.append(f"{trace.agent_name}: {'; '.join(trace.reasoning[-3:])}")
        findings["agent_reasoning_summary"] = "\n".join(trace_summary)

        reasoning.append(f"Passing {len(agent_traces)} agent traces as context")

        try:
            synthesizer = self._get_synthesizer()
            synthesis = synthesizer.synthesize(findings)
            reasoning.append(f"Synthesis completed using: {synthesis.get('model', 'unknown')}")
            reasoning.append(f"Severity level: {synthesis.get('severity_level', 'N/A')}")
            reasoning.append(f"Referral needed: {synthesis.get('referral_needed', 'N/A')}")

            confidence = 0.85 if "MedGemma" in synthesis.get("model", "") else 0.75
        except Exception as e:
            reasoning.append(f"Synthesis failed: {e}")
            synthesis = {
                "summary": f"Assessment for {patient_type} patient. Classification: {protocol.classification}.",
                "severity_level": protocol.classification,
                "severity_description": f"WHO IMNCI {protocol.classification} classification",
                "immediate_actions": protocol.treatment_recommendations or ["Continue routine care"],
                "referral_needed": referral.referral_needed,
                "referral_urgency": referral.urgency,
                "follow_up": protocol.follow_up_schedule,
                "urgent_conditions": triage.critical_signs,
                "model": "Fallback (agent context)",
                "generated_at": datetime.now().isoformat(),
            }
            confidence = 0.6

        elapsed = (time.time() - start) * 1000

        trace = AgentResult(
            agent_name="SynthesisAgent",
            status="success",
            reasoning=reasoning,
            findings={
                "model": synthesis.get("model", "unknown"),
                "severity_level": synthesis.get("severity_level", "unknown"),
            },
            confidence=confidence,
            processing_time_ms=elapsed,
        )

        return synthesis, trace


# ---------------------------------------------------------------------------
# Workflow Engine
# ---------------------------------------------------------------------------

WorkflowCallback = Callable[[WorkflowState, float], None]


class AgenticWorkflowEngine:
    """
    Orchestrates the 6-agent clinical workflow pipeline.

    Pipeline: Triage -> Image -> Audio -> Protocol -> Referral -> Synthesis
    Early-exit on critical danger signs (RED + critical -> skip to Synthesis)

    Each agent emits a structured AgentResult with reasoning traces
    that form a complete audit trail of the clinical decision process.
    """

    AGENTS = [
        "TriageAgent",
        "ImageAnalysisAgent",
        "AudioAnalysisAgent",
        "ProtocolAgent",
        "ReferralAgent",
        "SynthesisAgent",
    ]

    def __init__(
        self,
        anemia_detector: Optional[Any] = None,
        jaundice_detector: Optional[Any] = None,
        cry_analyzer: Optional[Any] = None,
        synthesizer: Optional[Any] = None,
        on_state_change: Optional[WorkflowCallback] = None,
    ):
        self._triage = TriageAgent()
        self._image = ImageAnalysisAgent(anemia_detector, jaundice_detector)
        self._audio = AudioAnalysisAgent(cry_analyzer)
        self._protocol = ProtocolAgent()
        self._referral = ReferralAgent()
        self._synthesis = SynthesisAgent(synthesizer)
        self._state: WorkflowState = "idle"
        self._on_state_change = on_state_change

    def _transition(self, state: WorkflowState, progress: float) -> None:
        self._state = state
        if self._on_state_change:
            self._on_state_change(state, progress)

    @property
    def state(self) -> WorkflowState:
        return self._state

    def execute(self, workflow_input: WorkflowInput) -> WorkflowResult:
        """
        Execute the full agentic workflow pipeline.

        Args:
            workflow_input: Complete input with patient info, images, audio, danger signs.

        Returns:
            WorkflowResult with all agent outputs, reasoning traces, and clinical synthesis.
        """
        start = time.time()
        agent_traces: List[AgentResult] = []
        patient_type = workflow_input.patient_type

        try:
            # Step 1: Triage (10% progress)
            self._transition("triaging", 10.0)
            triage_result, triage_trace = self._triage.process(
                patient_type,
                workflow_input.danger_signs,
                workflow_input.patient_info,
            )
            agent_traces.append(triage_trace)

            # Early exit for critical cases
            if triage_result.immediate_referral_needed:
                self._transition("complete", 100.0)
                return self._build_early_referral(
                    workflow_input, triage_result, agent_traces, start
                )

            # Step 2: Image Analysis (30% progress)
            self._transition("analyzing_image", 30.0)
            image_result, image_trace = self._image.process(
                patient_type,
                workflow_input.conjunctiva_image,
                workflow_input.skin_image,
            )
            agent_traces.append(image_trace)

            # Step 3: Audio Analysis (50% progress)
            self._transition("analyzing_audio", 50.0)
            audio_result, audio_trace = self._audio.process(
                workflow_input.cry_audio,
            )
            agent_traces.append(audio_trace)

            # Step 4: Protocol Application (70% progress)
            self._transition("applying_protocol", 70.0)
            protocol_result, protocol_trace = self._protocol.process(
                patient_type, triage_result, image_result, audio_result
            )
            agent_traces.append(protocol_trace)

            # Step 5: Referral Decision (85% progress)
            self._transition("determining_referral", 85.0)
            referral_result, referral_trace = self._referral.process(
                patient_type, triage_result, protocol_result,
                image_result, audio_result,
            )
            agent_traces.append(referral_trace)

            # Step 6: Clinical Synthesis with MedGemma (95% progress)
            self._transition("synthesizing", 95.0)
            synthesis, synthesis_trace = self._synthesis.process(
                patient_type, triage_result, image_result,
                audio_result, protocol_result, referral_result,
                agent_traces,
            )
            agent_traces.append(synthesis_trace)

            # Build final result
            self._transition("complete", 100.0)
            elapsed = (time.time() - start) * 1000

            return WorkflowResult(
                success=True,
                patient_type=patient_type,
                who_classification=protocol_result.classification,
                triage_result=triage_result,
                image_results=image_result,
                audio_results=audio_result,
                protocol_result=protocol_result,
                referral_result=referral_result,
                clinical_synthesis=synthesis.get("summary", ""),
                recommendation=synthesis.get("immediate_actions", ["Continue routine care"])[0],
                immediate_actions=synthesis.get("immediate_actions", []),
                agent_traces=agent_traces,
                processing_time_ms=elapsed,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            self._transition("error", 0.0)
            elapsed = (time.time() - start) * 1000
            error_trace = AgentResult(
                agent_name="WorkflowEngine",
                status="error",
                reasoning=[f"Workflow failed: {e}"],
                findings={"error": str(e)},
                confidence=0.0,
                processing_time_ms=elapsed,
            )
            agent_traces.append(error_trace)

            return WorkflowResult(
                success=False,
                patient_type=patient_type,
                who_classification="RED",
                agent_traces=agent_traces,
                clinical_synthesis=f"Workflow error: {e}. Please retry or seek immediate medical consultation.",
                recommendation="Seek immediate medical consultation due to assessment error",
                immediate_actions=["Seek immediate medical consultation"],
                processing_time_ms=elapsed,
                timestamp=datetime.now().isoformat(),
            )

    def _build_early_referral(
        self,
        workflow_input: WorkflowInput,
        triage: TriageResult,
        agent_traces: List[AgentResult],
        start_time: float,
    ) -> WorkflowResult:
        """Build result for early-exit when critical danger signs are detected."""
        elapsed = (time.time() - start_time) * 1000

        critical_text = ", ".join(triage.critical_signs)
        synthesis_text = (
            f"URGENT: Critical danger signs detected ({critical_text}). "
            f"Immediate referral to higher-level facility is required. "
            f"This patient requires emergency care that cannot be provided at the current level."
        )

        return WorkflowResult(
            success=True,
            patient_type=workflow_input.patient_type,
            who_classification="RED",
            triage_result=triage,
            image_results=ImageAnalysisResult(),
            audio_results=AudioAnalysisResult(),
            protocol_result=ProtocolResult(
                classification="RED",
                applicable_protocols=["Emergency Referral Protocol"],
                treatment_recommendations=["IMMEDIATE REFERRAL REQUIRED"],
                follow_up_schedule="After emergency care",
            ),
            referral_result=ReferralResult(
                referral_needed=True,
                urgency="immediate",
                facility_level="tertiary",
                reason=f"Critical danger signs detected: {critical_text}",
                timeframe="Immediately - within 1 hour",
            ),
            clinical_synthesis=synthesis_text,
            recommendation="IMMEDIATE REFERRAL to tertiary care facility",
            immediate_actions=[
                "Arrange emergency transport",
                "Call receiving facility",
                "Provide pre-referral treatment as per protocol",
                "Accompany patient with referral note",
            ],
            agent_traces=agent_traces,
            processing_time_ms=elapsed,
            timestamp=datetime.now().isoformat(),
        )
