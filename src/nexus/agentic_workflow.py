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
    Initial risk stratification based on danger signs and patient info.

    Uses rule-based scoring:
    - Critical signs: +30 points
    - High signs: +15 points
    - Medium signs: +5 points
    - Additional risk factors from patient info
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

        reasoning.append(f"Starting triage for {patient_type} patient")

        # Score danger signs
        present_signs = [s for s in danger_signs if s.present]
        reasoning.append(f"Evaluating {len(present_signs)} present danger signs out of {len(danger_signs)} checked")

        for sign in present_signs:
            if sign.severity == "critical":
                score += 30
                critical_signs.append(sign.label)
                reasoning.append(f"CRITICAL: '{sign.label}' detected (+30 points)")
            elif sign.severity == "high":
                score += 15
                reasoning.append(f"HIGH: '{sign.label}' detected (+15 points)")
            elif sign.severity == "medium":
                score += 5
                reasoning.append(f"MEDIUM: '{sign.label}' detected (+5 points)")

        # Patient-specific risk factors
        if patient_type == "pregnant":
            if patient_info.gestational_weeks is not None:
                if patient_info.gestational_weeks < 28:
                    score += 10
                    reasoning.append(f"Preterm risk: gestational age {patient_info.gestational_weeks} weeks (<28) (+10)")
                elif patient_info.gestational_weeks > 42:
                    score += 15
                    reasoning.append(f"Post-term risk: gestational age {patient_info.gestational_weeks} weeks (>42) (+15)")
                else:
                    reasoning.append(f"Gestational age {patient_info.gestational_weeks} weeks - within normal range")
        elif patient_type == "newborn":
            if patient_info.birth_weight is not None and patient_info.birth_weight < 2500:
                score += 10
                reasoning.append(f"Low birth weight: {patient_info.birth_weight}g (<2500g) (+10)")
            if patient_info.apgar_score is not None and patient_info.apgar_score < 7:
                score += 15
                reasoning.append(f"Low APGAR score: {patient_info.apgar_score} (<7) (+15)")
            if patient_info.age_hours is not None and patient_info.age_hours < 24:
                score += 5
                reasoning.append(f"First day of life: {patient_info.age_hours} hours (+5)")

        # Determine risk level
        if score >= 30 or len(critical_signs) > 0:
            risk_level: SeverityLevel = "RED"
        elif score >= 15:
            risk_level = "YELLOW"
        else:
            risk_level = "GREEN"

        critical_detected = len(critical_signs) > 0
        immediate_referral = risk_level == "RED" and critical_detected

        reasoning.append(f"Total triage score: {score}")
        reasoning.append(f"Risk classification: {risk_level}")
        if immediate_referral:
            reasoning.append("IMMEDIATE REFERRAL REQUIRED - critical danger signs with RED classification")

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
                "immediate_referral": immediate_referral,
            },
            confidence=1.0,
            processing_time_ms=elapsed,
        )

        return result, trace


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

            confidence = 1.0 - abs(risk - 0.5) * 2  # Higher confidence when score is extreme
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
    Applies WHO IMNCI guidelines to classify severity and recommend treatment.

    Uses rules-based logic following WHO Integrated Management of
    Childhood Illness (IMNCI) protocols.
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

        reasoning.append(f"Applying WHO IMNCI protocols for {patient_type} patient")
        reasoning.append(f"Initial classification from triage: {classification}")

        # ---- Maternal protocols ----
        if patient_type == "pregnant":
            protocols.append("WHO IMNCI Maternal Care")
            reasoning.append("Applying maternal care protocols")

            if image.anemia and image.anemia.get("is_anemic"):
                protocols.append("Anemia Management Protocol")
                est_hb = image.anemia.get("estimated_hemoglobin", 0)

                if est_hb and est_hb < 7:
                    classification = "RED"
                    recommendations.append("URGENT: Severe anemia - consider blood transfusion")
                    reasoning.append(f"Severe anemia: estimated Hb={est_hb} g/dL (<7) -> RED")
                else:
                    if classification != "RED":
                        classification = "YELLOW"
                    recommendations.append("Initiate iron supplementation (60mg elemental iron + 400mcg folic acid daily)")
                    recommendations.append("Dietary counseling for iron-rich foods")
                    reasoning.append(f"Moderate anemia detected -> YELLOW, iron supplementation recommended")

            if triage.critical_signs_detected:
                protocols.append("Emergency Obstetric Care Protocol")
                recommendations.append("Immediate assessment for emergency obstetric conditions")
                reasoning.append("Critical danger signs present -> emergency obstetric protocol applied")

        # ---- Newborn protocols ----
        if patient_type == "newborn":
            protocols.append("WHO IMNCI Newborn Care")
            reasoning.append("Applying newborn care protocols")

            # Jaundice
            if image.jaundice and image.jaundice.get("has_jaundice"):
                protocols.append("Neonatal Jaundice Protocol")
                est_bili = image.jaundice.get("estimated_bilirubin", 0)

                if image.jaundice.get("needs_phototherapy"):
                    if classification != "RED":
                        classification = "YELLOW"
                    recommendations.append("Initiate phototherapy")
                    recommendations.append("Monitor bilirubin levels every 6-12 hours")
                    reasoning.append(f"Jaundice requiring phototherapy: bilirubin ~{est_bili} mg/dL -> YELLOW")
                else:
                    recommendations.append("Continue breastfeeding")
                    recommendations.append("Monitor for increasing jaundice")
                    reasoning.append("Mild jaundice - breastfeeding and monitoring recommended")

                if est_bili and est_bili > 20:
                    classification = "RED"
                    recommendations.append("URGENT: Severe hyperbilirubinemia - consider exchange transfusion")
                    reasoning.append(f"Severe hyperbilirubinemia: bilirubin={est_bili} mg/dL (>20) -> RED")

            # Cry / asphyxia
            if audio and audio.cry and audio.cry.get("is_abnormal"):
                protocols.append("Birth Asphyxia Assessment Protocol")
                asphyxia_risk = audio.cry.get("asphyxia_risk", 0)

                if asphyxia_risk > 0.7:
                    classification = "RED"
                    recommendations.append("URGENT: High asphyxia risk - immediate neonatal assessment")
                    reasoning.append(f"High asphyxia risk: {asphyxia_risk:.1%} (>70%) -> RED")
                elif asphyxia_risk > 0.4:
                    if classification != "RED":
                        classification = "YELLOW"
                    recommendations.append("Monitor neurological status")
                    recommendations.append("Consider head ultrasound")
                    reasoning.append(f"Moderate asphyxia risk: {asphyxia_risk:.1%} (>40%) -> YELLOW")

            # Neonatal anemia
            if image.anemia and image.anemia.get("is_anemic"):
                protocols.append("Neonatal Anemia Protocol")
                recommendations.append("Check hematocrit and reticulocyte count")
                if classification != "RED":
                    classification = "YELLOW"
                reasoning.append("Neonatal anemia detected -> YELLOW, blood work recommended")

        # Follow-up schedule
        if classification == "RED":
            follow_up = "Immediate referral - no follow-up at this level"
        elif classification == "YELLOW":
            follow_up = "Follow-up in 2-3 days or sooner if condition worsens"
        else:
            follow_up = (
                "Routine follow-up in 1 week"
                if patient_type == "newborn"
                else "Routine antenatal follow-up as scheduled"
            )

        reasoning.append(f"Final WHO IMNCI classification: {classification}")
        reasoning.append(f"Protocols applied: {', '.join(protocols)}")
        reasoning.append(f"Follow-up: {follow_up}")

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
            },
            confidence=1.0,
            processing_time_ms=elapsed,
        )

        return result, trace


class ReferralAgent:
    """
    Synthesizes all results to determine referral decision.

    Considers triage results, protocol classification, and specific
    condition thresholds to determine urgency and facility level.
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

        reasoning.append("Evaluating referral decision based on all agent findings")

        # Check critical danger signs
        if triage.immediate_referral_needed:
            referral_needed = True
            urgency = "immediate"
            facility_level = "tertiary"
            reasons.append(f"Critical danger signs: {', '.join(triage.critical_signs)}")
            reasoning.append(f"Critical danger signs detected -> immediate referral to tertiary facility")

        # Check protocol classification
        if protocol.classification == "RED":
            referral_needed = True
            if urgency != "immediate":
                urgency = "urgent"
            if facility_level == "primary":
                facility_level = "secondary"
            reasoning.append("RED classification -> referral needed (urgent if not already immediate)")
        elif protocol.classification == "YELLOW":
            if not referral_needed:
                urgency = "routine"
            reasoning.append("YELLOW classification -> routine referral consideration")

        # Condition-specific checks
        if patient_type == "pregnant":
            if (image.anemia and image.anemia.get("is_anemic")
                    and image.anemia.get("estimated_hemoglobin", 99) < 7):
                referral_needed = True
                if urgency != "immediate":
                    urgency = "urgent"
                facility_level = "secondary"
                reasons.append("Severe anemia requiring blood transfusion")
                reasoning.append("Severe maternal anemia (Hb<7) -> urgent referral to secondary facility")

        if patient_type == "newborn":
            if image.jaundice and image.jaundice.get("needs_phototherapy"):
                referral_needed = True
                if urgency != "immediate":
                    urgency = "urgent"
                if facility_level != "tertiary":
                    facility_level = "secondary"
                reasons.append("Jaundice requiring phototherapy")
                reasoning.append("Phototherapy needed -> urgent referral to secondary facility")

            if (audio and audio.cry
                    and audio.cry.get("asphyxia_risk", 0) > 0.7):
                referral_needed = True
                urgency = "immediate"
                facility_level = "tertiary"
                reasons.append("High birth asphyxia risk")
                reasoning.append("High asphyxia risk (>70%) -> immediate referral to tertiary facility")

        # Determine timeframe
        timeframe_map = {
            "immediate": "Within 1 hour",
            "urgent": "Within 4-6 hours",
            "routine": "Within 24-48 hours",
            "none": "Not applicable",
        }
        timeframe = timeframe_map[urgency]

        reason_text = "; ".join(reasons) if reasons else "No referral required"
        reasoning.append(f"Final decision: referral={'YES' if referral_needed else 'NO'}, urgency={urgency}")
        reasoning.append(f"Facility level: {facility_level}, Timeframe: {timeframe}")

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
