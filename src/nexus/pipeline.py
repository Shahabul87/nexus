"""
NEXUS Pipeline Module

Integrates all detection modules into a unified diagnostic pipeline
for maternal-neonatal care.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class PatientInfo:
    """Patient information for context."""
    patient_id: str
    age_days: Optional[int] = None  # For neonates
    gestational_age: Optional[int] = None  # Weeks
    birth_weight: Optional[int] = None  # Grams
    gender: Optional[str] = None
    is_maternal: bool = False  # True for mother, False for neonate


@dataclass
class AssessmentResult:
    """Complete assessment result."""
    patient: PatientInfo
    timestamp: str
    anemia_result: Optional[Dict] = None
    jaundice_result: Optional[Dict] = None
    cry_result: Optional[Dict] = None
    overall_risk: str = "unknown"
    priority_actions: List[str] = None
    referral_needed: bool = False


class NEXUSPipeline:
    """
    NEXUS Integrated Diagnostic Pipeline

    Combines anemia, jaundice, and cry analysis into a unified
    assessment workflow for maternal-neonatal care.
    """

    # Default paths for trained model checkpoints
    DEFAULT_CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "models" / "checkpoints"
    DEFAULT_LINEAR_PROBE_DIR = Path(__file__).parent.parent.parent / "models" / "linear_probes"

    def __init__(
        self,
        device: Optional[str] = None,
        lazy_load: bool = True,
        anemia_checkpoint: Optional[Union[str, Path]] = None,
        jaundice_checkpoint: Optional[Union[str, Path]] = None,
        cry_checkpoint: Optional[Union[str, Path]] = None,
        use_linear_probes: bool = True,
    ):
        """
        Initialize NEXUS Pipeline.

        Args:
            device: Device for model inference
            lazy_load: If True, load models only when needed
            anemia_checkpoint: Path to trained anemia classifier checkpoint
            jaundice_checkpoint: Path to trained jaundice classifier checkpoint
            cry_checkpoint: Path to trained cry classifier checkpoint
            use_linear_probes: If True, auto-load linear probes from default dir
        """
        self.device = device
        self.lazy_load = lazy_load

        # Store checkpoint paths
        self.anemia_checkpoint = anemia_checkpoint
        self.jaundice_checkpoint = jaundice_checkpoint
        self.cry_checkpoint = cry_checkpoint

        # Auto-detect checkpoints from default locations
        if use_linear_probes:
            self._auto_detect_checkpoints()

        self._anemia_detector = None
        self._jaundice_detector = None
        self._cry_analyzer = None

        if not lazy_load:
            self._load_all_models()

        print("NEXUS Pipeline initialized")

    def verify_hai_def_compliance(self) -> Dict:
        """
        Verify which HAI-DEF models are loaded and report compliance.

        Returns:
            Dictionary with model status and compliance flag.
        """
        from .anemia_detector import MEDSIGLIP_MODEL_IDS
        from .cry_analyzer import CryAnalyzer

        status = {
            "medsiglip": {
                "expected": "google/medsiglip-448",
                "configured_models": MEDSIGLIP_MODEL_IDS,
                "anemia_loaded": self._anemia_detector is not None,
                "jaundice_loaded": self._jaundice_detector is not None,
            },
            "hear": {
                "expected": CryAnalyzer.HEAR_MODEL_ID,
                "cry_loaded": self._cry_analyzer is not None,
                "hear_active": getattr(self._cry_analyzer, '_hear_available', False) if self._cry_analyzer else False,
            },
            "medgemma": {
                "expected": "google/medgemma-4b-it",
            },
        }

        # Check loaded model names
        if self._anemia_detector:
            status["medsiglip"]["anemia_model"] = getattr(self._anemia_detector, 'model_name', 'unknown')
        if self._jaundice_detector:
            status["medsiglip"]["jaundice_model"] = getattr(self._jaundice_detector, 'model_name', 'unknown')

        # Overall compliance
        anemia_ok = "medsiglip" in status["medsiglip"].get("anemia_model", "")
        jaundice_ok = "medsiglip" in status["medsiglip"].get("jaundice_model", "")
        hear_ok = status["hear"]["hear_active"]

        status["compliant"] = anemia_ok or jaundice_ok or hear_ok
        status["all_hai_def"] = anemia_ok and jaundice_ok and hear_ok

        return status

    def _auto_detect_checkpoints(self) -> None:
        """Auto-detect trained checkpoints from default directories."""
        # Check for linear probes (.joblib sklearn models)
        if self.anemia_checkpoint is None:
            anemia_probe = self.DEFAULT_LINEAR_PROBE_DIR / "anemia_linear_probe.joblib"
            if anemia_probe.exists():
                self.anemia_checkpoint = anemia_probe
                print(f"Auto-detected anemia probe: {anemia_probe}")

        if self.jaundice_checkpoint is None:
            jaundice_probe = self.DEFAULT_LINEAR_PROBE_DIR / "jaundice_linear_probe.joblib"
            if jaundice_probe.exists():
                self.jaundice_checkpoint = jaundice_probe
                print(f"Auto-detected jaundice probe: {jaundice_probe}")

        if self.cry_checkpoint is None:
            cry_probe = self.DEFAULT_LINEAR_PROBE_DIR / "cry_linear_probe.joblib"
            if cry_probe.exists():
                self.cry_checkpoint = cry_probe
                print(f"Auto-detected cry probe: {cry_probe}")

        # Also check checkpoint dir for full fine-tuned models
        if self.anemia_checkpoint is None:
            anemia_best = self.DEFAULT_CHECKPOINT_DIR / "anemia_best.pt"
            if anemia_best.exists():
                self.anemia_checkpoint = anemia_best
                print(f"Auto-detected anemia checkpoint: {anemia_best}")

    def _load_all_models(self) -> None:
        """Load all detection models."""
        self._get_anemia_detector()
        self._get_jaundice_detector()
        self._get_cry_analyzer()

    def _get_anemia_detector(self):
        """Get or create anemia detector with optional trained classifier."""
        if self._anemia_detector is None:
            from .anemia_detector import AnemiaDetector

            # Initialize detector
            self._anemia_detector = AnemiaDetector(device=self.device)

            # Load trained classifier if available
            if self.anemia_checkpoint:
                self._load_classifier_checkpoint(
                    self._anemia_detector,
                    self.anemia_checkpoint,
                    "anemia"
                )

        return self._anemia_detector

    def _get_jaundice_detector(self):
        """Get or create jaundice detector with optional trained classifier."""
        if self._jaundice_detector is None:
            from .jaundice_detector import JaundiceDetector

            self._jaundice_detector = JaundiceDetector(device=self.device)

            # Load trained classifier if available
            if self.jaundice_checkpoint:
                self._load_classifier_checkpoint(
                    self._jaundice_detector,
                    self.jaundice_checkpoint,
                    "jaundice"
                )

        return self._jaundice_detector

    def _get_cry_analyzer(self):
        """Get or create cry analyzer with optional trained classifier."""
        if self._cry_analyzer is None:
            from .cry_analyzer import CryAnalyzer

            # Cry analyzer supports classifier_path directly
            classifier_path = str(self.cry_checkpoint) if self.cry_checkpoint else None
            self._cry_analyzer = CryAnalyzer(
                device=self.device,
                classifier_path=classifier_path
            )

        return self._cry_analyzer

    def _load_classifier_checkpoint(
        self,
        detector,
        checkpoint_path: Union[str, Path],
        model_type: str
    ) -> None:
        """
        Load a trained classifier checkpoint into a detector.

        Supports both linear probes (sklearn) and PyTorch checkpoints.
        """
        import torch

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"Warning: {model_type} checkpoint not found: {checkpoint_path}")
            return

        try:
            # Check if it's a sklearn model (joblib)
            if checkpoint_path.suffix in ['.pkl', '.joblib']:
                import joblib
                classifier = joblib.load(checkpoint_path)
                detector.classifier = classifier
                print(f"Loaded sklearn classifier for {model_type}")

            # Check if it's a PyTorch model
            elif checkpoint_path.suffix == '.pt':
                checkpoint = torch.load(checkpoint_path, map_location=self.device or 'cpu')

                # Handle different checkpoint formats
                if 'classifier' in checkpoint:
                    # Linear probe format
                    detector.classifier = checkpoint['classifier']
                    print(f"Loaded linear probe for {model_type}")
                elif 'model_state_dict' in checkpoint:
                    # Full model checkpoint - would need separate handling
                    print(f"Note: Full model checkpoint for {model_type} - using zero-shot")
                else:
                    print(f"Unknown checkpoint format for {model_type}")

        except Exception as e:
            print(f"Warning: Could not load {model_type} checkpoint: {e}")

    def assess_maternal(
        self,
        patient: PatientInfo,
        conjunctiva_image: Optional[Union[str, Path]] = None,
    ) -> AssessmentResult:
        """
        Perform maternal health assessment.

        Currently focuses on anemia detection via conjunctiva imaging.

        Args:
            patient: Patient information
            conjunctiva_image: Path to conjunctiva image

        Returns:
            AssessmentResult with findings
        """
        result = AssessmentResult(
            patient=patient,
            timestamp=datetime.now().isoformat(),
            priority_actions=[],
        )

        # Anemia detection
        if conjunctiva_image:
            detector = self._get_anemia_detector()
            result.anemia_result = detector.detect(conjunctiva_image)

            # Add color analysis
            color_info = detector.analyze_color_features(conjunctiva_image)
            result.anemia_result["color_analysis"] = color_info

            # Determine actions
            if result.anemia_result["risk_level"] == "high":
                result.priority_actions.append("URGENT: Refer for blood test - suspected severe anemia")
                result.referral_needed = True
                result.overall_risk = "high"
            elif result.anemia_result["risk_level"] == "medium":
                result.priority_actions.append("Schedule blood test within 48 hours")
                result.overall_risk = "medium"
            else:
                result.overall_risk = "low"

        return result

    def assess_neonate(
        self,
        patient: PatientInfo,
        skin_image: Optional[Union[str, Path]] = None,
        cry_audio: Optional[Union[str, Path]] = None,
    ) -> AssessmentResult:
        """
        Perform neonatal health assessment.

        Includes jaundice detection and cry analysis.

        Args:
            patient: Patient information
            skin_image: Path to skin/sclera image for jaundice
            cry_audio: Path to cry audio file

        Returns:
            AssessmentResult with findings
        """
        result = AssessmentResult(
            patient=patient,
            timestamp=datetime.now().isoformat(),
            priority_actions=[],
        )

        risk_scores = []

        # Jaundice detection
        if skin_image:
            detector = self._get_jaundice_detector()
            result.jaundice_result = detector.detect(skin_image)

            # Add zone analysis
            zone_info = detector.analyze_kramer_zones(skin_image)
            result.jaundice_result["zone_analysis"] = zone_info

            if result.jaundice_result["severity"] == "critical":
                result.priority_actions.insert(0, "CRITICAL: Immediate phototherapy required")
                result.referral_needed = True
                risk_scores.append(1.0)
            elif result.jaundice_result["severity"] == "severe":
                result.priority_actions.append("URGENT: Start phototherapy")
                result.referral_needed = True
                risk_scores.append(0.8)
            elif result.jaundice_result["severity"] == "moderate":
                result.priority_actions.append("Monitor closely, recheck in 12-24 hours")
                risk_scores.append(0.5)
            else:
                risk_scores.append(0.2)

        # Cry analysis
        if cry_audio:
            analyzer = self._get_cry_analyzer()
            result.cry_result = analyzer.analyze(cry_audio)

            if result.cry_result["risk_level"] == "high":
                result.priority_actions.insert(0, "URGENT: Abnormal cry - assess for birth asphyxia")
                result.referral_needed = True
                risk_scores.append(1.0)
            elif result.cry_result["risk_level"] == "medium":
                result.priority_actions.append("Monitor cry patterns, reassess in 30 minutes")
                risk_scores.append(0.5)
            else:
                risk_scores.append(0.2)

        # Determine overall risk
        if risk_scores:
            max_risk = max(risk_scores)
            if max_risk >= 0.8:
                result.overall_risk = "high"
            elif max_risk >= 0.5:
                result.overall_risk = "medium"
            else:
                result.overall_risk = "low"

        return result

    def full_assessment(
        self,
        patient: PatientInfo,
        conjunctiva_image: Optional[Union[str, Path]] = None,
        skin_image: Optional[Union[str, Path]] = None,
        cry_audio: Optional[Union[str, Path]] = None,
    ) -> AssessmentResult:
        """
        Perform full assessment (maternal or neonatal based on patient info).

        Args:
            patient: Patient information
            conjunctiva_image: For maternal anemia screening
            skin_image: For neonatal jaundice detection
            cry_audio: For neonatal cry analysis

        Returns:
            Complete AssessmentResult
        """
        if patient.is_maternal:
            return self.assess_maternal(patient, conjunctiva_image)
        else:
            return self.assess_neonate(patient, skin_image, cry_audio)

    def generate_report(self, result: AssessmentResult) -> str:
        """
        Generate a text report from assessment result.

        Args:
            result: AssessmentResult from assessment

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "NEXUS HEALTH ASSESSMENT REPORT",
            "=" * 60,
            "",
            f"Patient ID: {result.patient.patient_id}",
            f"Assessment Time: {result.timestamp}",
            f"Patient Type: {'Maternal' if result.patient.is_maternal else 'Neonatal'}",
            "",
        ]

        if result.patient.age_days is not None:
            lines.append(f"Age: {result.patient.age_days} days")
        if result.patient.gestational_age is not None:
            lines.append(f"Gestational Age: {result.patient.gestational_age} weeks")
        if result.patient.birth_weight is not None:
            lines.append(f"Birth Weight: {result.patient.birth_weight} grams")

        lines.extend(["", "-" * 60, "FINDINGS", "-" * 60, ""])

        # Anemia findings
        if result.anemia_result:
            lines.extend([
                "ANEMIA SCREENING:",
                f"  Status: {'ANEMIC' if result.anemia_result['is_anemic'] else 'Normal'}",
                f"  Confidence: {result.anemia_result['confidence']:.1%}",
                f"  Risk Level: {result.anemia_result['risk_level'].upper()}",
                "",
            ])

        # Jaundice findings
        if result.jaundice_result:
            lines.extend([
                "JAUNDICE ASSESSMENT:",
                f"  Status: {'JAUNDICE DETECTED' if result.jaundice_result['has_jaundice'] else 'Normal'}",
                f"  Estimated Bilirubin: {result.jaundice_result['estimated_bilirubin']} mg/dL",
                f"  Severity: {result.jaundice_result['severity'].upper()}",
                f"  Phototherapy Needed: {'YES' if result.jaundice_result['needs_phototherapy'] else 'No'}",
                "",
            ])

        # Cry analysis findings
        if result.cry_result:
            lines.extend([
                "CRY ANALYSIS:",
                f"  Status: {'ABNORMAL' if result.cry_result['is_abnormal'] else 'Normal'}",
                f"  Asphyxia Risk: {result.cry_result['asphyxia_risk']:.1%}",
                f"  Cry Type: {result.cry_result['cry_type']}",
                f"  Risk Level: {result.cry_result['risk_level'].upper()}",
                "",
            ])

        lines.extend(["-" * 60, "OVERALL ASSESSMENT", "-" * 60, ""])
        lines.append(f"Overall Risk Level: {result.overall_risk.upper()}")
        lines.append(f"Referral Needed: {'YES' if result.referral_needed else 'No'}")

        if result.priority_actions:
            lines.extend(["", "PRIORITY ACTIONS:"])
            for i, action in enumerate(result.priority_actions, 1):
                lines.append(f"  {i}. {action}")

        lines.extend(["", "=" * 60])

        return "\n".join(lines)

    def to_json(self, result: AssessmentResult) -> str:
        """Convert assessment result to JSON string."""
        data = {
            "patient": {
                "patient_id": result.patient.patient_id,
                "age_days": result.patient.age_days,
                "gestational_age": result.patient.gestational_age,
                "birth_weight": result.patient.birth_weight,
                "gender": result.patient.gender,
                "is_maternal": result.patient.is_maternal,
            },
            "timestamp": result.timestamp,
            "anemia_result": result.anemia_result,
            "jaundice_result": result.jaundice_result,
            "cry_result": result.cry_result,
            "overall_risk": result.overall_risk,
            "priority_actions": result.priority_actions,
            "referral_needed": result.referral_needed,
        }
        return json.dumps(data, indent=2)


def demo():
    """Demo the NEXUS pipeline."""
    print("NEXUS Pipeline Demo")
    print("=" * 60)

    # Initialize pipeline
    pipeline = NEXUSPipeline(lazy_load=True)

    # Demo maternal assessment
    print("\n--- Maternal Assessment Demo ---")
    maternal_patient = PatientInfo(
        patient_id="M001",
        is_maternal=True,
    )

    data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    anemia_images = list((data_dir / "eyes-defy-anemia").rglob("*.jpg"))[:1]

    if anemia_images:
        result = pipeline.assess_maternal(maternal_patient, anemia_images[0])
        print(pipeline.generate_report(result))

    # Demo neonatal assessment
    print("\n--- Neonatal Assessment Demo ---")
    neonatal_patient = PatientInfo(
        patient_id="N001",
        age_days=3,
        gestational_age=38,
        birth_weight=3200,
        gender="M",
        is_maternal=False,
    )

    jaundice_images = list((data_dir / "neojaundice" / "images").glob("*.jpg"))[:1]
    cry_files = list((data_dir / "donate-a-cry").rglob("*.wav"))[:1]

    skin_image = jaundice_images[0] if jaundice_images else None
    cry_audio = cry_files[0] if cry_files else None

    if skin_image or cry_audio:
        result = pipeline.assess_neonate(neonatal_patient, skin_image, cry_audio)
        print(pipeline.generate_report(result))


if __name__ == "__main__":
    demo()
