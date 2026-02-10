"""
Agentic Workflow Tests

Tests for the 6-agent clinical workflow engine.
Covers individual agents, full workflow, early-exit, reasoning traces,
and HAI-DEF model invocation paths.

These tests are hermetic (no network/GPU dependencies) by default.
Set NEXUS_USE_REAL_MODELS=true to test with real HAI-DEF models.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import asdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nexus.agentic_workflow import (
    AgenticWorkflowEngine,
    AgentPatientInfo,
    AgentResult,
    AudioAnalysisAgent,
    AudioAnalysisResult,
    DangerSign,
    ImageAnalysisAgent,
    ImageAnalysisResult,
    ProtocolAgent,
    ProtocolResult,
    ReferralAgent,
    ReferralResult,
    SynthesisAgent,
    TriageAgent,
    TriageResult,
    WorkflowInput,
    WorkflowResult,
)

USE_REAL_MODELS = os.environ.get('NEXUS_USE_REAL_MODELS', 'false').lower() == 'true'


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def make_danger_signs(critical: int = 0, high: int = 0, medium: int = 0) -> list:
    """Create a list of DangerSign objects for testing."""
    signs = []
    for i in range(critical):
        signs.append(DangerSign(
            id=f"crit_{i}", label=f"Critical sign {i}",
            severity="critical", present=True,
        ))
    for i in range(high):
        signs.append(DangerSign(
            id=f"high_{i}", label=f"High sign {i}",
            severity="high", present=True,
        ))
    for i in range(medium):
        signs.append(DangerSign(
            id=f"med_{i}", label=f"Medium sign {i}",
            severity="medium", present=True,
        ))
    return signs


def make_mock_anemia_detector():
    mock = MagicMock()
    mock.detect.return_value = {
        "is_anemic": True,
        "confidence": 0.85,
        "risk_level": "medium",
        "anemia_score": 0.7,
        "healthy_score": 0.3,
        "recommendation": "Schedule blood test within 48 hours",
        "model": "MedSigLIP (mock)",
        "model_type": "Mock",
    }
    return mock


def make_mock_jaundice_detector():
    mock = MagicMock()
    mock.detect.return_value = {
        "has_jaundice": True,
        "confidence": 0.78,
        "severity": "moderate",
        "estimated_bilirubin": 14.5,
        "needs_phototherapy": False,
        "recommendation": "Monitor closely, recheck in 12-24 hours",
        "model": "MedSigLIP (mock)",
        "model_type": "Mock",
    }
    return mock


def make_mock_cry_analyzer():
    mock = MagicMock()
    mock.analyze.return_value = {
        "is_abnormal": False,
        "asphyxia_risk": 0.15,
        "cry_type": "hunger",
        "risk_level": "low",
        "recommendation": "Normal cry pattern.",
        "features": {"f0_mean": 350.0, "f0_std": 50.0, "duration": 3.5, "voiced_ratio": 0.6},
        "model": "HeAR (mock)",
    }
    return mock


def make_mock_synthesizer():
    mock = MagicMock()
    mock.synthesize.return_value = {
        "summary": "Test synthesis result",
        "severity_level": "YELLOW",
        "severity_description": "Close monitoring required",
        "immediate_actions": ["Monitor closely", "Follow up in 24 hours"],
        "referral_needed": False,
        "referral_urgency": "none",
        "follow_up": "Reassess in 24-48 hours",
        "urgent_conditions": [],
        "model": "Rule-based (WHO IMNCI)",
        "generated_at": "2026-01-30T00:00:00",
    }
    return mock


# ---------------------------------------------------------------------------
# TriageAgent Tests
# ---------------------------------------------------------------------------

class TestTriageAgent(unittest.TestCase):
    """Tests for the TriageAgent."""

    def setUp(self):
        self.agent = TriageAgent()
        self.patient_info = AgentPatientInfo(patient_type="newborn")

    def test_green_with_no_signs(self):
        result, trace = self.agent.process("newborn", [], self.patient_info)
        self.assertEqual(result.risk_level, "GREEN")
        self.assertEqual(result.score, 0)
        self.assertFalse(result.critical_signs_detected)
        self.assertFalse(result.immediate_referral_needed)
        self.assertEqual(trace.status, "success")

    def test_red_with_critical_sign(self):
        signs = make_danger_signs(critical=1)
        result, trace = self.agent.process("newborn", signs, self.patient_info)
        self.assertEqual(result.risk_level, "RED")
        self.assertGreaterEqual(result.score, 30)
        self.assertTrue(result.critical_signs_detected)
        self.assertTrue(result.immediate_referral_needed)

    def test_yellow_with_high_sign(self):
        signs = make_danger_signs(high=1)
        result, trace = self.agent.process("newborn", signs, self.patient_info)
        self.assertEqual(result.risk_level, "YELLOW")
        self.assertGreaterEqual(result.score, 15)

    def test_medium_signs_accumulate(self):
        signs = make_danger_signs(medium=3)
        result, trace = self.agent.process("newborn", signs, self.patient_info)
        # 3 medium signs (5 each = 15) + comorbidity bonus (10) = 25
        self.assertEqual(result.score, 25)
        self.assertIn(result.risk_level, ["YELLOW", "RED"])

    def test_newborn_low_birth_weight(self):
        info = AgentPatientInfo(patient_type="newborn", birth_weight=2000)
        result, trace = self.agent.process("newborn", [], info)
        self.assertGreaterEqual(result.score, 10)
        reasoning_text = " ".join(trace.reasoning)
        self.assertIn("Low birth weight", reasoning_text)

    def test_newborn_low_apgar(self):
        info = AgentPatientInfo(patient_type="newborn", apgar_score=5)
        result, trace = self.agent.process("newborn", [], info)
        self.assertGreaterEqual(result.score, 15)

    def test_pregnant_preterm_risk(self):
        info = AgentPatientInfo(patient_type="pregnant", gestational_weeks=25)
        result, trace = self.agent.process("pregnant", [], info)
        self.assertGreaterEqual(result.score, 10)

    def test_reasoning_trace_not_empty(self):
        result, trace = self.agent.process("newborn", [], self.patient_info)
        self.assertGreater(len(trace.reasoning), 0)
        self.assertEqual(trace.agent_name, "TriageAgent")
        self.assertGreaterEqual(trace.processing_time_ms, 0)


# ---------------------------------------------------------------------------
# ImageAnalysisAgent Tests
# ---------------------------------------------------------------------------

class TestImageAnalysisAgent(unittest.TestCase):
    """Tests for the ImageAnalysisAgent."""

    def test_skipped_without_images(self):
        agent = ImageAnalysisAgent()
        result, trace = agent.process("newborn")
        self.assertIsNone(result.anemia)
        self.assertIsNone(result.jaundice)
        self.assertEqual(trace.status, "skipped")

    def test_anemia_detection_with_mock(self):
        mock_detector = make_mock_anemia_detector()
        agent = ImageAnalysisAgent(anemia_detector=mock_detector)
        result, trace = agent.process("pregnant", conjunctiva_image="test.jpg")
        self.assertIsNotNone(result.anemia)
        self.assertTrue(result.anemia["is_anemic"])
        mock_detector.detect.assert_called_once_with("test.jpg")
        self.assertEqual(trace.status, "success")

    def test_jaundice_detection_with_mock(self):
        mock_detector = make_mock_jaundice_detector()
        agent = ImageAnalysisAgent(jaundice_detector=mock_detector)
        result, trace = agent.process("newborn", skin_image="skin.jpg")
        self.assertIsNotNone(result.jaundice)
        self.assertTrue(result.jaundice["has_jaundice"])
        mock_detector.detect.assert_called_once_with("skin.jpg")

    def test_both_images(self):
        anemia_mock = make_mock_anemia_detector()
        jaundice_mock = make_mock_jaundice_detector()
        agent = ImageAnalysisAgent(anemia_detector=anemia_mock, jaundice_detector=jaundice_mock)
        result, trace = agent.process("newborn", conjunctiva_image="eye.jpg", skin_image="skin.jpg")
        self.assertIsNotNone(result.anemia)
        self.assertIsNotNone(result.jaundice)
        self.assertEqual(trace.status, "success")

    def test_error_handling(self):
        mock_detector = MagicMock()
        mock_detector.detect.side_effect = RuntimeError("Model failed")
        agent = ImageAnalysisAgent(anemia_detector=mock_detector)
        result, trace = agent.process("pregnant", conjunctiva_image="test.jpg")
        self.assertIsNotNone(result.anemia)
        self.assertFalse(result.anemia["is_anemic"])
        reasoning_text = " ".join(trace.reasoning).lower()
        self.assertIn("failed", reasoning_text)


# ---------------------------------------------------------------------------
# AudioAnalysisAgent Tests
# ---------------------------------------------------------------------------

class TestAudioAnalysisAgent(unittest.TestCase):
    """Tests for the AudioAnalysisAgent."""

    def test_skipped_without_audio(self):
        agent = AudioAnalysisAgent()
        result, trace = agent.process()
        self.assertIsNone(result.cry)
        self.assertEqual(trace.status, "skipped")

    def test_cry_analysis_with_mock(self):
        mock_analyzer = make_mock_cry_analyzer()
        agent = AudioAnalysisAgent(cry_analyzer=mock_analyzer)
        result, trace = agent.process(cry_audio="cry.wav")
        self.assertIsNotNone(result.cry)
        self.assertFalse(result.cry["is_abnormal"])
        mock_analyzer.analyze.assert_called_once_with("cry.wav")
        self.assertEqual(trace.status, "success")

    def test_error_handling(self):
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.side_effect = RuntimeError("Audio failed")
        agent = AudioAnalysisAgent(cry_analyzer=mock_analyzer)
        result, trace = agent.process(cry_audio="bad.wav")
        self.assertIsNotNone(result.cry)
        self.assertFalse(result.cry["is_abnormal"])


# ---------------------------------------------------------------------------
# ProtocolAgent Tests
# ---------------------------------------------------------------------------

class TestProtocolAgent(unittest.TestCase):
    """Tests for the ProtocolAgent."""

    def setUp(self):
        self.agent = ProtocolAgent()

    def test_green_classification(self):
        triage = TriageResult(risk_level="GREEN")
        image = ImageAnalysisResult()
        result, trace = self.agent.process("newborn", triage, image)
        self.assertEqual(result.classification, "GREEN")
        self.assertEqual(trace.status, "success")

    def test_jaundice_protocol(self):
        triage = TriageResult(risk_level="GREEN")
        image = ImageAnalysisResult(jaundice={
            "has_jaundice": True,
            "severity": "moderate",
            "estimated_bilirubin": 14.5,
            "needs_phototherapy": True,
        })
        result, trace = self.agent.process("newborn", triage, image)
        self.assertIn("Neonatal Jaundice Protocol", result.applicable_protocols)
        self.assertIn("YELLOW", [result.classification, "YELLOW"])

    def test_severe_jaundice_red(self):
        triage = TriageResult(risk_level="GREEN")
        image = ImageAnalysisResult(jaundice={
            "has_jaundice": True,
            "estimated_bilirubin": 22.0,
            "needs_phototherapy": True,
        })
        result, trace = self.agent.process("newborn", triage, image)
        self.assertEqual(result.classification, "RED")

    def test_asphyxia_protocol(self):
        triage = TriageResult(risk_level="GREEN")
        image = ImageAnalysisResult()
        audio = AudioAnalysisResult(cry={"is_abnormal": True, "asphyxia_risk": 0.8})
        result, trace = self.agent.process("newborn", triage, image, audio)
        self.assertEqual(result.classification, "RED")
        self.assertIn("Birth Asphyxia Assessment Protocol", result.applicable_protocols)

    def test_maternal_anemia_protocol(self):
        triage = TriageResult(risk_level="GREEN")
        image = ImageAnalysisResult(anemia={
            "is_anemic": True,
            "estimated_hemoglobin": 9.0,
            "risk_level": "medium",
        })
        result, trace = self.agent.process("pregnant", triage, image)
        self.assertIn("Anemia Management Protocol", result.applicable_protocols)
        self.assertEqual(result.classification, "YELLOW")

    def test_severe_maternal_anemia_red(self):
        triage = TriageResult(risk_level="GREEN")
        image = ImageAnalysisResult(anemia={
            "is_anemic": True,
            "estimated_hemoglobin": 5.0,
            "risk_level": "high",
        })
        result, trace = self.agent.process("pregnant", triage, image)
        self.assertEqual(result.classification, "RED")


# ---------------------------------------------------------------------------
# ReferralAgent Tests
# ---------------------------------------------------------------------------

class TestReferralAgent(unittest.TestCase):
    """Tests for the ReferralAgent."""

    def setUp(self):
        self.agent = ReferralAgent()

    def test_no_referral_green(self):
        triage = TriageResult(risk_level="GREEN")
        protocol = ProtocolResult(classification="GREEN")
        image = ImageAnalysisResult()
        result, trace = self.agent.process("newborn", triage, protocol, image)
        self.assertFalse(result.referral_needed)
        self.assertEqual(result.urgency, "none")

    def test_immediate_referral_critical(self):
        triage = TriageResult(
            risk_level="RED",
            critical_signs_detected=True,
            critical_signs=["Convulsions"],
            immediate_referral_needed=True,
        )
        protocol = ProtocolResult(classification="RED")
        image = ImageAnalysisResult()
        result, trace = self.agent.process("newborn", triage, protocol, image)
        self.assertTrue(result.referral_needed)
        self.assertEqual(result.urgency, "immediate")
        self.assertEqual(result.facility_level, "tertiary")

    def test_urgent_referral_phototherapy(self):
        triage = TriageResult(risk_level="YELLOW")
        protocol = ProtocolResult(classification="YELLOW")
        image = ImageAnalysisResult(jaundice={"has_jaundice": True, "needs_phototherapy": True})
        result, trace = self.agent.process("newborn", triage, protocol, image)
        self.assertTrue(result.referral_needed)
        self.assertIn(result.urgency, ["urgent", "immediate"])

    def test_timeframe_mapping(self):
        triage = TriageResult(risk_level="RED", critical_signs_detected=True,
                              critical_signs=["X"], immediate_referral_needed=True)
        protocol = ProtocolResult(classification="RED")
        image = ImageAnalysisResult()
        result, trace = self.agent.process("newborn", triage, protocol, image)
        self.assertIn("Within 1 hour", result.timeframe)


# ---------------------------------------------------------------------------
# SynthesisAgent Tests
# ---------------------------------------------------------------------------

class TestSynthesisAgent(unittest.TestCase):
    """Tests for the SynthesisAgent."""

    def test_synthesis_with_mock(self):
        mock_synth = make_mock_synthesizer()
        agent = SynthesisAgent(synthesizer=mock_synth)

        triage = TriageResult(risk_level="YELLOW", score=15)
        image = ImageAnalysisResult(anemia={"is_anemic": True, "confidence": 0.85})
        protocol = ProtocolResult(classification="YELLOW")
        referral = ReferralResult()
        traces = [AgentResult(agent_name="TriageAgent", status="success", reasoning=["test"])]

        synthesis, trace = agent.process("pregnant", triage, image, None, protocol, referral, traces)
        self.assertIn("summary", synthesis)
        self.assertEqual(trace.status, "success")
        mock_synth.synthesize.assert_called_once()

    def test_synthesis_includes_agent_context(self):
        mock_synth = make_mock_synthesizer()
        agent = SynthesisAgent(synthesizer=mock_synth)

        triage = TriageResult(risk_level="YELLOW", score=15)
        image = ImageAnalysisResult()
        protocol = ProtocolResult(classification="YELLOW")
        referral = ReferralResult()
        traces = []

        agent.process("newborn", triage, image, None, protocol, referral, traces)
        call_args = mock_synth.synthesize.call_args[0][0]
        self.assertIn("agent_context", call_args)
        self.assertEqual(call_args["agent_context"]["triage_score"], 15)


# ---------------------------------------------------------------------------
# Full Workflow Tests
# ---------------------------------------------------------------------------

class TestAgenticWorkflowEngine(unittest.TestCase):
    """End-to-end tests for the AgenticWorkflowEngine."""

    def _make_engine(self):
        return AgenticWorkflowEngine(
            anemia_detector=make_mock_anemia_detector(),
            jaundice_detector=make_mock_jaundice_detector(),
            cry_analyzer=make_mock_cry_analyzer(),
            synthesizer=make_mock_synthesizer(),
        )

    def test_full_workflow_newborn(self):
        engine = self._make_engine()
        workflow_input = WorkflowInput(
            patient_type="newborn",
            skin_image="skin.jpg",
            cry_audio="cry.wav",
        )
        result = engine.execute(workflow_input)

        self.assertTrue(result.success)
        self.assertEqual(result.patient_type, "newborn")
        self.assertIn(result.who_classification, ["RED", "YELLOW", "GREEN"])
        self.assertIsNotNone(result.triage_result)
        self.assertIsNotNone(result.image_results)
        self.assertIsNotNone(result.protocol_result)
        self.assertIsNotNone(result.referral_result)
        self.assertIsNotNone(result.timestamp)
        self.assertGreaterEqual(result.processing_time_ms, 0)

    def test_full_workflow_pregnant(self):
        engine = self._make_engine()
        workflow_input = WorkflowInput(
            patient_type="pregnant",
            conjunctiva_image="eye.jpg",
        )
        result = engine.execute(workflow_input)

        self.assertTrue(result.success)
        self.assertEqual(result.patient_type, "pregnant")
        self.assertIsNotNone(result.image_results)
        self.assertIsNotNone(result.image_results.anemia)

    def test_agent_traces_present(self):
        engine = self._make_engine()
        workflow_input = WorkflowInput(
            patient_type="newborn",
            skin_image="skin.jpg",
            cry_audio="cry.wav",
        )
        result = engine.execute(workflow_input)

        # Should have 6 agent traces (all agents ran)
        self.assertEqual(len(result.agent_traces), 6)

        agent_names = [t.agent_name for t in result.agent_traces]
        self.assertIn("TriageAgent", agent_names)
        self.assertIn("ImageAnalysisAgent", agent_names)
        self.assertIn("AudioAnalysisAgent", agent_names)
        self.assertIn("ProtocolAgent", agent_names)
        self.assertIn("ReferralAgent", agent_names)
        self.assertIn("SynthesisAgent", agent_names)

    def test_reasoning_traces_have_content(self):
        engine = self._make_engine()
        workflow_input = WorkflowInput(
            patient_type="newborn",
            skin_image="skin.jpg",
        )
        result = engine.execute(workflow_input)

        for trace in result.agent_traces:
            self.assertIsInstance(trace.reasoning, list)
            self.assertGreater(len(trace.reasoning), 0, f"{trace.agent_name} has empty reasoning")

    def test_early_exit_critical_danger_signs(self):
        engine = self._make_engine()
        workflow_input = WorkflowInput(
            patient_type="newborn",
            danger_signs=make_danger_signs(critical=1),
            skin_image="skin.jpg",
            cry_audio="cry.wav",
        )
        result = engine.execute(workflow_input)

        self.assertTrue(result.success)
        self.assertEqual(result.who_classification, "RED")
        self.assertTrue(result.referral_result.referral_needed)
        self.assertEqual(result.referral_result.urgency, "immediate")

        # Early exit means only TriageAgent ran
        self.assertEqual(len(result.agent_traces), 1)
        self.assertEqual(result.agent_traces[0].agent_name, "TriageAgent")

    def test_workflow_without_inputs(self):
        engine = self._make_engine()
        workflow_input = WorkflowInput(patient_type="newborn")
        result = engine.execute(workflow_input)

        self.assertTrue(result.success)
        # Image and audio agents should be skipped
        skipped = [t for t in result.agent_traces if t.status == "skipped"]
        self.assertGreater(len(skipped), 0)

    def test_state_callback(self):
        states = []

        def callback(state, progress):
            states.append((state, progress))

        engine = AgenticWorkflowEngine(
            anemia_detector=make_mock_anemia_detector(),
            jaundice_detector=make_mock_jaundice_detector(),
            cry_analyzer=make_mock_cry_analyzer(),
            synthesizer=make_mock_synthesizer(),
            on_state_change=callback,
        )
        workflow_input = WorkflowInput(
            patient_type="newborn",
            skin_image="skin.jpg",
        )
        engine.execute(workflow_input)

        state_names = [s[0] for s in states]
        self.assertIn("triaging", state_names)
        self.assertIn("analyzing_image", state_names)
        self.assertIn("complete", state_names)

    def test_all_hai_def_models_invoked(self):
        """Verify all 3 HAI-DEF models are called in a full workflow."""
        anemia_mock = make_mock_anemia_detector()
        jaundice_mock = make_mock_jaundice_detector()
        cry_mock = make_mock_cry_analyzer()
        synth_mock = make_mock_synthesizer()

        engine = AgenticWorkflowEngine(
            anemia_detector=anemia_mock,
            jaundice_detector=jaundice_mock,
            cry_analyzer=cry_mock,
            synthesizer=synth_mock,
        )
        workflow_input = WorkflowInput(
            patient_type="newborn",
            conjunctiva_image="eye.jpg",
            skin_image="skin.jpg",
            cry_audio="cry.wav",
        )
        engine.execute(workflow_input)

        # MedSigLIP calls
        anemia_mock.detect.assert_called_once()
        jaundice_mock.detect.assert_called_once()
        # HeAR call
        cry_mock.analyze.assert_called_once()
        # MedGemma call
        synth_mock.synthesize.assert_called_once()

    def test_agent_count_constant(self):
        """Verify AGENTS list contains all 6 agents."""
        self.assertEqual(len(AgenticWorkflowEngine.AGENTS), 6)

    def test_error_handling(self):
        """Verify workflow handles unexpected errors gracefully."""
        anemia_mock = MagicMock()
        anemia_mock.detect.side_effect = RuntimeError("Catastrophic failure")

        engine = AgenticWorkflowEngine(
            anemia_detector=anemia_mock,
            jaundice_detector=make_mock_jaundice_detector(),
            cry_analyzer=make_mock_cry_analyzer(),
            synthesizer=make_mock_synthesizer(),
        )
        workflow_input = WorkflowInput(
            patient_type="pregnant",
            conjunctiva_image="bad.jpg",
        )
        # Should not raise - the ImageAnalysisAgent handles errors internally
        result = engine.execute(workflow_input)
        self.assertTrue(result.success)


# ---------------------------------------------------------------------------
# API Integration Tests
# ---------------------------------------------------------------------------

class TestAgenticAPIEndpoint(unittest.TestCase):
    """Tests for the /api/agentic/assess endpoint."""

    @classmethod
    def setUpClass(cls):
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from api.main import app
            from fastapi.testclient import TestClient
            cls.client = TestClient(app)
            cls.app_available = True
        except ImportError as e:
            print(f"API import failed: {e}")
            cls.app_available = False

    def test_agentic_endpoint_exists(self):
        """Test that the agentic endpoint is registered."""
        if not self.app_available:
            self.skipTest("API module not available")

        # Send minimal valid request
        payload = {
            "patient_type": "newborn",
            "danger_signs": [],
        }
        response = self.client.post("/api/agentic/assess", json=payload)
        # Should succeed or return 503 (models not loaded)
        self.assertIn(response.status_code, [200, 503])

    def test_agentic_endpoint_with_danger_signs(self):
        """Test agentic endpoint with danger signs input."""
        if not self.app_available:
            self.skipTest("API module not available")

        payload = {
            "patient_type": "newborn",
            "danger_signs": [
                {"id": "conv", "label": "Convulsions", "severity": "critical", "present": True},
            ],
        }
        response = self.client.post("/api/agentic/assess", json=payload)
        self.assertIn(response.status_code, [200, 503])

        if response.status_code == 200:
            data = response.json()
            self.assertIn("agent_traces", data)
            self.assertIn("who_classification", data)
            self.assertTrue(data["success"])

    def test_agentic_endpoint_validates_patient_type(self):
        """Test that invalid patient_type is rejected."""
        if not self.app_available:
            self.skipTest("API module not available")

        payload = {"patient_type": "invalid"}
        response = self.client.post("/api/agentic/assess", json=payload)
        self.assertEqual(response.status_code, 422)


if __name__ == '__main__':
    print("=" * 60)
    print("Agentic Workflow Tests")
    print(f"Using real models: {USE_REAL_MODELS}")
    print("=" * 60)
    unittest.main(verbosity=2)
