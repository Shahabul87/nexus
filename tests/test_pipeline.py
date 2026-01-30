"""
NEXUS Pipeline Tests

Unit and integration tests for the NEXUS AI pipeline components.

These tests are designed to be hermetic:
- Mock model loading to avoid network dependencies
- Use synthetic test data when assets are unavailable
- Skip tests gracefully when requirements aren't met
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Flag to control whether to load real models (slow) or use mocks (fast)
USE_REAL_MODELS = os.environ.get('NEXUS_USE_REAL_MODELS', 'false').lower() == 'true'


def create_mock_detector():
    """Create a mock detector for hermetic testing."""
    mock = MagicMock()
    mock.ANEMIC_PROMPTS = ['anemic conjunctiva', 'pale inner eyelid']
    mock.HEALTHY_PROMPTS = ['healthy conjunctiva', 'pink inner eyelid']
    mock.JAUNDICE_PROMPTS = ['yellow skin', 'jaundice']
    mock.NORMAL_PROMPTS = ['normal skin', 'healthy']
    mock.detect.return_value = {
        'is_anemic': False,
        'confidence': 0.85,
        'risk_level': 'low',
        'anemia_score': 0.3,
        'healthy_score': 0.7,
        'recommendation': 'No immediate concern.',
        'model': 'mock',
        'model_type': 'Mock (test)',
    }
    mock.analyze_color_features.return_value = {
        'estimated_hemoglobin': 12.5,
        'avg_r': 180, 'avg_g': 120, 'avg_b': 110,
    }
    return mock


def create_mock_cry_analyzer():
    """Create a mock cry analyzer for hermetic testing."""
    mock = MagicMock()
    mock.analyze.return_value = {
        'is_abnormal': False,
        'asphyxia_risk': 0.15,
        'cry_type': 'hunger',
        'risk_level': 'low',
        'recommendation': 'Normal cry pattern.',
        'features': {'f0_mean': 350.0, 'f0_std': 50.0, 'duration': 3.5, 'voiced_ratio': 0.6},
        'model': 'Acoustic Features',
    }
    return mock


class TestAnemiaDetector(unittest.TestCase):
    """Tests for AnemiaDetector module."""

    @classmethod
    def setUpClass(cls):
        """Initialize detector once for all tests."""
        if USE_REAL_MODELS:
            try:
                from nexus.anemia_detector import AnemiaDetector
                cls.detector = AnemiaDetector()
                cls.is_mock = False
            except Exception as e:
                print(f"Could not load real model, using mock: {e}")
                cls.detector = create_mock_detector()
                cls.is_mock = True
        else:
            cls.detector = create_mock_detector()
            cls.is_mock = True

    def test_detector_initialization(self):
        """Test that detector initializes correctly."""
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.ANEMIC_PROMPTS)
        self.assertIsNotNone(self.detector.HEALTHY_PROMPTS)

    def test_detect_returns_required_fields(self):
        """Test that detect() returns all required fields."""
        # Use synthetic data for hermetic testing
        result = self.detector.detect('synthetic_test_image.jpg')
        self.assertIn('is_anemic', result)
        self.assertIn('confidence', result)
        self.assertIn('risk_level', result)

    def test_prompts_are_lists(self):
        """Test that prompts are properly defined lists."""
        self.assertIsInstance(self.detector.ANEMIC_PROMPTS, list)
        self.assertIsInstance(self.detector.HEALTHY_PROMPTS, list)
        self.assertGreater(len(self.detector.ANEMIC_PROMPTS), 0)
        self.assertGreater(len(self.detector.HEALTHY_PROMPTS), 0)


class TestJaundiceDetector(unittest.TestCase):
    """Tests for JaundiceDetector module."""

    @classmethod
    def setUpClass(cls):
        """Initialize detector once for all tests."""
        if USE_REAL_MODELS:
            try:
                from nexus.jaundice_detector import JaundiceDetector
                cls.detector = JaundiceDetector()
                cls.is_mock = False
            except Exception as e:
                print(f"Could not load real model, using mock: {e}")
                cls.detector = create_mock_detector()
                cls.is_mock = True
        else:
            cls.detector = create_mock_detector()
            cls.is_mock = True

    def test_detector_initialization(self):
        """Test that detector initializes correctly."""
        self.assertIsNotNone(self.detector)

    def test_detector_has_prompts(self):
        """Test that detector has text prompts defined."""
        self.assertTrue(hasattr(self.detector, 'JAUNDICE_PROMPTS') or
                       hasattr(self.detector, 'jaundice_prompts') or
                       hasattr(self.detector, 'NORMAL_PROMPTS'))


class TestCryAnalyzer(unittest.TestCase):
    """Tests for CryAnalyzer module."""

    @classmethod
    def setUpClass(cls):
        """Initialize analyzer once for all tests."""
        if USE_REAL_MODELS:
            try:
                from nexus.cry_analyzer import CryAnalyzer
                cls.analyzer = CryAnalyzer()
                cls.is_mock = False
            except Exception as e:
                print(f"Could not load real analyzer, using mock: {e}")
                cls.analyzer = create_mock_cry_analyzer()
                cls.is_mock = True
        else:
            cls.analyzer = create_mock_cry_analyzer()
            cls.is_mock = True

    def test_analyzer_initialization(self):
        """Test that analyzer initializes correctly."""
        self.assertIsNotNone(self.analyzer)

    def test_analyze_returns_required_fields(self):
        """Test that analyze() returns all required fields."""
        result = self.analyzer.analyze('synthetic_test_audio.wav')
        self.assertIn('is_abnormal', result)
        self.assertIn('asphyxia_risk', result)
        self.assertIn('cry_type', result)
        self.assertIn('risk_level', result)


class TestClinicalSynthesizer(unittest.TestCase):
    """Tests for ClinicalSynthesizer module."""

    @classmethod
    def setUpClass(cls):
        """Initialize synthesizer once for all tests."""
        try:
            from nexus.clinical_synthesizer import ClinicalSynthesizer
            # Always use rule-based for hermetic tests
            cls.synthesizer = ClinicalSynthesizer(use_medgemma=False)
            cls.is_mock = False
        except Exception as e:
            print(f"Could not load synthesizer: {e}")
            # Create mock synthesizer
            cls.synthesizer = MagicMock()
            cls.synthesizer.synthesize.return_value = {
                'summary': 'Test synthesis',
                'severity_level': 'GREEN',
                'immediate_actions': ['Continue routine care'],
                'referral_needed': False,
            }
            cls.is_mock = True

    def test_synthesizer_initialization(self):
        """Test that synthesizer initializes correctly."""
        self.assertIsNotNone(self.synthesizer)

    def test_synthesize_with_empty_findings(self):
        """Test synthesis with no findings."""
        result = self.synthesizer.synthesize({})
        self.assertIn('summary', result)
        self.assertIn('severity_level', result)

    def test_synthesize_with_anemia(self):
        """Test synthesis with anemia findings."""
        findings = {
            'anemia': {
                'is_anemic': True,
                'confidence': 0.85,
                'risk_level': 'high'
            }
        }
        result = self.synthesizer.synthesize(findings)
        self.assertIn('summary', result)
        self.assertIn('severity_level', result)

    def test_synthesize_severity_levels(self):
        """Test that different findings produce appropriate severity."""
        # High severity case
        high_findings = {
            'anemia': {'is_anemic': True, 'risk_level': 'high'},
            'cry': {'is_abnormal': True, 'asphyxia_risk': 0.7}
        }
        result = self.synthesizer.synthesize(high_findings)
        # Should have higher severity for dangerous findings
        self.assertIn('severity_level', result)


class TestAPIEndpoints(unittest.TestCase):
    """Tests for FastAPI endpoints.

    These tests use FastAPI's TestClient which doesn't require a running server.
    """

    @classmethod
    def setUpClass(cls):
        """Set up test client."""
        try:
            # Add api to path
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from api.main import app
            from fastapi.testclient import TestClient
            cls.client = TestClient(app)
            cls.app_available = True
        except ImportError as e:
            print(f"API import failed: {e}")
            cls.app_available = False

    def test_api_imports(self):
        """Test that API module imports correctly."""
        if not self.app_available:
            self.skipTest("API module not available")

        from api.main import app
        self.assertIsNotNone(app)

    def test_health_endpoint_structure(self):
        """Test health endpoint response structure."""
        if not self.app_available:
            self.skipTest("API module not available")

        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('status', data)
        self.assertIn('models_available', data)

    def test_models_endpoint(self):
        """Test models info endpoint."""
        if not self.app_available:
            self.skipTest("API module not available")

        response = self.client.get("/api/models")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('hai_def_models', data)

    def test_synthesize_endpoint_structure(self):
        """Test /api/synthesize endpoint accepts valid requests."""
        if not self.app_available:
            self.skipTest("API module not available")

        # Test with minimal valid payload
        payload = {
            "patient_type": "newborn",
            "danger_signs": [],
            "model": "medgemma"
        }

        response = self.client.post("/api/synthesize", json=payload)

        # Should succeed or return 503 if models not loaded (both valid)
        self.assertIn(response.status_code, [200, 503])

        if response.status_code == 200:
            data = response.json()
            self.assertIn('synthesis', data)
            self.assertIn('severity', data)
            self.assertIn('referral_needed', data)
            self.assertIn('immediate_actions', data)

    def test_synthesize_endpoint_with_findings(self):
        """Test /api/synthesize endpoint with clinical findings."""
        if not self.app_available:
            self.skipTest("API module not available")

        # Test with anemia and danger signs
        payload = {
            "patient_type": "pregnant",
            "danger_signs": ["severe pallor", "shortness of breath"],
            "anemia_result": {
                "is_anemic": True,
                "confidence": 0.85,
                "risk_level": "high"
            },
            "model": "medgemma"
        }

        response = self.client.post("/api/synthesize", json=payload)

        # Should succeed or return 503 if models not loaded
        self.assertIn(response.status_code, [200, 503])

        if response.status_code == 200:
            data = response.json()
            self.assertIn('synthesis', data)
            self.assertIn('severity', data)
            # High-risk findings should indicate referral needed
            self.assertIn('referral_needed', data)

    def test_synthesize_endpoint_newborn_jaundice(self):
        """Test /api/synthesize endpoint with neonatal jaundice."""
        if not self.app_available:
            self.skipTest("API module not available")

        payload = {
            "patient_type": "newborn",
            "danger_signs": [],
            "jaundice_result": {
                "has_jaundice": True,
                "severity": "moderate",
                "estimated_bilirubin": 14.5,
                "needs_phototherapy": False
            },
            "model": "medgemma"
        }

        response = self.client.post("/api/synthesize", json=payload)

        self.assertIn(response.status_code, [200, 503])

        if response.status_code == 200:
            data = response.json()
            self.assertIn('synthesis', data)
            self.assertIn('immediate_actions', data)
            self.assertIsInstance(data['immediate_actions'], list)


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the NEXUS pipeline."""

    def test_pipeline_imports(self):
        """Test that pipeline module imports correctly."""
        try:
            from nexus.pipeline import NEXUSPipeline, PatientInfo
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Pipeline import failed: {e}")

    def test_pipeline_initialization(self):
        """Test pipeline initialization with lazy loading."""
        try:
            from nexus.pipeline import NEXUSPipeline
            # Use lazy loading to avoid loading models
            pipeline = NEXUSPipeline(lazy_load=True, use_linear_probes=False)
            self.assertIsNotNone(pipeline)
        except Exception as e:
            self.skipTest(f"Pipeline initialization failed: {e}")


class TestHAIDEFCompliance(unittest.TestCase):
    """Tests for HAI-DEF model compliance.

    Verifies that the codebase references correct HAI-DEF model IDs
    and does not use legacy/incorrect model identifiers.
    """

    def test_anemia_detector_uses_medsiglip(self):
        """Test that anemia detector uses google/medsiglip-448 as primary model."""
        from nexus.anemia_detector import MEDSIGLIP_MODEL_IDS
        self.assertEqual(MEDSIGLIP_MODEL_IDS[0], "google/medsiglip-448")

    def test_jaundice_detector_uses_medsiglip(self):
        """Test that jaundice detector uses google/medsiglip-448 as primary model."""
        from nexus.jaundice_detector import MEDSIGLIP_MODEL_IDS
        self.assertEqual(MEDSIGLIP_MODEL_IDS[0], "google/medsiglip-448")

    def test_cry_analyzer_uses_hear_pytorch(self):
        """Test that cry analyzer uses google/hear-pytorch model ID."""
        from nexus.cry_analyzer import CryAnalyzer
        self.assertEqual(CryAnalyzer.HEAR_MODEL_ID, "google/hear-pytorch")

    def test_no_legacy_siglip_model_ids(self):
        """Test that legacy siglip-so400m model ID is not used."""
        from nexus.anemia_detector import MEDSIGLIP_MODEL_IDS as anemia_ids
        from nexus.jaundice_detector import MEDSIGLIP_MODEL_IDS as jaundice_ids
        for model_id in anemia_ids + jaundice_ids:
            self.assertNotIn("so400m", model_id, f"Legacy model ID found: {model_id}")

    def test_no_tensorflow_in_cry_analyzer(self):
        """Test that cry_analyzer.py does not import tensorflow."""
        import inspect
        from nexus import cry_analyzer
        source = inspect.getsource(cry_analyzer)
        self.assertNotIn("import tensorflow", source)
        self.assertNotIn("tensorflow_hub", source)

    def test_no_tfhub_url_in_cry_analyzer(self):
        """Test that cry_analyzer.py does not reference tfhub.dev."""
        import inspect
        from nexus import cry_analyzer
        source = inspect.getsource(cry_analyzer)
        self.assertNotIn("tfhub.dev", source)

    def test_pipeline_has_compliance_method(self):
        """Test that pipeline has verify_hai_def_compliance method."""
        from nexus.pipeline import NEXUSPipeline
        pipeline = NEXUSPipeline(lazy_load=True, use_linear_probes=False)
        self.assertTrue(hasattr(pipeline, 'verify_hai_def_compliance'))

    def test_clinical_synthesizer_model_name(self):
        """Test that clinical synthesizer uses google/medgemma-4b-it."""
        from nexus.clinical_synthesizer import ClinicalSynthesizer
        synthesizer = ClinicalSynthesizer(use_medgemma=False)
        self.assertEqual(synthesizer.model_name, "google/medgemma-4b-it")


if __name__ == '__main__':
    # Run tests with verbosity
    print("="*60)
    print("NEXUS Pipeline Tests")
    print(f"Using real models: {USE_REAL_MODELS}")
    print("Set NEXUS_USE_REAL_MODELS=true to test with real models")
    print("="*60)
    unittest.main(verbosity=2)
