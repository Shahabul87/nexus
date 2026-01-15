"""
NEXUS API End-to-End Tests

Tests all FastAPI endpoints for the NEXUS platform.
"""

import sys
from pathlib import Path
import pytest
import base64
from io import BytesIO

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "api"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi.testclient import TestClient
from api.main import app

# Create test client
client = TestClient(app)


# Test fixtures
@pytest.fixture
def sample_image_base64():
    """Create a simple test image in base64 format."""
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("PIL not available for image creation")

    # Create a simple red/pink image (simulating conjunctiva)
    img = Image.new('RGB', (100, 100), color=(200, 100, 100))
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


@pytest.fixture
def sample_audio_base64():
    """Create a simple test audio in base64 format."""
    try:
        import numpy as np
        import wave
    except ImportError:
        pytest.skip("numpy/wave not available for audio creation")

    # Create a simple sine wave audio
    sample_rate = 16000
    duration = 2.0  # 2 seconds
    frequency = 440  # Hz (A4 note)

    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio_data = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

    buffer = BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

    return base64.b64encode(buffer.getvalue()).decode('utf-8')


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check(self):
        """Test health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data


class TestProtocolEndpoints:
    """Tests for the WHO IMNCI protocol endpoints."""

    def test_list_protocols(self):
        """Test listing all available protocols."""
        response = client.get("/api/protocols")
        assert response.status_code == 200

        data = response.json()
        assert "protocols" in data
        assert "count" in data
        assert data["count"] >= 5  # We have at least 5 protocols

    def test_get_anemia_protocol(self):
        """Test getting anemia protocol."""
        response = client.get("/api/protocol/anemia")
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "Maternal Anemia Management"
        assert "steps" in data
        assert len(data["steps"]) > 0
        assert "referral_criteria" in data
        assert "warning_signs" in data

    def test_get_jaundice_protocol(self):
        """Test getting jaundice protocol."""
        response = client.get("/api/protocol/jaundice")
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "Neonatal Jaundice Management"
        assert "steps" in data

    def test_get_asphyxia_protocol(self):
        """Test getting asphyxia protocol."""
        response = client.get("/api/protocol/asphyxia")
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "Birth Asphyxia Management"
        assert "steps" in data

    def test_get_invalid_protocol(self):
        """Test getting non-existent protocol returns 404."""
        response = client.get("/api/protocol/invalid_condition")
        assert response.status_code == 404


class TestModelsEndpoint:
    """Tests for the /api/models endpoint."""

    def test_get_models_info(self):
        """Test getting HAI-DEF models information."""
        response = client.get("/api/models")
        assert response.status_code == 200

        data = response.json()
        assert "hai_def_models" in data
        assert "medsiglip" in data["hai_def_models"]
        assert "hear" in data["hai_def_models"]
        assert "medgemma" in data["hai_def_models"]


class TestAnemiaEndpoint:
    """Tests for the /api/anemia/detect endpoint."""

    def test_anemia_detection_requires_image(self):
        """Test that anemia endpoint requires image data."""
        response = client.post(
            "/api/anemia/detect",
            json={"image": "", "model": "medsiglip"}
        )
        # Should fail validation due to short image
        assert response.status_code == 422

    def test_anemia_detection_with_valid_image(self, sample_image_base64):
        """Test anemia detection with a valid image."""
        response = client.post(
            "/api/anemia/detect",
            json={"image": sample_image_base64, "model": "medsiglip"}
        )

        # Should return 200 if models are available, 503 if not
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "is_anemic" in data
            assert "confidence" in data
            assert "risk_level" in data
            assert "recommendation" in data


class TestJaundiceEndpoint:
    """Tests for the /api/jaundice/detect endpoint."""

    def test_jaundice_detection_requires_image(self):
        """Test that jaundice endpoint requires image data."""
        response = client.post(
            "/api/jaundice/detect",
            json={"image": "", "model": "medsiglip"}
        )
        assert response.status_code == 422

    def test_jaundice_detection_with_valid_image(self, sample_image_base64):
        """Test jaundice detection with a valid image."""
        response = client.post(
            "/api/jaundice/detect",
            json={"image": sample_image_base64, "model": "medsiglip"}
        )

        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "has_jaundice" in data
            assert "confidence" in data
            assert "severity" in data
            assert "estimated_bilirubin" in data


class TestCryEndpoint:
    """Tests for the /api/cry/analyze endpoint."""

    def test_cry_analysis_requires_audio(self):
        """Test that cry endpoint requires audio data."""
        response = client.post(
            "/api/cry/analyze",
            json={"audio": "", "model": "hear"}
        )
        assert response.status_code == 422

    def test_cry_analysis_with_valid_audio(self, sample_audio_base64):
        """Test cry analysis with valid audio."""
        response = client.post(
            "/api/cry/analyze",
            json={"audio": sample_audio_base64, "model": "hear"}
        )

        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "is_abnormal" in data
            assert "asphyxia_risk" in data
            assert "cry_type" in data
            assert "risk_level" in data


class TestSynthesizeEndpoint:
    """Tests for the /api/synthesize endpoint."""

    def test_synthesize_with_minimal_input(self):
        """Test synthesis with minimal input."""
        response = client.post(
            "/api/synthesize",
            json={
                "patient_type": "newborn",
                "danger_signs": [],
            }
        )

        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "synthesis" in data
            assert "severity" in data
            assert "referral_needed" in data

    def test_synthesize_with_findings(self):
        """Test synthesis with all findings."""
        response = client.post(
            "/api/synthesize",
            json={
                "patient_type": "newborn",
                "danger_signs": ["fever", "poor feeding"],
                "anemia_result": {
                    "is_anemic": True,
                    "confidence": 0.85,
                    "risk_level": "medium",
                },
                "jaundice_result": {
                    "has_jaundice": True,
                    "severity": "mild",
                    "needs_phototherapy": False,
                },
            }
        )

        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "synthesis" in data
            assert data["severity"] in ["GREEN", "YELLOW", "RED"]

    def test_synthesize_validates_patient_type(self):
        """Test that invalid patient_type is rejected."""
        response = client.post(
            "/api/synthesize",
            json={
                "patient_type": "invalid",
                "danger_signs": [],
            }
        )

        assert response.status_code == 422


class TestCombinedEndpoint:
    """Tests for the /api/combined/assess endpoint."""

    def test_combined_with_no_input(self):
        """Test combined assessment with no inputs (should still work)."""
        response = client.post(
            "/api/combined/assess",
            json={}
        )

        assert response.status_code in [200, 503]

    def test_combined_with_image(self, sample_image_base64):
        """Test combined assessment with only conjunctiva image."""
        response = client.post(
            "/api/combined/assess",
            json={
                "conjunctiva_image": sample_image_base64,
            }
        )

        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "summary" in data
            assert "severity_level" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
