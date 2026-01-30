"""
HAI-DEF Integration Tests

Tests that real HAI-DEF models load and produce correct outputs.
These tests download models and require GPU/network - guarded by NEXUS_USE_REAL_MODELS=true.

Usage:
    NEXUS_USE_REAL_MODELS=true HF_TOKEN=<token> python -m pytest tests/test_hai_def_integration.py -v
"""

import sys
import os
import unittest
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

USE_REAL_MODELS = os.environ.get('NEXUS_USE_REAL_MODELS', 'false').lower() == 'true'


@unittest.skipUnless(USE_REAL_MODELS, "Requires NEXUS_USE_REAL_MODELS=true")
class TestMedSigLIPIntegration(unittest.TestCase):
    """Test MedSigLIP-448 loads and produces embeddings."""

    @classmethod
    def setUpClass(cls):
        from nexus.anemia_detector import AnemiaDetector
        cls.detector = AnemiaDetector()

    def test_loads_medsiglip(self):
        """Test that MedSigLIP-448 is loaded as primary model."""
        self.assertIn("medsiglip", self.detector.model_name)

    def test_produces_embeddings(self):
        """Test that MedSigLIP produces image embeddings."""
        import numpy as np
        from PIL import Image
        import tempfile

        # Create synthetic test image
        img = Image.new('RGB', (448, 448), color=(180, 100, 100))
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            img.save(f, format='JPEG')
            temp_path = f.name

        try:
            result = self.detector.detect(temp_path)
            self.assertIn('confidence', result)
            self.assertIn('is_anemic', result)
            self.assertGreater(result['confidence'], 0.0)
        finally:
            os.unlink(temp_path)

    def test_text_embeddings_precomputed(self):
        """Test that text embeddings are pre-computed."""
        self.assertIsNotNone(self.detector.anemic_embeddings)
        self.assertIsNotNone(self.detector.healthy_embeddings)


@unittest.skipUnless(USE_REAL_MODELS, "Requires NEXUS_USE_REAL_MODELS=true")
class TestHeARIntegration(unittest.TestCase):
    """Test HeAR-PyTorch loads and produces embeddings."""

    @classmethod
    def setUpClass(cls):
        from nexus.cry_analyzer import CryAnalyzer
        cls.analyzer = CryAnalyzer(use_hear=True)

    def test_hear_available(self):
        """Test that HeAR PyTorch model loaded successfully."""
        self.assertTrue(self.analyzer._hear_available)
        self.assertIsNotNone(self.analyzer.hear_model)

    def test_produces_embeddings(self):
        """Test that HeAR produces embeddings from audio."""
        import numpy as np

        # Create synthetic audio (2 seconds of sine wave at 16kHz)
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t)

        embeddings = self.analyzer.extract_hear_embeddings(audio)
        self.assertIsNotNone(embeddings)
        self.assertGreater(len(embeddings), 0)


@unittest.skipUnless(USE_REAL_MODELS, "Requires NEXUS_USE_REAL_MODELS=true")
class TestMedGemmaIntegration(unittest.TestCase):
    """Test MedGemma tokenizer and model load."""

    def test_tokenizer_loads(self):
        """Test that MedGemma tokenizer loads with HF token."""
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            self.skipTest("HF_TOKEN not set")

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "google/medgemma-4b-it", token=hf_token
        )
        self.assertIsNotNone(tokenizer)

    def test_synthesizer_with_medgemma(self):
        """Test clinical synthesizer initializes with MedGemma enabled."""
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            self.skipTest("HF_TOKEN not set")

        from nexus.clinical_synthesizer import ClinicalSynthesizer
        synthesizer = ClinicalSynthesizer(use_medgemma=True)
        self.assertTrue(synthesizer._medgemma_available)


@unittest.skipUnless(USE_REAL_MODELS, "Requires NEXUS_USE_REAL_MODELS=true")
class TestLinearProbeCompatibility(unittest.TestCase):
    """Test that linear probes match embedding dimensions."""

    def test_anemia_probe_dimensions(self):
        """Test anemia probe input dimensions match model output."""
        probe_path = Path(__file__).parent.parent / "models" / "linear_probes" / "anemia_linear_probe.joblib"
        if not probe_path.exists():
            self.skipTest("Anemia probe not found - run train_linear_probes.py first")

        import joblib
        probe = joblib.load(probe_path)
        probe_dim = probe.n_features_in_

        from nexus.anemia_detector import AnemiaDetector
        detector = AnemiaDetector()

        # Extract embedding to check dimensions
        import numpy as np
        from PIL import Image
        import tempfile

        img = Image.new('RGB', (448, 448), color=(180, 100, 100))
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            img.save(f, format='JPEG')
            temp_path = f.name

        try:
            import torch
            with torch.no_grad():
                inputs = detector.processor(
                    images=Image.open(temp_path).convert("RGB"),
                    return_tensors="pt"
                ).to(detector.device)

                if hasattr(detector.model, 'get_image_features'):
                    embedding = detector.model.get_image_features(**inputs)
                else:
                    outputs = detector.model(**inputs)
                    if hasattr(outputs, 'image_embeds'):
                        embedding = outputs.image_embeds
                    else:
                        embedding = outputs.vision_model_output.pooler_output

            model_dim = embedding.shape[-1]
            self.assertEqual(
                probe_dim, model_dim,
                f"Probe expects {probe_dim}-dim but model produces {model_dim}-dim. Retrain probes."
            )
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    print("=" * 60)
    print("HAI-DEF Integration Tests")
    print(f"Using real models: {USE_REAL_MODELS}")
    if not USE_REAL_MODELS:
        print("Set NEXUS_USE_REAL_MODELS=true and HF_TOKEN=<token> to run these tests")
    print("=" * 60)
    unittest.main(verbosity=2)
