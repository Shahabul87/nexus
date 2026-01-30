"""
Cry Analyzer Module

Uses HeAR from Google HAI-DEF for infant cry analysis and birth asphyxia detection.
Implements embedding extraction + linear classifier per NEXUS_MASTER_PLAN.md.

HAI-DEF Model: HeAR (Health Acoustic Representations)
Source: https://github.com/Google-Health/google-health/tree/master/health_acoustic_representations
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
import os

try:
    import librosa
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

try:
    from sklearn.linear_model import LogisticRegression
    import joblib
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# HeAR PyTorch via HuggingFace
try:
    from transformers import AutoModel as HearAutoModel
    HAS_HEAR_PYTORCH = True
except ImportError:
    HAS_HEAR_PYTORCH = False


class CryAnalyzer:
    """
    Analyzes infant cry audio for birth asphyxia detection using HeAR.

    HAI-DEF Model: HeAR (google/hear-pytorch)
    Method: Embedding extraction + linear classifier
    Expected Accuracy: 85-93% (per NEXUS_MASTER_PLAN.md)

    Process:
    1. Split audio into 2-second chunks (HeAR requirement)
    2. Extract HeAR embeddings (512-dim per chunk)
    3. Aggregate embeddings (mean pooling)
    4. Classify with trained linear model or rule-based fallback
    """

    # HeAR model configuration
    SAMPLE_RATE = 16000           # Hz - HeAR requires 16kHz
    CHUNK_DURATION = 2.0          # seconds - HeAR chunk size
    CHUNK_SIZE = 32000            # samples (2 seconds at 16kHz)
    EMBEDDING_DIM = 512           # HeAR embedding dimension

    # Acoustic feature thresholds (fallback if HeAR unavailable)
    NORMAL_F0_RANGE = (250, 450)  # Hz
    ASPHYXIA_F0_THRESHOLD = 500   # Hz - higher F0 indicates distress
    MIN_CRY_DURATION = 0.5        # seconds

    # HeAR model ID on HuggingFace (PyTorch)
    HEAR_MODEL_ID = "google/hear-pytorch"

    def __init__(
        self,
        device: Optional[str] = None,
        classifier_path: Optional[str] = None,
        use_hear: bool = True,
    ):
        """
        Initialize the Cry Analyzer with HeAR.

        Args:
            device: Device to run model on
            classifier_path: Path to trained linear classifier (optional)
            use_hear: Whether to use HeAR embeddings (True) or acoustic features (False)
        """
        if not HAS_AUDIO:
            raise ImportError("librosa and soundfile required. Install with: pip install librosa soundfile")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier_path = classifier_path
        self.classifier = None
        self.hear_model = None
        self.use_hear = use_hear
        self._hear_available = False

        # Try to load HeAR model
        if use_hear:
            self._load_hear_model()

        # Load trained classifier if provided
        if classifier_path and Path(classifier_path).exists() and HAS_SKLEARN:
            self.classifier = joblib.load(classifier_path)
            print(f"Loaded classifier from {classifier_path}")

        mode = "HeAR" if self._hear_available else "Acoustic Features (HeAR unavailable)"
        print(f"Cry Analyzer (HAI-DEF {mode}) initialized on {self.device}")

    def _load_hear_model(self) -> None:
        """Load HeAR model from HuggingFace (PyTorch).

        HeAR (Health Acoustic Representations) is a Google HAI-DEF model
        for health-related audio analysis. It produces 512-dimensional
        embeddings from 2-second audio chunks at 16kHz.
        """
        if not HAS_HEAR_PYTORCH:
            print("Warning: transformers not available. Install with: pip install transformers")
            print("Falling back to acoustic feature extraction (deterministic)")
            self._hear_available = False
            return

        hf_token = os.environ.get("HF_TOKEN")

        try:
            print(f"Loading HeAR model from HuggingFace: {self.HEAR_MODEL_ID}")
            self.hear_model = HearAutoModel.from_pretrained(
                self.HEAR_MODEL_ID,
                token=hf_token,
                trust_remote_code=True,
            )
            self.hear_model = self.hear_model.to(self.device)
            self.hear_model.eval()
            self._hear_available = True
            print("HeAR model loaded successfully (PyTorch)")

        except Exception as e:
            print(f"Warning: Could not load HeAR model: {e}")
            print("Falling back to acoustic feature extraction (deterministic)")
            self.hear_model = None
            self._hear_available = False

    def _split_audio_chunks(self, audio: np.ndarray) -> List[np.ndarray]:
        """
        Split audio into 2-second chunks for HeAR processing.

        Args:
            audio: Audio signal array (16kHz)

        Returns:
            List of audio chunks (each 2 seconds / 32000 samples)
        """
        chunks = []
        for i in range(0, len(audio), self.CHUNK_SIZE):
            chunk = audio[i:i + self.CHUNK_SIZE]
            if len(chunk) < self.CHUNK_SIZE:
                # Pad with zeros if needed
                chunk = np.pad(chunk, (0, self.CHUNK_SIZE - len(chunk)))
            chunks.append(chunk)
        return chunks

    def extract_hear_embeddings(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract HeAR embeddings from audio using PyTorch.

        HeAR is a ViT model that expects mel-PCEN spectrograms, not raw audio.
        Pipeline: raw audio (32000 samples) → preprocess_audio() → (1, 1, 192, 128)
                  → ViT forward pass → pool last_hidden_state → embedding

        Args:
            audio: Audio signal (16kHz)

        Returns:
            Aggregated embedding (HeAR hidden_size dim, or 8-dim fallback)
        """
        if not self._hear_available or self.hear_model is None:
            # Fallback: use acoustic features as pseudo-embeddings
            # This is deterministic - same audio always produces same features
            features = self.extract_features(audio, self.SAMPLE_RATE)
            # Create a feature vector from acoustic features
            feature_vector = np.array([
                features.get("f0_mean", 0),
                features.get("f0_std", 0),
                features.get("f0_range", 0),
                features.get("voiced_ratio", 0),
                features.get("spectral_centroid_mean", 0),
                features.get("spectral_bandwidth_mean", 0),
                features.get("zcr_mean", 0),
                features.get("rms_mean", 0),
            ])
            return feature_vector

        from nexus.hear_preprocessing import preprocess_audio

        # Split into 2-second chunks for HeAR
        chunks = self._split_audio_chunks(audio)

        # Extract embeddings for each chunk using HeAR (PyTorch)
        embeddings = []
        with torch.no_grad():
            for chunk in chunks:
                # Convert raw audio to tensor: (1, 32000)
                chunk_tensor = torch.tensor(
                    chunk.astype(np.float32)
                ).unsqueeze(0).to(self.device)

                # Preprocess: raw audio → mel-PCEN spectrogram (1, 1, 192, 128)
                spectrogram = preprocess_audio(chunk_tensor)

                # Forward pass: HeAR ViT expects pixel_values
                output = self.hear_model(
                    pixel_values=spectrogram,
                    return_dict=True,
                )

                # Extract embedding from ViT output
                if hasattr(output, 'pooler_output') and output.pooler_output is not None:
                    embedding = output.pooler_output
                elif hasattr(output, 'last_hidden_state'):
                    # Mean pool over sequence dimension (skip CLS token)
                    embedding = output.last_hidden_state[:, 1:, :].mean(dim=1)
                elif isinstance(output, torch.Tensor):
                    embedding = output
                else:
                    embedding = list(output.values())[0] if hasattr(output, 'values') else output[0]

                embeddings.append(embedding.cpu().numpy().squeeze())

        # Aggregate embeddings (mean pooling across chunks)
        aggregated = np.mean(embeddings, axis=0)
        return aggregated

    def load_audio(
        self,
        audio_path: Union[str, Path],
        sr: int = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file.

        Args:
            audio_path: Path to audio file
            sr: Target sample rate (uses file's native if None)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        sr = sr or self.SAMPLE_RATE
        audio, file_sr = librosa.load(audio_path, sr=sr)
        return audio, sr

    def extract_features(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Extract acoustic features from cry audio.

        Features based on cry analysis literature:
        - Fundamental frequency (F0)
        - MFCCs (mel-frequency cepstral coefficients)
        - Spectral features
        - Temporal features

        Args:
            audio: Audio signal array
            sr: Sample rate

        Returns:
            Dictionary of extracted features
        """
        features = {}

        # Ensure minimum length
        if len(audio) < sr * self.MIN_CRY_DURATION:
            # Pad if too short
            audio = np.pad(audio, (0, int(sr * self.MIN_CRY_DURATION) - len(audio)))

        # Duration
        features["duration"] = len(audio) / sr

        # Fundamental frequency (F0) using pyin
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr,
            )

        # F0 statistics (ignoring unvoiced frames)
        f0_valid = f0[~np.isnan(f0)]
        if len(f0_valid) > 0:
            features["f0_mean"] = float(np.mean(f0_valid))
            features["f0_std"] = float(np.std(f0_valid))
            features["f0_min"] = float(np.min(f0_valid))
            features["f0_max"] = float(np.max(f0_valid))
            features["f0_range"] = features["f0_max"] - features["f0_min"]
        else:
            features["f0_mean"] = 0
            features["f0_std"] = 0
            features["f0_min"] = 0
            features["f0_max"] = 0
            features["f0_range"] = 0

        # Voiced ratio (cry vs silence)
        features["voiced_ratio"] = float(np.mean(voiced_flag))

        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f"mfcc_{i}_mean"] = float(np.mean(mfccs[i]))
            features[f"mfcc_{i}_std"] = float(np.std(mfccs[i]))

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)

        features["spectral_centroid_mean"] = float(np.mean(spectral_centroid))
        features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))
        features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))

        # Zero crossing rate (higher in noisy/irregular cries)
        zcr = librosa.feature.zero_crossing_rate(audio)
        features["zcr_mean"] = float(np.mean(zcr))
        features["zcr_std"] = float(np.std(zcr))

        # RMS energy
        rms = librosa.feature.rms(y=audio)
        features["rms_mean"] = float(np.mean(rms))
        features["rms_std"] = float(np.std(rms))

        # Tempo estimation (cry rhythm)
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
        features["tempo"] = float(tempo[0]) if len(tempo) > 0 else 0

        return features

    def analyze(self, audio_path: Union[str, Path]) -> Dict:
        """
        Analyze cry audio for health indicators.

        Uses HeAR embeddings + classifier when available, falls back to
        rule-based acoustic analysis when HeAR is unavailable.

        Args:
            audio_path: Path to cry audio file

        Returns:
            Dictionary containing:
                - is_abnormal: Boolean indicating abnormal cry
                - asphyxia_risk: Risk score for birth asphyxia (0-1)
                - cry_type: Detected cry type
                - features: Extracted acoustic features
                - risk_level: "low", "medium", "high"
                - recommendation: Clinical recommendation
        """
        # Load audio
        audio, sr = self.load_audio(audio_path)

        # Extract acoustic features (always needed for cry_type and feature reporting)
        features = self.extract_features(audio, sr)

        # Determine cry type based on acoustic features
        cry_type = self._classify_cry_type(features)

        # Try HeAR-based classification first
        if self._hear_available and self.hear_model is not None:
            asphyxia_risk, model_used = self._analyze_with_hear(audio)
        else:
            asphyxia_risk, model_used = self._analyze_with_rules(features)

        # Determine risk level and recommendation based on risk score
        if asphyxia_risk > 0.6:
            risk_level = "high"
            is_abnormal = True
            recommendation = "URGENT: High-pitched abnormal cry detected. Assess for birth asphyxia immediately. Check APGAR score and vital signs."
        elif asphyxia_risk > 0.3:
            risk_level = "medium"
            is_abnormal = True
            recommendation = "CAUTION: Some abnormal cry characteristics. Monitor closely and reassess in 30 minutes."
        else:
            risk_level = "low"
            is_abnormal = False
            recommendation = "Normal cry pattern. Continue routine care."

        return {
            "is_abnormal": is_abnormal,
            "asphyxia_risk": round(asphyxia_risk, 3),
            "cry_type": cry_type,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "features": {
                "f0_mean": round(features["f0_mean"], 1),
                "f0_std": round(features["f0_std"], 1),
                "duration": round(features["duration"], 2),
                "voiced_ratio": round(features["voiced_ratio"], 2),
            },
            "model": model_used,
            "model_note": self._get_model_note(model_used),
        }

    def _analyze_with_hear(self, audio: np.ndarray) -> Tuple[float, str]:
        """
        Analyze cry using HeAR embeddings.

        Args:
            audio: Audio signal array (16kHz)

        Returns:
            Tuple of (asphyxia_risk, model_name)
        """
        # Extract HeAR embeddings
        embeddings = self.extract_hear_embeddings(audio)

        # Use trained classifier if available
        if self.classifier is not None and HAS_SKLEARN:
            # Classifier expects 2D input: [n_samples, n_features]
            embeddings_2d = embeddings.reshape(1, -1)

            # Get probability prediction
            if hasattr(self.classifier, 'predict_proba'):
                proba = self.classifier.predict_proba(embeddings_2d)
                # Assume binary classification: [normal, asphyxia]
                asphyxia_risk = float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0, 0])
            else:
                # Fallback to binary prediction
                prediction = self.classifier.predict(embeddings_2d)
                asphyxia_risk = float(prediction[0])

            return asphyxia_risk, "HeAR + Classifier"

        # No classifier: use embedding-based heuristic
        # HeAR embeddings capture acoustic patterns; we use simple statistics
        # This is a fallback when no trained classifier is available
        embedding_mean = float(np.mean(embeddings))
        embedding_std = float(np.std(embeddings))
        embedding_max = float(np.max(np.abs(embeddings)))

        # Heuristic: abnormal cries tend to have higher variance in embeddings
        # These thresholds should be calibrated on labeled data
        risk_score = 0.0
        if embedding_std > 0.5:
            risk_score += 0.3
        if embedding_max > 2.0:
            risk_score += 0.2
        if abs(embedding_mean) > 0.3:
            risk_score += 0.2

        return min(risk_score, 1.0), "HeAR (uncalibrated)"

    def _analyze_with_rules(self, features: Dict) -> Tuple[float, str]:
        """
        Analyze cry using rule-based acoustic features.

        Fallback when HeAR is unavailable.

        Args:
            features: Extracted acoustic features

        Returns:
            Tuple of (asphyxia_risk, model_name)
        """
        # Rule-based asphyxia risk assessment
        # Based on medical literature on cry acoustics
        asphyxia_indicators = 0
        max_indicators = 5

        # High F0 (> 500 Hz) is associated with asphyxia
        if features["f0_mean"] > self.ASPHYXIA_F0_THRESHOLD:
            asphyxia_indicators += 1

        # High F0 variability
        if features["f0_std"] > 100:
            asphyxia_indicators += 1

        # Wide F0 range
        if features["f0_range"] > 300:
            asphyxia_indicators += 1

        # Low voiced ratio (fragmented cry)
        if features["voiced_ratio"] < 0.3:
            asphyxia_indicators += 1

        # High zero crossing rate (irregular)
        if features["zcr_mean"] > 0.15:
            asphyxia_indicators += 1

        asphyxia_risk = asphyxia_indicators / max_indicators
        return asphyxia_risk, "Acoustic Features"

    def _get_model_note(self, model_used: str) -> str:
        """Get descriptive note for the model used."""
        notes = {
            "HeAR + Classifier": "HAI-DEF HeAR embeddings with trained linear classifier",
            "HeAR (uncalibrated)": "HAI-DEF HeAR embeddings with heuristic scoring (no trained classifier)",
            "Acoustic Features": "Deterministic acoustic feature extraction (HeAR unavailable)",
        }
        return notes.get(model_used, model_used)

    def _classify_cry_type(self, features: Dict) -> str:
        """
        Classify cry type based on acoustic features.

        Categories based on donate-a-cry corpus:
        - hunger: Regular rhythm, moderate pitch
        - pain: High pitch, irregular
        - discomfort: Variable pitch, whimpering
        - tired: Low energy, fragmented
        - belly_pain: High pitch, straining patterns
        """
        f0_mean = features["f0_mean"]
        f0_std = features["f0_std"]
        rms_mean = features["rms_mean"]
        voiced_ratio = features["voiced_ratio"]

        # Simple rule-based classification
        if f0_mean > 500 and f0_std > 80:
            return "pain"
        elif f0_mean > 450 and rms_mean > 0.1:
            return "belly_pain"
        elif voiced_ratio < 0.4 and rms_mean < 0.05:
            return "tired"
        elif f0_std < 50 and voiced_ratio > 0.5:
            return "hunger"
        else:
            return "discomfort"

    def analyze_batch(
        self,
        audio_paths: List[Union[str, Path]],
    ) -> List[Dict]:
        """
        Analyze multiple cry audio files.

        Args:
            audio_paths: List of paths to audio files

        Returns:
            List of analysis results
        """
        results = []
        for path in audio_paths:
            try:
                result = self.analyze(path)
                result["file"] = str(path)
                results.append(result)
            except Exception as e:
                results.append({
                    "file": str(path),
                    "error": str(e),
                    "is_abnormal": None,
                })
        return results

    def get_spectrogram(
        self,
        audio_path: Union[str, Path],
        n_mels: int = 128,
    ) -> np.ndarray:
        """
        Generate mel spectrogram for visualization.

        Args:
            audio_path: Path to audio file
            n_mels: Number of mel bands

        Returns:
            Mel spectrogram array (dB scale)
        """
        audio, sr = self.load_audio(audio_path)

        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=n_mels,
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        return mel_spec_db


def test_analyzer():
    """Test the cry analyzer with sample audio files."""
    print("Testing Cry Analyzer...")

    analyzer = CryAnalyzer()

    # Check for available audio files
    data_dirs = [
        Path(__file__).parent.parent.parent / "data" / "raw" / "cryceleb" / "audio",
        Path(__file__).parent.parent.parent / "data" / "raw" / "donate-a-cry",
        Path(__file__).parent.parent.parent / "data" / "raw" / "infant-cry-dataset" / "cry",
    ]

    audio_files = []
    for data_dir in data_dirs:
        if data_dir.exists():
            audio_files.extend(list(data_dir.rglob("*.wav"))[:2])

    if audio_files:
        for audio_path in audio_files[:5]:
            print(f"\nAnalyzing: {audio_path.name}")
            try:
                result = analyzer.analyze(audio_path)
                print(f"  Abnormal cry: {result['is_abnormal']}")
                print(f"  Asphyxia risk: {result['asphyxia_risk']:.1%}")
                print(f"  Cry type: {result['cry_type']}")
                print(f"  Risk level: {result['risk_level']}")
                print(f"  F0 mean: {result['features']['f0_mean']} Hz")
            except Exception as e:
                print(f"  Error: {e}")
    else:
        print("No audio files found. Please download datasets first.")


if __name__ == "__main__":
    test_analyzer()
