# NEXUS Technical Implementation Guide

## Quick Reference

| Component | Status | Priority | Difficulty |
|-----------|--------|----------|------------|
| MedSigLIP Zero-Shot | Ready to implement | P0 | Easy |
| MedSigLIP Linear Probe | Ready to implement | P1 | Easy |
| HeAR Cry Analysis | Ready (need dataset) | P1 | Medium |
| MedGemma Prompting | Ready to implement | P1 | Easy |
| Mobile App | Ready to implement | P2 | Medium |
| Edge Deployment | Week 3 | P2 | Hard |

---

## Part 1: Environment Setup

### 1.1 Python Environment

```bash
# Create virtual environment
python -m venv nexus-env
source nexus-env/bin/activate  # On Windows: nexus-env\Scripts\activate

# Install core dependencies
pip install torch torchvision torchaudio
pip install transformers accelerate
pip install scikit-learn pandas numpy
pip install jupyter notebook
pip install pillow soundfile librosa

# Install HAI-DEF specific
pip install huggingface_hub
pip install sentencepiece

# Create requirements.txt
pip freeze > requirements.txt
```

### 1.2 Required Hardware

| Task | Minimum | Recommended |
|------|---------|-------------|
| Development | 16GB RAM, CPU | 32GB RAM, GPU |
| MedGemma 4B | 16GB RAM, T4 GPU | 24GB+ VRAM |
| MedSigLIP | 8GB RAM | 16GB RAM |
| HeAR | 4GB RAM | 8GB RAM |

### 1.3 Directory Structure

```bash
# Create project structure
mkdir -p nexus/{src/{ml,mobile},models,data,notebooks,submission}

cd nexus
```

---

## Part 2: MedSigLIP Implementation

### 2.1 Zero-Shot Anemia Detection

```python
# src/ml/anemia_detector.py

import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
import numpy as np
from typing import Dict, Tuple, Optional

class AnemiaDetector:
    """
    Zero-shot anemia detection using MedSigLIP.
    Analyzes conjunctiva images to detect anemia.
    """

    def __init__(self, model_name: str = "google/medsiglip-448"):
        """Initialize the detector with MedSigLIP model."""
        print(f"Loading MedSigLIP from {model_name}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()

        # Define classification labels
        self.labels = {
            "anemic": [
                "anemic pale conjunctiva",
                "conjunctiva showing pallor consistent with anemia",
                "pale inner eyelid indicating low hemoglobin"
            ],
            "healthy": [
                "healthy pink conjunctiva",
                "normal well-perfused conjunctiva",
                "pink inner eyelid with good blood supply"
            ]
        }
        print("Model loaded successfully!")

    def detect(self, image: Image.Image) -> Dict:
        """
        Detect anemia from a conjunctiva image.

        Args:
            image: PIL Image of the conjunctiva

        Returns:
            Dict with detection results
        """
        # Prepare all text prompts
        all_texts = self.labels["anemic"] + self.labels["healthy"]

        # Process inputs
        inputs = self.processor(
            images=image,
            text=all_texts,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image[0]  # Shape: (num_texts,)
            probs = torch.softmax(logits, dim=0).cpu().numpy()

        # Aggregate probabilities
        anemic_prob = np.mean(probs[:len(self.labels["anemic"])])
        healthy_prob = np.mean(probs[len(self.labels["anemic"]):])

        # Normalize
        total = anemic_prob + healthy_prob
        anemic_prob /= total
        healthy_prob /= total

        # Determine result
        is_anemic = anemic_prob > healthy_prob
        confidence = anemic_prob if is_anemic else healthy_prob

        return {
            "is_anemic": bool(is_anemic),
            "confidence": float(confidence),
            "anemia_probability": float(anemic_prob),
            "severity": self._estimate_severity(anemic_prob),
            "recommendation": self._get_recommendation(anemic_prob)
        }

    def _estimate_severity(self, prob: float) -> str:
        """Estimate anemia severity based on probability."""
        if prob > 0.85:
            return "severe"
        elif prob > 0.7:
            return "moderate"
        elif prob > 0.5:
            return "mild"
        return "normal"

    def _get_recommendation(self, prob: float) -> str:
        """Get clinical recommendation based on probability."""
        if prob > 0.85:
            return "URGENT: Severe anemia suspected. Immediate referral for blood test and treatment."
        elif prob > 0.7:
            return "WARNING: Moderate anemia likely. Start iron supplementation. Schedule follow-up."
        elif prob > 0.5:
            return "CAUTION: Mild anemia possible. Consider dietary counseling and iron supplementation."
        return "Normal conjunctiva appearance. Continue routine monitoring."


# Example usage
if __name__ == "__main__":
    detector = AnemiaDetector()

    # Test with sample image
    img = Image.open("test_conjunctiva.jpg")
    result = detector.detect(img)

    print(f"Is Anemic: {result['is_anemic']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Severity: {result['severity']}")
    print(f"Recommendation: {result['recommendation']}")
```

### 2.2 Zero-Shot Jaundice Detection

```python
# src/ml/jaundice_detector.py

import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
import numpy as np
from typing import Dict

class JaundiceDetector:
    """
    Zero-shot neonatal jaundice detection using MedSigLIP.
    Analyzes newborn skin images to detect jaundice.
    """

    def __init__(self, model_name: str = "google/medsiglip-448"):
        """Initialize the detector with MedSigLIP model."""
        print(f"Loading MedSigLIP from {model_name}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()

        # Define classification labels for different severities
        self.labels = {
            "severe_jaundice": [
                "severe neonatal jaundice with deep yellow skin",
                "newborn with severe hyperbilirubinemia",
                "critically jaundiced infant requiring immediate treatment"
            ],
            "moderate_jaundice": [
                "moderate neonatal jaundice",
                "newborn with yellow tinted skin",
                "infant showing signs of hyperbilirubinemia"
            ],
            "mild_jaundice": [
                "mild neonatal jaundice",
                "slightly yellow newborn skin",
                "early stage jaundice in newborn"
            ],
            "normal": [
                "healthy newborn skin color",
                "normal non-jaundiced infant skin",
                "newborn with pink healthy skin tone"
            ]
        }
        print("Model loaded successfully!")

    def detect(self, image: Image.Image) -> Dict:
        """
        Detect jaundice from a newborn skin image.

        Args:
            image: PIL Image of newborn skin (forehead or chest)

        Returns:
            Dict with detection results
        """
        # Prepare all text prompts
        all_texts = []
        for category in ["severe_jaundice", "moderate_jaundice", "mild_jaundice", "normal"]:
            all_texts.extend(self.labels[category])

        # Process inputs
        inputs = self.processor(
            images=image,
            text=all_texts,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = torch.softmax(logits, dim=0).cpu().numpy()

        # Aggregate probabilities by category
        idx = 0
        category_probs = {}
        for category in ["severe_jaundice", "moderate_jaundice", "mild_jaundice", "normal"]:
            n_labels = len(self.labels[category])
            category_probs[category] = np.mean(probs[idx:idx+n_labels])
            idx += n_labels

        # Normalize
        total = sum(category_probs.values())
        for cat in category_probs:
            category_probs[cat] /= total

        # Determine severity
        severity = max(category_probs, key=category_probs.get)
        is_jaundiced = severity != "normal"

        # Calculate overall jaundice probability
        jaundice_prob = 1 - category_probs["normal"]

        return {
            "is_jaundiced": bool(is_jaundiced),
            "severity": severity.replace("_", " ").title(),
            "confidence": float(category_probs[severity]),
            "jaundice_probability": float(jaundice_prob),
            "severity_scores": {k: float(v) for k, v in category_probs.items()},
            "recommendation": self._get_recommendation(severity, category_probs)
        }

    def _get_recommendation(self, severity: str, probs: Dict) -> str:
        """Get clinical recommendation based on severity."""
        if severity == "severe_jaundice":
            return "EMERGENCY: Severe jaundice detected. Immediate referral for phototherapy or exchange transfusion."
        elif severity == "moderate_jaundice":
            return "URGENT: Moderate jaundice detected. Refer for bilirubin test. Consider phototherapy."
        elif severity == "mild_jaundice":
            return "MONITOR: Mild jaundice detected. Encourage frequent feeding. Recheck in 24 hours."
        return "Normal skin color. Continue routine newborn care."


# Example usage
if __name__ == "__main__":
    detector = JaundiceDetector()

    # Test with sample image
    img = Image.open("test_newborn_skin.jpg")
    result = detector.detect(img)

    print(f"Is Jaundiced: {result['is_jaundiced']}")
    print(f"Severity: {result['severity']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Recommendation: {result['recommendation']}")
```

### 2.3 Linear Probe Training (For Higher Accuracy)

```python
# src/ml/train_linear_probe.py

import torch
from transformers import AutoModel, AutoProcessor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from typing import List, Tuple

class LinearProbeTrainer:
    """
    Train linear classifiers on MedSigLIP embeddings for higher accuracy.
    """

    def __init__(self, model_name: str = "google/medsiglip-448"):
        """Initialize with MedSigLIP model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()

    def extract_embeddings(self, image_paths: List[str]) -> np.ndarray:
        """
        Extract MedSigLIP embeddings for a list of images.

        Args:
            image_paths: List of paths to images

        Returns:
            numpy array of embeddings (n_samples, embedding_dim)
        """
        embeddings = []

        for path in tqdm(image_paths, desc="Extracting embeddings"):
            try:
                img = Image.open(path).convert("RGB")
                inputs = self.processor(images=img, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = self.model.get_image_features(**inputs)
                    embedding = outputs.cpu().numpy().flatten()
                    embeddings.append(embedding)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue

        return np.array(embeddings)

    def train_classifier(
        self,
        image_paths: List[str],
        labels: List[int],
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[LogisticRegression, dict]:
        """
        Train a linear probe classifier.

        Args:
            image_paths: List of paths to images
            labels: List of labels (0 or 1)
            test_size: Fraction for test split
            random_state: Random seed

        Returns:
            Trained classifier and metrics dict
        """
        print("Extracting embeddings...")
        embeddings = self.extract_embeddings(image_paths)

        print(f"Training on {len(embeddings)} samples...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )

        # Train logistic regression
        classifier = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=random_state
        )
        classifier.fit(X_train, y_train)

        # Evaluate
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        metrics = {
            "accuracy": accuracy,
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "n_train": len(X_train),
            "n_test": len(X_test)
        }

        print(f"\nAccuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))

        return classifier, metrics

    def save_classifier(self, classifier: LogisticRegression, path: str):
        """Save trained classifier to disk."""
        joblib.dump(classifier, path)
        print(f"Classifier saved to {path}")

    def load_classifier(self, path: str) -> LogisticRegression:
        """Load trained classifier from disk."""
        return joblib.load(path)


def train_anemia_classifier(data_dir: str, output_path: str):
    """
    Train anemia classifier on Eyes-Defy-Anemia dataset.

    Args:
        data_dir: Path to Eyes-Defy-Anemia dataset
        output_path: Path to save trained classifier
    """
    trainer = LinearProbeTrainer()

    # Load dataset (adjust paths based on actual dataset structure)
    image_paths = []
    labels = []

    # Example: assuming directory structure is:
    # data_dir/anemic/*.jpg
    # data_dir/healthy/*.jpg

    anemic_dir = os.path.join(data_dir, "anemic")
    healthy_dir = os.path.join(data_dir, "healthy")

    if os.path.exists(anemic_dir):
        for fname in os.listdir(anemic_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(anemic_dir, fname))
                labels.append(1)  # 1 = anemic

    if os.path.exists(healthy_dir):
        for fname in os.listdir(healthy_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(healthy_dir, fname))
                labels.append(0)  # 0 = healthy

    print(f"Found {len(image_paths)} images ({sum(labels)} anemic, {len(labels) - sum(labels)} healthy)")

    # Train classifier
    classifier, metrics = trainer.train_classifier(image_paths, labels)

    # Save
    trainer.save_classifier(classifier, output_path)

    return metrics


# Example usage
if __name__ == "__main__":
    # Train anemia classifier
    metrics = train_anemia_classifier(
        data_dir="data/eyes-defy-anemia",
        output_path="models/classifiers/anemia_classifier.joblib"
    )
```

---

## Part 3: HeAR Cry Analysis

### 3.1 HeAR Embedding Extraction

```python
# src/ml/hear_embeddings.py

import torch
import numpy as np
import soundfile as sf
import librosa
from typing import List, Optional
import os

class HeARProcessor:
    """
    Process audio files and extract HeAR embeddings.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize HeAR processor.

        Note: HeAR model requires special access. Contact Google at
        health_acoustic_representations@google.com
        """
        self.sample_rate = 16000  # HeAR requires 16kHz
        self.chunk_duration = 2.0  # HeAR processes 2-second chunks
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)

        # Load model if path provided
        # For now, we'll use a placeholder until HeAR access is granted
        self.model = None
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load HeAR model from path."""
        # Implementation depends on HeAR distribution format
        # This is a placeholder
        print(f"Loading HeAR model from {model_path}")
        # self.model = torch.load(model_path)

    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load and preprocess audio file for HeAR.

        Args:
            audio_path: Path to audio file

        Returns:
            numpy array of audio samples at 16kHz
        """
        # Load audio
        audio, sr = sf.read(audio_path)

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample to 16kHz if needed
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        return audio.astype(np.float32)

    def split_into_chunks(self, audio: np.ndarray) -> List[np.ndarray]:
        """
        Split audio into 2-second chunks for HeAR.

        Args:
            audio: Audio samples at 16kHz

        Returns:
            List of 2-second audio chunks
        """
        chunks = []

        for i in range(0, len(audio), self.chunk_samples):
            chunk = audio[i:i + self.chunk_samples]

            # Pad if needed
            if len(chunk) < self.chunk_samples:
                chunk = np.pad(chunk, (0, self.chunk_samples - len(chunk)))

            chunks.append(chunk)

        return chunks

    def extract_embeddings(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract HeAR embeddings from audio.

        Args:
            audio: Audio samples at 16kHz

        Returns:
            Aggregated embedding vector (512-dim)
        """
        chunks = self.split_into_chunks(audio)

        if self.model is None:
            # Fallback: use acoustic features
            print("Warning: HeAR model not loaded. Using acoustic features.")
            return self._extract_acoustic_features(audio)

        # Extract embeddings for each chunk
        embeddings = []
        for chunk in chunks:
            # HeAR expects (batch, samples) = (1, 32000)
            chunk_tensor = torch.tensor(chunk).unsqueeze(0)
            with torch.no_grad():
                embedding = self.model(chunk_tensor)
            embeddings.append(embedding.numpy())

        # Aggregate embeddings (mean pooling)
        aggregated = np.mean(embeddings, axis=0).flatten()

        return aggregated

    def _extract_acoustic_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Fallback: Extract acoustic features when HeAR is unavailable.

        This provides a 512-dim feature vector similar to HeAR embeddings.
        """
        features = []

        # MFCC features (13 coefficients x 2 = 26 stats)
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        features.extend([np.mean(spectral_centroid), np.std(spectral_centroid)])

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)
        features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])

        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
        features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.extend([np.mean(zcr), np.std(zcr)])

        # RMS energy
        rms = librosa.feature.rms(y=audio)
        features.extend([np.mean(rms), np.std(rms)])

        # Pitch features
        pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        if len(pitch_values) > 0:
            features.extend([np.mean(pitch_values), np.std(pitch_values),
                           np.min(pitch_values), np.max(pitch_values)])
        else:
            features.extend([0, 0, 0, 0])

        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))

        # Pad or truncate to 512 dimensions
        features = np.array(features)
        if len(features) < 512:
            features = np.pad(features, (0, 512 - len(features)))
        else:
            features = features[:512]

        return features


# Example usage
if __name__ == "__main__":
    processor = HeARProcessor()

    # Load and process audio
    audio = processor.load_audio("test_cry.wav")
    embeddings = processor.extract_embeddings(audio)

    print(f"Embedding shape: {embeddings.shape}")
```

### 3.2 Cry Classifier

```python
# src/ml/cry_analyzer.py

import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
from typing import Dict, Optional
from .hear_embeddings import HeARProcessor

class CryAnalyzer:
    """
    Analyze infant cries for signs of distress or birth asphyxia.
    """

    def __init__(
        self,
        classifier_path: Optional[str] = None,
        hear_model_path: Optional[str] = None
    ):
        """
        Initialize cry analyzer.

        Args:
            classifier_path: Path to trained classifier
            hear_model_path: Path to HeAR model
        """
        self.processor = HeARProcessor(hear_model_path)
        self.classifier = None

        if classifier_path:
            self.classifier = joblib.load(classifier_path)

    def analyze(self, audio_path: str) -> Dict:
        """
        Analyze infant cry audio.

        Args:
            audio_path: Path to audio file

        Returns:
            Analysis results dict
        """
        # Load and extract embeddings
        audio = self.processor.load_audio(audio_path)
        embeddings = self.processor.extract_embeddings(audio)

        # Classify if model available
        if self.classifier is not None:
            prediction = self.classifier.predict(embeddings.reshape(1, -1))[0]
            probabilities = self.classifier.predict_proba(embeddings.reshape(1, -1))[0]
            is_abnormal = bool(prediction == 1)
            confidence = float(probabilities[prediction])
            abnormal_prob = float(probabilities[1])
        else:
            # Fallback: rule-based analysis
            is_abnormal, abnormal_prob = self._rule_based_analysis(audio)
            confidence = 0.7  # Lower confidence for rule-based

        return {
            "is_abnormal": is_abnormal,
            "confidence": confidence,
            "abnormal_probability": abnormal_prob,
            "severity": self._estimate_severity(abnormal_prob),
            "recommendation": self._get_recommendation(abnormal_prob),
            "audio_duration": len(audio) / self.processor.sample_rate
        }

    def _rule_based_analysis(self, audio: np.ndarray) -> tuple:
        """
        Fallback rule-based analysis when classifier unavailable.

        Based on research: abnormal cries tend to have:
        - Higher fundamental frequency
        - More irregular patterns
        - Shorter cry episodes
        """
        import librosa

        # Extract features
        pitches, magnitudes = librosa.piptrack(y=audio, sr=self.processor.sample_rate)
        valid_pitches = pitches[magnitudes > np.median(magnitudes)]

        if len(valid_pitches) == 0:
            return False, 0.3  # No valid pitch detected

        mean_pitch = np.mean(valid_pitches)
        pitch_std = np.std(valid_pitches)

        # Abnormal cry indicators (based on research)
        # Normal infant cry: 250-600 Hz
        # Asphyxia cry: often higher (>600 Hz) or more variable

        score = 0.0

        # High pitch indicator
        if mean_pitch > 600:
            score += 0.3
        elif mean_pitch > 500:
            score += 0.1

        # High variability indicator
        if pitch_std > 150:
            score += 0.3
        elif pitch_std > 100:
            score += 0.1

        # Short cry segments indicator
        rms = librosa.feature.rms(y=audio)[0]
        cry_segments = np.sum(rms > np.mean(rms))
        segment_ratio = cry_segments / len(rms)

        if segment_ratio < 0.3:
            score += 0.2

        # Normalize score
        abnormal_prob = min(score, 1.0)
        is_abnormal = abnormal_prob > 0.5

        return is_abnormal, abnormal_prob

    def _estimate_severity(self, prob: float) -> str:
        """Estimate severity based on probability."""
        if prob > 0.8:
            return "critical"
        elif prob > 0.6:
            return "concerning"
        elif prob > 0.4:
            return "mild concern"
        return "normal"

    def _get_recommendation(self, prob: float) -> str:
        """Get clinical recommendation."""
        if prob > 0.8:
            return "EMERGENCY: Abnormal cry pattern detected. Signs consistent with birth asphyxia. Immediate neonatal resuscitation required."
        elif prob > 0.6:
            return "URGENT: Concerning cry pattern. Close monitoring required. Consider immediate medical evaluation."
        elif prob > 0.4:
            return "MONITOR: Slightly unusual cry pattern. Continue close observation. Recheck if symptoms persist."
        return "Normal cry pattern. Continue routine newborn care."


def train_cry_classifier(data_dir: str, output_path: str):
    """
    Train cry classifier on Baby Chillanto dataset.

    Args:
        data_dir: Path to Baby Chillanto dataset
        output_path: Path to save trained classifier
    """
    import os
    from tqdm import tqdm

    processor = HeARProcessor()

    embeddings = []
    labels = []

    # Load asphyxia samples
    asphyxia_dir = os.path.join(data_dir, "asphyxia")
    if os.path.exists(asphyxia_dir):
        for fname in tqdm(os.listdir(asphyxia_dir), desc="Loading asphyxia samples"):
            if fname.endswith(".wav"):
                try:
                    audio = processor.load_audio(os.path.join(asphyxia_dir, fname))
                    emb = processor.extract_embeddings(audio)
                    embeddings.append(emb)
                    labels.append(1)  # 1 = asphyxia/abnormal
                except Exception as e:
                    print(f"Error processing {fname}: {e}")

    # Load normal samples
    normal_dir = os.path.join(data_dir, "normal")
    if os.path.exists(normal_dir):
        for fname in tqdm(os.listdir(normal_dir), desc="Loading normal samples"):
            if fname.endswith(".wav"):
                try:
                    audio = processor.load_audio(os.path.join(normal_dir, fname))
                    emb = processor.extract_embeddings(audio)
                    embeddings.append(emb)
                    labels.append(0)  # 0 = normal
                except Exception as e:
                    print(f"Error processing {fname}: {e}")

    print(f"Loaded {len(embeddings)} samples ({sum(labels)} asphyxia, {len(labels) - sum(labels)} normal)")

    # Train classifier
    X = np.array(embeddings)
    y = np.array(labels)

    classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
    classifier.fit(X, y)

    # Save
    joblib.dump(classifier, output_path)
    print(f"Classifier saved to {output_path}")

    return classifier


# Example usage
if __name__ == "__main__":
    analyzer = CryAnalyzer()

    # Test with sample audio
    result = analyzer.analyze("test_cry.wav")

    print(f"Is Abnormal: {result['is_abnormal']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Severity: {result['severity']}")
    print(f"Recommendation: {result['recommendation']}")
```

---

## Part 4: MedGemma Clinical Synthesis

### 4.1 Clinical Synthesizer

```python
# src/ml/clinical_synthesizer.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, Optional
from datetime import datetime

class ClinicalSynthesizer:
    """
    Synthesize clinical findings into recommendations using MedGemma.
    """

    def __init__(self, model_name: str = "google/medgemma-4b-it"):
        """
        Initialize with MedGemma model.

        Note: MedGemma requires accepting license on HuggingFace.
        """
        print(f"Loading MedGemma from {model_name}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        print("Model loaded successfully!")

    def synthesize(self, findings: Dict) -> Dict:
        """
        Synthesize all findings into clinical recommendations.

        Args:
            findings: Dict with anemia, jaundice, cry results

        Returns:
            Clinical synthesis with recommendations
        """
        prompt = self._build_prompt(findings)
        response = self._generate(prompt)

        return {
            "summary": response,
            "findings": findings,
            "generated_at": datetime.now().isoformat(),
            "model": "medgemma-4b"
        }

    def _build_prompt(self, findings: Dict) -> str:
        """Build prompt for MedGemma."""
        # Extract finding details safely
        anemia = findings.get("anemia", {})
        jaundice = findings.get("jaundice", {})
        cry = findings.get("cry", {})
        symptoms = findings.get("symptoms", "None reported")
        patient_info = findings.get("patient_info", {})

        prompt = f"""<start_of_turn>user
You are an AI assistant helping community health workers provide maternal and neonatal care. Analyze the following assessment findings and provide clear, actionable recommendations.

PATIENT INFORMATION:
- Type: {patient_info.get('type', 'Newborn')}
- Age: {patient_info.get('age', 'Unknown')}
- Location: {patient_info.get('location', 'Rural health post')}

ASSESSMENT FINDINGS:

1. ANEMIA SCREENING (Conjunctiva Analysis):
   - Result: {"Anemia detected" if anemia.get("is_anemic") else "No anemia detected"}
   - Confidence: {anemia.get("confidence", "N/A"):.0%}
   - Severity: {anemia.get("severity", "N/A")}

2. JAUNDICE SCREENING (Skin Analysis):
   - Result: {"Jaundice detected" if jaundice.get("is_jaundiced") else "No jaundice detected"}
   - Severity: {jaundice.get("severity", "N/A")}
   - Confidence: {jaundice.get("confidence", "N/A"):.0%}

3. CRY ANALYSIS (Audio):
   - Result: {"Abnormal cry pattern" if cry.get("is_abnormal") else "Normal cry pattern"}
   - Confidence: {cry.get("confidence", "N/A"):.0%}
   - Severity: {cry.get("severity", "N/A")}

4. REPORTED SYMPTOMS:
   {symptoms}

Based on these findings, provide:

1. **ASSESSMENT SUMMARY** (2-3 sentences summarizing the key concerns)

2. **SEVERITY LEVEL** (Choose one):
   - GREEN: Routine care, no immediate concerns
   - YELLOW: Close monitoring needed, schedule follow-up
   - RED: Urgent referral required

3. **IMMEDIATE ACTIONS** (What should the CHW do right now?)

4. **REFERRAL RECOMMENDATION** (Yes/No, and urgency if yes)

5. **FOLLOW-UP PLAN** (When to recheck, what to monitor)

Use simple, clear language that a community health worker with basic training can understand. Be specific and actionable.
<end_of_turn>
<start_of_turn>model
"""
        return prompt

    def _generate(self, prompt: str, max_new_tokens: int = 500) -> str:
        """Generate response from MedGemma."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the model's response
        if "<start_of_turn>model" in response:
            response = response.split("<start_of_turn>model")[-1].strip()

        return response

    def quick_triage(self, findings: Dict) -> Dict:
        """
        Quick triage without full synthesis - faster for mobile.

        Returns severity level and key action only.
        """
        # Rule-based quick triage
        severity_scores = []

        if findings.get("anemia", {}).get("is_anemic"):
            if findings["anemia"].get("severity") == "severe":
                severity_scores.append(3)
            elif findings["anemia"].get("severity") == "moderate":
                severity_scores.append(2)
            else:
                severity_scores.append(1)

        if findings.get("jaundice", {}).get("is_jaundiced"):
            sev = findings["jaundice"].get("severity", "").lower()
            if "severe" in sev:
                severity_scores.append(3)
            elif "moderate" in sev:
                severity_scores.append(2)
            else:
                severity_scores.append(1)

        if findings.get("cry", {}).get("is_abnormal"):
            if findings["cry"].get("severity") == "critical":
                severity_scores.append(3)
            elif findings["cry"].get("severity") == "concerning":
                severity_scores.append(2)
            else:
                severity_scores.append(1)

        # Determine overall severity
        if not severity_scores:
            level = "GREEN"
            action = "Continue routine care. No immediate concerns detected."
        elif max(severity_scores) >= 3:
            level = "RED"
            action = "URGENT: Immediate referral required. Transport to health facility."
        elif max(severity_scores) >= 2:
            level = "YELLOW"
            action = "Close monitoring needed. Schedule follow-up within 24 hours."
        else:
            level = "GREEN"
            action = "Minor concerns detected. Continue monitoring, counsel mother."

        return {
            "severity_level": level,
            "action": action,
            "requires_referral": level == "RED",
            "follow_up_hours": 24 if level == "YELLOW" else 72
        }


# Example usage
if __name__ == "__main__":
    synthesizer = ClinicalSynthesizer()

    # Example findings
    findings = {
        "anemia": {
            "is_anemic": True,
            "confidence": 0.85,
            "severity": "moderate"
        },
        "jaundice": {
            "is_jaundiced": True,
            "confidence": 0.72,
            "severity": "Mild Jaundice"
        },
        "cry": {
            "is_abnormal": False,
            "confidence": 0.90,
            "severity": "normal"
        },
        "symptoms": "Baby feeding well, slightly fussy",
        "patient_info": {
            "type": "Newborn",
            "age": "3 days",
            "location": "Village health post"
        }
    }

    # Full synthesis
    result = synthesizer.synthesize(findings)
    print("Full Synthesis:")
    print(result["summary"])

    # Quick triage
    triage = synthesizer.quick_triage(findings)
    print(f"\nQuick Triage: {triage['severity_level']}")
    print(f"Action: {triage['action']}")
```

---

## Part 5: Integration & Main Pipeline

### 5.1 NEXUS Pipeline

```python
# src/ml/nexus_pipeline.py

from typing import Dict, Optional
from PIL import Image
import json
from datetime import datetime

from .anemia_detector import AnemiaDetector
from .jaundice_detector import JaundiceDetector
from .cry_analyzer import CryAnalyzer
from .clinical_synthesizer import ClinicalSynthesizer

class NEXUSPipeline:
    """
    Main NEXUS pipeline integrating all components.
    """

    def __init__(
        self,
        anemia_classifier_path: Optional[str] = None,
        jaundice_classifier_path: Optional[str] = None,
        cry_classifier_path: Optional[str] = None,
        use_medgemma: bool = True
    ):
        """
        Initialize NEXUS pipeline.

        Args:
            anemia_classifier_path: Path to trained anemia classifier (optional)
            jaundice_classifier_path: Path to trained jaundice classifier (optional)
            cry_classifier_path: Path to trained cry classifier (optional)
            use_medgemma: Whether to use MedGemma for synthesis
        """
        print("Initializing NEXUS Pipeline...")

        self.anemia_detector = AnemiaDetector()
        self.jaundice_detector = JaundiceDetector()
        self.cry_analyzer = CryAnalyzer(classifier_path=cry_classifier_path)

        if use_medgemma:
            self.synthesizer = ClinicalSynthesizer()
        else:
            self.synthesizer = None

        print("NEXUS Pipeline ready!")

    def assess_pregnant_woman(
        self,
        conjunctiva_image: Image.Image,
        symptoms: str = ""
    ) -> Dict:
        """
        Assess pregnant woman for anemia and other conditions.

        Args:
            conjunctiva_image: PIL Image of conjunctiva
            symptoms: Reported symptoms string

        Returns:
            Assessment results
        """
        findings = {
            "assessment_type": "pregnant_woman",
            "timestamp": datetime.now().isoformat()
        }

        # Anemia detection
        findings["anemia"] = self.anemia_detector.detect(conjunctiva_image)
        findings["symptoms"] = symptoms
        findings["patient_info"] = {"type": "Pregnant Woman"}

        # Synthesize if available
        if self.synthesizer:
            synthesis = self.synthesizer.synthesize(findings)
            findings["clinical_synthesis"] = synthesis["summary"]
        else:
            triage = self._quick_triage(findings)
            findings["triage"] = triage

        return findings

    def assess_newborn(
        self,
        skin_image: Optional[Image.Image] = None,
        cry_audio_path: Optional[str] = None,
        symptoms: str = ""
    ) -> Dict:
        """
        Comprehensive newborn assessment.

        Args:
            skin_image: PIL Image of newborn skin (for jaundice)
            cry_audio_path: Path to cry audio file
            symptoms: Reported symptoms string

        Returns:
            Assessment results
        """
        findings = {
            "assessment_type": "newborn",
            "timestamp": datetime.now().isoformat(),
            "patient_info": {"type": "Newborn"}
        }

        # Jaundice detection
        if skin_image:
            findings["jaundice"] = self.jaundice_detector.detect(skin_image)

        # Cry analysis
        if cry_audio_path:
            findings["cry"] = self.cry_analyzer.analyze(cry_audio_path)

        findings["symptoms"] = symptoms

        # Synthesize if available
        if self.synthesizer:
            synthesis = self.synthesizer.synthesize(findings)
            findings["clinical_synthesis"] = synthesis["summary"]
        else:
            triage = self._quick_triage(findings)
            findings["triage"] = triage

        return findings

    def full_assessment(
        self,
        conjunctiva_image: Optional[Image.Image] = None,
        skin_image: Optional[Image.Image] = None,
        cry_audio_path: Optional[str] = None,
        symptoms: str = "",
        patient_info: Optional[Dict] = None
    ) -> Dict:
        """
        Full multi-modal assessment.

        Args:
            conjunctiva_image: PIL Image of conjunctiva
            skin_image: PIL Image of skin
            cry_audio_path: Path to cry audio
            symptoms: Reported symptoms
            patient_info: Patient information dict

        Returns:
            Complete assessment results
        """
        findings = {
            "assessment_type": "full",
            "timestamp": datetime.now().isoformat(),
            "patient_info": patient_info or {}
        }

        # Run all available assessments
        if conjunctiva_image:
            findings["anemia"] = self.anemia_detector.detect(conjunctiva_image)

        if skin_image:
            findings["jaundice"] = self.jaundice_detector.detect(skin_image)

        if cry_audio_path:
            findings["cry"] = self.cry_analyzer.analyze(cry_audio_path)

        findings["symptoms"] = symptoms

        # Synthesize
        if self.synthesizer:
            synthesis = self.synthesizer.synthesize(findings)
            findings["clinical_synthesis"] = synthesis["summary"]

        # Always include quick triage
        findings["triage"] = self._quick_triage(findings)

        return findings

    def _quick_triage(self, findings: Dict) -> Dict:
        """Quick rule-based triage."""
        if self.synthesizer:
            return self.synthesizer.quick_triage(findings)

        # Fallback rule-based
        is_urgent = False
        concerns = []

        if findings.get("anemia", {}).get("is_anemic"):
            concerns.append("anemia")
            if findings["anemia"].get("severity") == "severe":
                is_urgent = True

        if findings.get("jaundice", {}).get("is_jaundiced"):
            concerns.append("jaundice")
            if "severe" in findings["jaundice"].get("severity", "").lower():
                is_urgent = True

        if findings.get("cry", {}).get("is_abnormal"):
            concerns.append("abnormal cry")
            if findings["cry"].get("severity") == "critical":
                is_urgent = True

        if is_urgent:
            level = "RED"
            action = "URGENT: Immediate referral required."
        elif concerns:
            level = "YELLOW"
            action = f"Monitor closely for: {', '.join(concerns)}. Follow up in 24 hours."
        else:
            level = "GREEN"
            action = "No immediate concerns. Continue routine care."

        return {
            "severity_level": level,
            "action": action,
            "concerns": concerns,
            "requires_referral": is_urgent
        }

    def export_results(self, findings: Dict, output_path: str):
        """Export assessment results to JSON."""
        with open(output_path, 'w') as f:
            json.dump(findings, f, indent=2, default=str)
        print(f"Results exported to {output_path}")


# Example usage and demo
if __name__ == "__main__":
    # Initialize pipeline (without MedGemma for quick testing)
    pipeline = NEXUSPipeline(use_medgemma=False)

    # Demo: Newborn assessment
    print("\n=== DEMO: Newborn Assessment ===\n")

    # Load test images (replace with actual paths)
    try:
        skin_img = Image.open("test_data/newborn_skin.jpg")
    except:
        skin_img = None
        print("No skin image found, skipping jaundice detection")

    # Run assessment
    results = pipeline.assess_newborn(
        skin_image=skin_img,
        cry_audio_path="test_data/cry.wav" if False else None,  # Set to True if file exists
        symptoms="Baby is feeding well but appears slightly yellow"
    )

    print("Assessment Results:")
    print(json.dumps(results, indent=2, default=str))
```

---

## Part 6: Testing & Validation

### 6.1 Unit Tests

```python
# tests/test_detectors.py

import pytest
from PIL import Image
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.anemia_detector import AnemiaDetector
from src.ml.jaundice_detector import JaundiceDetector


class TestAnemiaDetector:
    """Test anemia detection functionality."""

    @pytest.fixture
    def detector(self):
        """Initialize detector."""
        return AnemiaDetector()

    def test_initialization(self, detector):
        """Test that detector initializes correctly."""
        assert detector.model is not None
        assert detector.processor is not None

    def test_detect_returns_expected_keys(self, detector):
        """Test that detect returns all expected keys."""
        # Create dummy image
        img = Image.new('RGB', (224, 224), color='red')

        result = detector.detect(img)

        assert "is_anemic" in result
        assert "confidence" in result
        assert "severity" in result
        assert "recommendation" in result

    def test_confidence_in_valid_range(self, detector):
        """Test that confidence is between 0 and 1."""
        img = Image.new('RGB', (224, 224), color='pink')

        result = detector.detect(img)

        assert 0 <= result["confidence"] <= 1

    def test_severity_is_valid(self, detector):
        """Test that severity is one of expected values."""
        img = Image.new('RGB', (224, 224), color='red')

        result = detector.detect(img)

        assert result["severity"] in ["normal", "mild", "moderate", "severe"]


class TestJaundiceDetector:
    """Test jaundice detection functionality."""

    @pytest.fixture
    def detector(self):
        """Initialize detector."""
        return JaundiceDetector()

    def test_initialization(self, detector):
        """Test that detector initializes correctly."""
        assert detector.model is not None
        assert detector.processor is not None

    def test_detect_returns_expected_keys(self, detector):
        """Test that detect returns all expected keys."""
        img = Image.new('RGB', (224, 224), color='yellow')

        result = detector.detect(img)

        assert "is_jaundiced" in result
        assert "severity" in result
        assert "confidence" in result
        assert "recommendation" in result

    def test_yellow_image_detected_as_jaundice(self, detector):
        """Test that yellow image is more likely to be jaundiced."""
        yellow_img = Image.new('RGB', (224, 224), color='yellow')
        pink_img = Image.new('RGB', (224, 224), color='pink')

        yellow_result = detector.detect(yellow_img)
        pink_result = detector.detect(pink_img)

        # Yellow should have higher jaundice probability
        assert yellow_result["jaundice_probability"] > pink_result["jaundice_probability"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### 6.2 Run Validation Script

```python
# scripts/validate_models.py

"""
Validate all NEXUS models are working correctly.
Run this before deployment.
"""

import sys
import os
from PIL import Image
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def validate_medsigLIP():
    """Validate MedSigLIP is working."""
    print("\n=== Validating MedSigLIP ===")

    try:
        from src.ml.anemia_detector import AnemiaDetector
        from src.ml.jaundice_detector import JaundiceDetector

        # Test anemia detector
        print("Testing AnemiaDetector...")
        anemia = AnemiaDetector()
        test_img = Image.new('RGB', (448, 448), color='pink')
        result = anemia.detect(test_img)
        print(f"  Result: {result['severity']}, confidence: {result['confidence']:.2%}")
        print("  AnemiaDetector: OK")

        # Test jaundice detector
        print("Testing JaundiceDetector...")
        jaundice = JaundiceDetector()
        result = jaundice.detect(test_img)
        print(f"  Result: {result['severity']}, confidence: {result['confidence']:.2%}")
        print("  JaundiceDetector: OK")

        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def validate_hear():
    """Validate HeAR processing is working."""
    print("\n=== Validating HeAR Processing ===")

    try:
        from src.ml.cry_analyzer import CryAnalyzer

        print("Testing CryAnalyzer...")
        analyzer = CryAnalyzer()

        # Create test audio (sine wave)
        import soundfile as sf
        duration = 3.0  # seconds
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

        # Save temp file
        test_path = "/tmp/test_audio.wav"
        sf.write(test_path, audio, sample_rate)

        # Analyze
        result = analyzer.analyze(test_path)
        print(f"  Result: abnormal={result['is_abnormal']}, confidence: {result['confidence']:.2%}")
        print("  CryAnalyzer: OK")

        # Cleanup
        os.remove(test_path)

        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def validate_pipeline():
    """Validate full pipeline is working."""
    print("\n=== Validating NEXUS Pipeline ===")

    try:
        from src.ml.nexus_pipeline import NEXUSPipeline

        print("Initializing pipeline (without MedGemma)...")
        pipeline = NEXUSPipeline(use_medgemma=False)

        # Test full assessment
        test_img = Image.new('RGB', (448, 448), color='peachpuff')
        result = pipeline.full_assessment(
            conjunctiva_image=test_img,
            skin_image=test_img,
            symptoms="Test symptoms"
        )

        print(f"  Triage level: {result['triage']['severity_level']}")
        print("  Pipeline: OK")

        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    """Run all validations."""
    print("=" * 50)
    print("NEXUS Model Validation")
    print("=" * 50)

    results = {
        "MedSigLIP": validate_medsigLIP(),
        "HeAR": validate_hear(),
        "Pipeline": validate_pipeline()
    }

    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)

    all_passed = True
    for component, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {component}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll validations passed!")
        return 0
    else:
        print("\nSome validations failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

---

## Next Steps

1. **Run environment setup** (Part 1)
2. **Test MedSigLIP zero-shot** on real images (Part 2.1, 2.2)
3. **Download datasets** and train linear probes (Part 2.3)
4. **Set up HeAR** (Part 3) - request access if needed
5. **Test MedGemma** prompting (Part 4)
6. **Run validation script** to ensure everything works

---

**Document Version**: 1.0
**Created**: January 14, 2026
**For**: NEXUS - MedGemma Impact Challenge
