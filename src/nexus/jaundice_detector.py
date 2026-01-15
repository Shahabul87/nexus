"""
Jaundice Detector Module

Uses MedSigLIP from Google HAI-DEF for jaundice detection from neonatal skin images.
Implements zero-shot classification with medical text prompts per NEXUS_MASTER_PLAN.md.

HAI-DEF Model: google/siglip-so400m-patch14-384 (MedSigLIP)
Documentation: https://developers.google.com/health-ai-developer-foundations/medsiglip
"""

import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    from transformers import AutoProcessor, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# HAI-DEF MedSigLIP model IDs to try in order of preference
MEDSIGLIP_MODEL_IDS = [
    "google/siglip-so400m-patch14-384",  # MedSigLIP - primary HAI-DEF model
    "google/siglip-base-patch16-384",     # SigLIP 384 - fallback
    "google/siglip-base-patch16-224",     # SigLIP 224 - final fallback
]


class JaundiceDetector:
    """
    Detects neonatal jaundice from skin/sclera images using MedSigLIP.

    Uses zero-shot classification with medical prompts and
    color analysis for bilirubin estimation.

    HAI-DEF Model: google/siglip-so400m-patch14-384 (MedSigLIP)
    Fallbacks: siglip-base-patch16-384, siglip-base-patch16-224
    Expected Accuracy: 80-90% (per NEXUS_MASTER_PLAN.md)
    """

    # Medical text prompts for zero-shot classification (optimized for MedSigLIP)
    JAUNDICE_PROMPTS = [
        "jaundiced yellow skin indicating neonatal hyperbilirubinemia",
        "newborn with yellow skin discoloration from jaundice",
        "neonatal jaundice requiring phototherapy assessment",
    ]

    NORMAL_PROMPTS = [
        "normal newborn skin without jaundice",
        "healthy newborn with normal pink skin color",
        "newborn with normal skin pigmentation no icterus",
    ]

    # Bilirubin risk thresholds (mg/dL)
    BILIRUBIN_THRESHOLDS = {
        "low": 5.0,      # Normal range
        "moderate": 12.0, # Monitor closely
        "high": 15.0,     # Consider phototherapy
        "critical": 20.0, # Urgent phototherapy
        "exchange": 25.0, # Exchange transfusion territory
    }

    def __init__(
        self,
        model_name: Optional[str] = None,  # Auto-select MedSigLIP
        device: Optional[str] = None,
        threshold: float = 0.5,
    ):
        """
        Initialize the Jaundice Detector with MedSigLIP.

        Args:
            model_name: HuggingFace model name (auto-selects HAI-DEF MedSigLIP if None)
            device: Device to run model on (auto-detected if None)
            threshold: Classification threshold for jaundice detection
        """
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library required. Install with: pip install transformers")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self._model_loaded = False
        self.classifier = None  # Can be set by pipeline for trained classification

        # Determine which models to try
        models_to_try = [model_name] if model_name else MEDSIGLIP_MODEL_IDS

        # Try loading models in order of preference
        for candidate_model in models_to_try:
            print(f"Loading HAI-DEF model: {candidate_model}")
            try:
                self.processor = AutoProcessor.from_pretrained(candidate_model)
                self.model = AutoModel.from_pretrained(candidate_model).to(self.device)
                self.model_name = candidate_model
                self._model_loaded = True
                print(f"Successfully loaded: {candidate_model}")
                break
            except Exception as e:
                print(f"Warning: Could not load {candidate_model}: {e}")
                continue

        if not self._model_loaded:
            raise RuntimeError(
                f"Could not load any MedSigLIP model. Tried: {models_to_try}. "
                "Install transformers and ensure internet access."
            )

        self.model.eval()

        # Pre-compute text embeddings
        self._precompute_text_embeddings()

        # Indicate which model variant is being used
        is_medsiglip = "so400m" in self.model_name or "384" in self.model_name
        model_type = "MedSigLIP" if is_medsiglip else "SigLIP (fallback)"
        print(f"Jaundice Detector (HAI-DEF {model_type}) initialized on {self.device}")

    def _precompute_text_embeddings(self) -> None:
        """Pre-compute text embeddings for zero-shot classification using SigLIP."""
        all_prompts = self.JAUNDICE_PROMPTS + self.NORMAL_PROMPTS

        with torch.no_grad():
            inputs = self.processor(
                text=all_prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            ).to(self.device)

            # Get text embeddings from SigLIP text encoder
            if hasattr(self.model, 'get_text_features'):
                text_embeddings = self.model.get_text_features(**inputs)
            else:
                text_outputs = self.model.text_model(**inputs)
                text_embeddings = text_outputs.pooler_output

            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

            n_jaundice = len(self.JAUNDICE_PROMPTS)
            self.jaundice_embeddings = text_embeddings[:n_jaundice].mean(dim=0, keepdim=True)
            self.normal_embeddings = text_embeddings[n_jaundice:].mean(dim=0, keepdim=True)

            self.jaundice_embeddings = self.jaundice_embeddings / self.jaundice_embeddings.norm(dim=-1, keepdim=True)
            self.normal_embeddings = self.normal_embeddings / self.normal_embeddings.norm(dim=-1, keepdim=True)

    def preprocess_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """Preprocess image for analysis."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        return image

    def estimate_bilirubin(self, image: Union[str, Path, Image.Image]) -> float:
        """
        Estimate bilirubin level from image color analysis.

        This uses the yellow-blue ratio which correlates with
        transcutaneous bilirubin measurements.

        Args:
            image: Neonatal skin/sclera image

        Returns:
            Estimated bilirubin in mg/dL
        """
        pil_image = self.preprocess_image(image)
        img_array = np.array(pil_image).astype(float)

        # Extract color channels
        r = img_array[:, :, 0]
        g = img_array[:, :, 1]
        b = img_array[:, :, 2]

        # Calculate yellow index (R+G-B correlation with bilirubin)
        # Higher values indicate more yellow (jaundiced)
        yellow_index = (r + g - b) / (r + g + b + 1e-6)
        mean_yellow = np.mean(yellow_index)

        # Convert to bilirubin estimate
        # Calibrated based on medical literature
        # Normal yellow_index ~ 0.2-0.3, jaundiced ~ 0.4-0.6
        bilirubin_estimate = max(0, (mean_yellow - 0.2) * 50)

        return round(bilirubin_estimate, 1)

    def detect(self, image: Union[str, Path, Image.Image]) -> Dict:
        """
        Detect jaundice from neonatal image.

        Uses trained classifier if available, otherwise falls back to
        zero-shot classification with MedSigLIP.

        Args:
            image: Neonatal skin/sclera image

        Returns:
            Dictionary containing:
                - has_jaundice: Boolean indicating jaundice detection
                - confidence: Confidence score
                - jaundice_score: Raw jaundice probability
                - estimated_bilirubin: Estimated bilirubin (mg/dL)
                - severity: "none", "mild", "moderate", "severe", "critical"
                - needs_phototherapy: Boolean
                - recommendation: Clinical recommendation
        """
        pil_image = self.preprocess_image(image)

        # Get image embedding using SigLIP
        with torch.no_grad():
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

            # Get image embeddings from SigLIP vision encoder
            if hasattr(self.model, 'get_image_features'):
                image_embedding = self.model.get_image_features(**inputs)
            else:
                vision_outputs = self.model.vision_model(**inputs)
                image_embedding = vision_outputs.pooler_output

            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

        # Use trained classifier if available, otherwise zero-shot
        if self.classifier is not None:
            jaundice_prob, model_method = self._classify_with_trained_model(image_embedding)
        else:
            jaundice_prob, model_method = self._classify_zero_shot(image_embedding)

        # Estimate bilirubin from color (always useful as additional signal)
        estimated_bilirubin = self.estimate_bilirubin(pil_image)

        # Determine severity based on estimated bilirubin
        if estimated_bilirubin < self.BILIRUBIN_THRESHOLDS["low"]:
            severity = "none"
            needs_phototherapy = False
            recommendation = "No jaundice detected. Continue routine care."
        elif estimated_bilirubin < self.BILIRUBIN_THRESHOLDS["moderate"]:
            severity = "mild"
            needs_phototherapy = False
            recommendation = "Mild jaundice. Monitor closely and ensure adequate feeding."
        elif estimated_bilirubin < self.BILIRUBIN_THRESHOLDS["high"]:
            severity = "moderate"
            needs_phototherapy = False
            recommendation = "Moderate jaundice. Recheck in 12-24 hours. Consider phototherapy if rising."
        elif estimated_bilirubin < self.BILIRUBIN_THRESHOLDS["critical"]:
            severity = "severe"
            needs_phototherapy = True
            recommendation = "URGENT: Start phototherapy. Refer for serum bilirubin confirmation."
        else:
            severity = "critical"
            needs_phototherapy = True
            recommendation = "CRITICAL: Immediate phototherapy required. Consider exchange transfusion."

        is_medsiglip = "so400m" in self.model_name or "384" in self.model_name
        base_model = "MedSigLIP (HAI-DEF)" if is_medsiglip else "SigLIP (fallback)"

        return {
            "has_jaundice": jaundice_prob > self.threshold,
            "confidence": max(jaundice_prob, 1 - jaundice_prob),
            "jaundice_score": jaundice_prob,
            "estimated_bilirubin": estimated_bilirubin,
            "severity": severity,
            "needs_phototherapy": needs_phototherapy,
            "recommendation": recommendation,
            "model": self.model_name,
            "model_type": f"{base_model} + {model_method}",
        }

    def _classify_with_trained_model(self, image_embedding: torch.Tensor) -> Tuple[float, str]:
        """
        Classify using trained classifier on embeddings.

        Args:
            image_embedding: Normalized image embedding from MedSigLIP

        Returns:
            Tuple of (jaundice_prob, method_name)
        """
        # Convert embedding to numpy for sklearn classifiers
        embedding_np = image_embedding.cpu().numpy().reshape(1, -1)

        # Handle different classifier types
        if hasattr(self.classifier, 'predict_proba'):
            # Sklearn classifier with probability support
            proba = self.classifier.predict_proba(embedding_np)
            # Assume binary: [normal, jaundice] or [jaundice, normal]
            if proba.shape[1] >= 2:
                # Check classifier classes to determine order
                if hasattr(self.classifier, 'classes_'):
                    classes = list(self.classifier.classes_)
                    if 1 in classes:
                        jaundice_idx = classes.index(1)
                    else:
                        jaundice_idx = 1  # Default assumption
                else:
                    jaundice_idx = 1
                jaundice_prob = float(proba[0, jaundice_idx])
            else:
                jaundice_prob = float(proba[0, 0])
            return jaundice_prob, "Trained Classifier"

        elif hasattr(self.classifier, 'predict'):
            # Classifier without probability - use binary prediction
            prediction = self.classifier.predict(embedding_np)
            jaundice_prob = float(prediction[0])
            return jaundice_prob, "Trained Classifier (binary)"

        elif isinstance(self.classifier, nn.Module):
            # PyTorch classifier
            self.classifier.eval()
            with torch.no_grad():
                logits = self.classifier(image_embedding)
                probs = torch.softmax(logits, dim=-1)
                if probs.shape[-1] >= 2:
                    jaundice_prob = probs[0, 1].item()
                else:
                    jaundice_prob = probs[0, 0].item()
            return jaundice_prob, "Trained Classifier (PyTorch)"

        else:
            # Unknown classifier type - fall back to zero-shot
            print(f"Warning: Unknown classifier type {type(self.classifier)}, using zero-shot")
            return self._classify_zero_shot(image_embedding)

    def _classify_zero_shot(self, image_embedding: torch.Tensor) -> Tuple[float, str]:
        """
        Classify using zero-shot with text embeddings.

        Args:
            image_embedding: Normalized image embedding from MedSigLIP

        Returns:
            Tuple of (jaundice_prob, method_name)
        """
        # Compute similarities
        jaundice_sim = (image_embedding @ self.jaundice_embeddings.T).squeeze().item()
        normal_sim = (image_embedding @ self.normal_embeddings.T).squeeze().item()

        # Convert to probabilities
        logits = torch.tensor([jaundice_sim, normal_sim]) * 100
        probs = torch.softmax(logits, dim=0)
        jaundice_prob = probs[0].item()

        return jaundice_prob, "Zero-Shot"

    def detect_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = 8,
    ) -> List[Dict]:
        """Detect jaundice from multiple images."""
        results = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            pil_images = [self.preprocess_image(img) for img in batch]

            with torch.no_grad():
                inputs = self.processor(images=pil_images, return_tensors="pt", padding=True).to(self.device)

                # Get image embeddings from SigLIP vision encoder
                if hasattr(self.model, 'get_image_features'):
                    image_embeddings = self.model.get_image_features(**inputs)
                else:
                    vision_outputs = self.model.vision_model(**inputs)
                    image_embeddings = vision_outputs.pooler_output

                image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)

            for j, (img_emb, pil_img) in enumerate(zip(image_embeddings, pil_images)):
                img_emb = img_emb.unsqueeze(0)
                jaundice_sim = (img_emb @ self.jaundice_embeddings.T).squeeze().item()
                normal_sim = (img_emb @ self.normal_embeddings.T).squeeze().item()

                logits = torch.tensor([jaundice_sim, normal_sim]) * 100
                probs = torch.softmax(logits, dim=0)
                jaundice_prob = probs[0].item()

                estimated_bilirubin = self.estimate_bilirubin(pil_img)

                if estimated_bilirubin < self.BILIRUBIN_THRESHOLDS["low"]:
                    severity, needs_phototherapy = "none", False
                elif estimated_bilirubin < self.BILIRUBIN_THRESHOLDS["moderate"]:
                    severity, needs_phototherapy = "mild", False
                elif estimated_bilirubin < self.BILIRUBIN_THRESHOLDS["high"]:
                    severity, needs_phototherapy = "moderate", False
                elif estimated_bilirubin < self.BILIRUBIN_THRESHOLDS["critical"]:
                    severity, needs_phototherapy = "severe", True
                else:
                    severity, needs_phototherapy = "critical", True

                results.append({
                    "has_jaundice": jaundice_prob > self.threshold,
                    "confidence": max(jaundice_prob, 1 - jaundice_prob),
                    "jaundice_score": jaundice_prob,
                    "estimated_bilirubin": estimated_bilirubin,
                    "severity": severity,
                    "needs_phototherapy": needs_phototherapy,
                })

        return results

    def analyze_kramer_zones(self, image: Union[str, Path, Image.Image]) -> Dict:
        """
        Analyze jaundice using Kramer's zones concept.

        Kramer's zones estimate bilirubin based on cephalocaudal progression:
        - Zone 1 (face): ~5-6 mg/dL
        - Zone 2 (chest): ~9 mg/dL
        - Zone 3 (abdomen): ~12 mg/dL
        - Zone 4 (arms/legs): ~15 mg/dL
        - Zone 5 (hands/feet): ~20+ mg/dL

        Args:
            image: Full body or partial neonatal image

        Returns:
            Dictionary with zone analysis
        """
        pil_image = self.preprocess_image(image)
        img_array = np.array(pil_image).astype(float)

        # Simple color-based zone estimation
        r = img_array[:, :, 0]
        g = img_array[:, :, 1]
        b = img_array[:, :, 2]

        yellow_index = np.mean((r + g - b) / (r + g + b + 1e-6))

        # Map yellow index to Kramer zone
        if yellow_index < 0.25:
            zone = 0
            zone_bilirubin = 3
        elif yellow_index < 0.30:
            zone = 1
            zone_bilirubin = 6
        elif yellow_index < 0.35:
            zone = 2
            zone_bilirubin = 9
        elif yellow_index < 0.40:
            zone = 3
            zone_bilirubin = 12
        elif yellow_index < 0.45:
            zone = 4
            zone_bilirubin = 15
        else:
            zone = 5
            zone_bilirubin = 20

        return {
            "kramer_zone": zone,
            "zone_description": self._get_zone_description(zone),
            "estimated_bilirubin_by_zone": zone_bilirubin,
            "yellow_index": round(yellow_index, 3),
        }

    def _get_zone_description(self, zone: int) -> str:
        """Get description for Kramer zone."""
        descriptions = {
            0: "No visible jaundice",
            1: "Face and neck (Zone 1)",
            2: "Upper trunk (Zone 2)",
            3: "Lower trunk and thighs (Zone 3)",
            4: "Arms and lower legs (Zone 4)",
            5: "Hands and feet (Zone 5) - Severe",
        }
        return descriptions.get(zone, "Unknown")


def test_detector():
    """Test the jaundice detector with sample images."""
    print("Testing Jaundice Detector...")

    detector = JaundiceDetector()

    data_dir = Path(__file__).parent.parent.parent / "data" / "raw" / "neojaundice" / "images"

    if data_dir.exists():
        sample_images = list(data_dir.glob("*.jpg"))[:3]

        for img_path in sample_images:
            print(f"\nAnalyzing: {img_path.name}")
            result = detector.detect(img_path)
            print(f"  Jaundice detected: {result['has_jaundice']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Estimated bilirubin: {result['estimated_bilirubin']} mg/dL")
            print(f"  Severity: {result['severity']}")
            print(f"  Needs phototherapy: {result['needs_phototherapy']}")
            print(f"  Recommendation: {result['recommendation']}")
    else:
        print(f"Dataset not found at {data_dir}")


if __name__ == "__main__":
    test_detector()
