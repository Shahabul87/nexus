"""
Anemia Detector Module

Uses MedSigLIP from Google HAI-DEF for anemia detection from conjunctiva images.
Implements zero-shot classification with medical text prompts per NEXUS_MASTER_PLAN.md.

HAI-DEF Model: google/medsiglip-448 (MedSigLIP)
Documentation: https://developers.google.com/health-ai-developer-foundations/medsiglip
"""

import os
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
    "google/medsiglip-448",              # MedSigLIP - official HAI-DEF model
    "google/siglip-base-patch16-224",    # SigLIP 224 - fallback
]


class AnemiaDetector:
    """
    Detects anemia from conjunctiva (inner eyelid) images using MedSigLIP.

    Uses zero-shot classification with medical prompts for detection.
    HAI-DEF Model: google/medsiglip-448 (MedSigLIP)
    Fallback: siglip-base-patch16-224
    """

    # Medical text prompts for zero-shot classification (optimized for MedSigLIP)
    # Expanded prompt set with specific clinical language for better discrimination
    ANEMIC_PROMPTS = [
        "pale conjunctiva with visible pallor indicating anemia",
        "conjunctival pallor grade 2 or higher with reduced vascularity",
        "white or very pale inner eyelid mucosa suggesting low hemoglobin",
        "conjunctiva showing significant pallor and poor blood perfusion",
        "anemic eye with pale pink to white palpebral conjunctiva",
        "inner eyelid lacking red coloration consistent with severe anemia",
        "conjunctiva with washed out appearance and faint vascular pattern",
        "pale mucous membrane of the lower eyelid suggesting iron deficiency",
    ]

    HEALTHY_PROMPTS = [
        "healthy red conjunctiva with rich vascular pattern",
        "well-perfused bright pink inner eyelid with visible blood vessels",
        "normal conjunctiva showing deep red-pink coloration",
        "conjunctiva with healthy blood supply and strong red color",
        "richly vascularized palpebral conjunctiva with normal hemoglobin",
        "inner eyelid with vibrant red-pink mucosa and clear vessels",
        "non-anemic conjunctiva showing robust red perfusion",
        "conjunctival mucosa with normal deep pink to red appearance",
    ]

    def __init__(
        self,
        model_name: Optional[str] = None,  # Auto-select MedSigLIP
        device: Optional[str] = None,
        threshold: float = 0.5,
    ):
        """
        Initialize the Anemia Detector with MedSigLIP.

        Args:
            model_name: HuggingFace model name (auto-selects HAI-DEF MedSigLIP if None)
            device: Device to run model on (auto-detected if None)
            threshold: Classification threshold for anemia detection
        """
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library required. Install with: pip install transformers")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self._model_loaded = False
        self.classifier = None  # Can be set by pipeline for trained classification

        # Determine which models to try
        models_to_try = [model_name] if model_name else MEDSIGLIP_MODEL_IDS

        # HuggingFace token for gated models
        hf_token = os.environ.get("HF_TOKEN")

        # Try loading models in order of preference
        for candidate_model in models_to_try:
            print(f"Loading HAI-DEF model: {candidate_model}")
            try:
                self.processor = AutoProcessor.from_pretrained(
                    candidate_model, token=hf_token
                )
                self.model = AutoModel.from_pretrained(
                    candidate_model, token=hf_token
                ).to(self.device)
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

        # Pre-compute text embeddings for efficiency
        self._precompute_text_embeddings()

        # Try to auto-load trained classifier
        self._auto_load_classifier()

        # Indicate which model variant is being used
        is_medsiglip = "medsiglip" in self.model_name
        model_type = "MedSigLIP" if is_medsiglip else "SigLIP (fallback)"
        classifier_status = "with trained classifier" if self.classifier else "zero-shot"
        print(f"Anemia Detector (HAI-DEF {model_type}, {classifier_status}) initialized on {self.device}")

    def _auto_load_classifier(self) -> None:
        """Auto-load trained anemia classifier if available."""
        if self.classifier is not None:
            return  # Already set externally

        try:
            import joblib
        except ImportError:
            return

        default_paths = [
            Path(__file__).parent.parent.parent / "models" / "linear_probes" / "anemia_classifier.joblib",
            Path("models/linear_probes/anemia_classifier.joblib"),
        ]

        for path in default_paths:
            if path.exists():
                try:
                    self.classifier = joblib.load(path)
                    print(f"Auto-loaded anemia classifier from {path}")
                    return
                except Exception as e:
                    print(f"Warning: Could not load classifier from {path}: {e}")

    # Logit temperature for softmax conversion (lower = more spread, higher = sharper)
    LOGIT_SCALE = 30.0

    def _precompute_text_embeddings(self) -> None:
        """Pre-compute text embeddings for zero-shot classification using SigLIP.

        Stores individual prompt embeddings for max-similarity scoring,
        which outperforms mean-pooled embeddings for medical image classification.
        """
        all_prompts = self.ANEMIC_PROMPTS + self.HEALTHY_PROMPTS

        with torch.no_grad():
            # SigLIP uses different API than CLIP
            inputs = self.processor(
                text=all_prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            ).to(self.device)

            # Get text embeddings - support multiple output APIs
            if hasattr(self.model, 'get_text_features'):
                text_embeddings = self.model.get_text_features(**inputs)
            else:
                outputs = self.model(**inputs)
                if hasattr(outputs, 'text_embeds'):
                    text_embeddings = outputs.text_embeds
                elif hasattr(outputs, 'text_model_output'):
                    text_embeddings = outputs.text_model_output.pooler_output
                else:
                    text_outputs = self.model.text_model(**inputs)
                    text_embeddings = text_outputs.pooler_output

            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

            # Store individual embeddings for max-similarity scoring
            n_anemic = len(self.ANEMIC_PROMPTS)
            self.anemic_embeddings_all = text_embeddings[:n_anemic]  # (N, D)
            self.healthy_embeddings_all = text_embeddings[n_anemic:]  # (M, D)

            # Also keep mean embeddings as fallback
            self.anemic_embeddings = self.anemic_embeddings_all.mean(dim=0, keepdim=True)
            self.healthy_embeddings = self.healthy_embeddings_all.mean(dim=0, keepdim=True)
            self.anemic_embeddings = self.anemic_embeddings / self.anemic_embeddings.norm(dim=-1, keepdim=True)
            self.healthy_embeddings = self.healthy_embeddings / self.healthy_embeddings.norm(dim=-1, keepdim=True)

    def preprocess_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """
        Preprocess image for analysis.

        Args:
            image: Path to image or PIL Image

        Returns:
            Preprocessed PIL Image
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Expected str, Path, or PIL Image, got {type(image)}")

        return image

    def detect(self, image: Union[str, Path, Image.Image]) -> Dict:
        """
        Detect anemia from conjunctiva image.

        Uses trained classifier if available, otherwise falls back to
        zero-shot classification with MedSigLIP.

        Args:
            image: Conjunctiva image (path or PIL Image)

        Returns:
            Dictionary containing:
                - is_anemic: Boolean indicating anemia detection
                - confidence: Confidence score (0-1)
                - anemia_score: Raw anemia probability
                - healthy_score: Raw healthy probability
                - risk_level: "high", "medium", or "low"
                - recommendation: Clinical recommendation
        """
        # Preprocess image
        pil_image = self.preprocess_image(image)

        # Get image embedding using SigLIP
        with torch.no_grad():
            inputs = self.processor(
                images=pil_image,
                return_tensors="pt",
            ).to(self.device)

            # Get image embeddings - support multiple output APIs
            if hasattr(self.model, 'get_image_features'):
                image_embedding = self.model.get_image_features(**inputs)
            else:
                outputs = self.model(**inputs)
                if hasattr(outputs, 'image_embeds'):
                    image_embedding = outputs.image_embeds
                elif hasattr(outputs, 'vision_model_output'):
                    image_embedding = outputs.vision_model_output.pooler_output
                else:
                    vision_outputs = self.model.vision_model(**inputs)
                    image_embedding = vision_outputs.pooler_output

            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

        # Use trained classifier if available, otherwise zero-shot
        if self.classifier is not None:
            anemia_prob, healthy_prob, model_method = self._classify_with_trained_model(image_embedding)
        else:
            anemia_prob, healthy_prob, model_method = self._classify_zero_shot(image_embedding)

        # Determine risk level
        if anemia_prob > 0.7:
            risk_level = "high"
            recommendation = "URGENT: Refer for blood test immediately. High likelihood of anemia."
        elif anemia_prob > 0.5:
            risk_level = "medium"
            recommendation = "Schedule blood test within 48 hours. Moderate anemia indicators present."
        else:
            risk_level = "low"
            recommendation = "No immediate concern. Routine follow-up recommended."

        is_medsiglip = "medsiglip" in self.model_name
        base_model = "MedSigLIP (HAI-DEF)" if is_medsiglip else "SigLIP (fallback)"

        return {
            "is_anemic": anemia_prob > self.threshold,
            "confidence": max(anemia_prob, healthy_prob),
            "anemia_score": anemia_prob,
            "healthy_score": healthy_prob,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "model": self.model_name,
            "model_type": f"{base_model} + {model_method}",
        }

    def _classify_with_trained_model(self, image_embedding: torch.Tensor) -> Tuple[float, float, str]:
        """
        Classify using trained classifier on embeddings.

        Args:
            image_embedding: Normalized image embedding from MedSigLIP

        Returns:
            Tuple of (anemia_prob, healthy_prob, method_name)
        """
        # Convert embedding to numpy for sklearn classifiers
        embedding_np = image_embedding.cpu().numpy().reshape(1, -1)

        # Handle different classifier types
        if hasattr(self.classifier, 'predict_proba'):
            # Sklearn classifier with probability support
            proba = self.classifier.predict_proba(embedding_np)
            # Assume binary: [healthy, anemic] or [anemic, healthy]
            if proba.shape[1] >= 2:
                # Check classifier classes to determine order
                if hasattr(self.classifier, 'classes_'):
                    classes = list(self.classifier.classes_)
                    if 1 in classes:
                        anemia_idx = classes.index(1)
                    else:
                        anemia_idx = 1  # Default assumption
                else:
                    anemia_idx = 1
                anemia_prob = float(proba[0, anemia_idx])
                healthy_prob = 1.0 - anemia_prob
            else:
                anemia_prob = float(proba[0, 0])
                healthy_prob = 1.0 - anemia_prob
            return anemia_prob, healthy_prob, "Trained Classifier"

        elif hasattr(self.classifier, 'predict'):
            # Classifier without probability - use binary prediction
            prediction = self.classifier.predict(embedding_np)
            anemia_prob = float(prediction[0])
            healthy_prob = 1.0 - anemia_prob
            return anemia_prob, healthy_prob, "Trained Classifier (binary)"

        elif isinstance(self.classifier, nn.Module):
            # PyTorch classifier
            self.classifier.eval()
            with torch.no_grad():
                logits = self.classifier(image_embedding)
                probs = torch.softmax(logits, dim=-1)
                if probs.shape[-1] >= 2:
                    anemia_prob = probs[0, 1].item()
                    healthy_prob = probs[0, 0].item()
                else:
                    anemia_prob = probs[0, 0].item()
                    healthy_prob = 1.0 - anemia_prob
            return anemia_prob, healthy_prob, "Trained Classifier (PyTorch)"

        else:
            # Unknown classifier type - fall back to zero-shot
            print(f"Warning: Unknown classifier type {type(self.classifier)}, using zero-shot")
            return self._classify_zero_shot(image_embedding)

    def _classify_zero_shot(self, image_embedding: torch.Tensor) -> Tuple[float, float, str]:
        """
        Classify using zero-shot with max-similarity scoring.

        Uses the maximum cosine similarity across all prompts per class
        rather than mean-pooled embeddings, which provides better
        discrimination for medical image classification.

        Args:
            image_embedding: Normalized image embedding from MedSigLIP

        Returns:
            Tuple of (anemia_prob, healthy_prob, method_name)
        """
        # Max-similarity: take the best-matching prompt per class
        anemia_sims = (image_embedding @ self.anemic_embeddings_all.T).squeeze(0)
        healthy_sims = (image_embedding @ self.healthy_embeddings_all.T).squeeze(0)

        # Ensure at least 1-D for .max() to work on single-image inputs
        if anemia_sims.dim() == 0:
            anemia_sims = anemia_sims.unsqueeze(0)
        if healthy_sims.dim() == 0:
            healthy_sims = healthy_sims.unsqueeze(0)

        anemia_sim = anemia_sims.max().item()
        healthy_sim = healthy_sims.max().item()

        # Convert to probabilities with tuned temperature
        logits = torch.tensor([anemia_sim, healthy_sim], device="cpu") * self.LOGIT_SCALE
        probs = torch.softmax(logits, dim=0)
        anemia_prob = probs[0].item()
        healthy_prob = probs[1].item()

        return anemia_prob, healthy_prob, "Zero-Shot"

    def detect_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = 8,
    ) -> List[Dict]:
        """
        Detect anemia from multiple images.

        Args:
            images: List of conjunctiva images
            batch_size: Batch size for processing

        Returns:
            List of detection results
        """
        results = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]

            # Process batch
            pil_images = [self.preprocess_image(img) for img in batch]

            with torch.no_grad():
                inputs = self.processor(
                    images=pil_images,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                # Get image embeddings - support multiple output APIs
                if hasattr(self.model, 'get_image_features'):
                    image_embeddings = self.model.get_image_features(**inputs)
                else:
                    outputs = self.model(**inputs)
                    if hasattr(outputs, 'image_embeds'):
                        image_embeddings = outputs.image_embeds
                    elif hasattr(outputs, 'vision_model_output'):
                        image_embeddings = outputs.vision_model_output.pooler_output
                    else:
                        vision_outputs = self.model.vision_model(**inputs)
                        image_embeddings = vision_outputs.pooler_output

                image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)

            # Compute max-similarities for each image
            for j, img_emb in enumerate(image_embeddings):
                img_emb = img_emb.unsqueeze(0)

                # Use trained classifier if available, otherwise zero-shot
                if self.classifier is not None:
                    anemia_prob, healthy_prob, _ = self._classify_with_trained_model(img_emb)
                    # Skip zero-shot path below
                    if anemia_prob > 0.7:
                        risk_level = "high"
                        recommendation = "URGENT: Refer for blood test immediately."
                    elif anemia_prob > 0.5:
                        risk_level = "medium"
                        recommendation = "Schedule blood test within 48 hours."
                    else:
                        risk_level = "low"
                        recommendation = "No immediate concern."

                    results.append({
                        "is_anemic": anemia_prob > self.threshold,
                        "confidence": max(anemia_prob, healthy_prob),
                        "anemia_score": anemia_prob,
                        "healthy_score": healthy_prob,
                        "risk_level": risk_level,
                        "recommendation": recommendation,
                    })
                    continue

                anemia_sims = (img_emb @ self.anemic_embeddings_all.T).squeeze(0)
                healthy_sims = (img_emb @ self.healthy_embeddings_all.T).squeeze(0)

                if anemia_sims.dim() == 0:
                    anemia_sims = anemia_sims.unsqueeze(0)
                if healthy_sims.dim() == 0:
                    healthy_sims = healthy_sims.unsqueeze(0)

                anemia_sim = anemia_sims.max().item()
                healthy_sim = healthy_sims.max().item()

                logits = torch.tensor([anemia_sim, healthy_sim], device="cpu") * self.LOGIT_SCALE
                probs = torch.softmax(logits, dim=0)
                anemia_prob = probs[0].item()
                healthy_prob = probs[1].item()

                if anemia_prob > 0.7:
                    risk_level = "high"
                    recommendation = "URGENT: Refer for blood test immediately."
                elif anemia_prob > 0.5:
                    risk_level = "medium"
                    recommendation = "Schedule blood test within 48 hours."
                else:
                    risk_level = "low"
                    recommendation = "No immediate concern."

                results.append({
                    "is_anemic": anemia_prob > self.threshold,
                    "confidence": max(anemia_prob, healthy_prob),
                    "anemia_score": anemia_prob,
                    "healthy_score": healthy_prob,
                    "risk_level": risk_level,
                    "recommendation": recommendation,
                })

        return results

    def analyze_color_features(self, image: Union[str, Path, Image.Image]) -> Dict:
        """
        Analyze color features of conjunctiva image.

        This provides interpretable features based on medical literature
        that correlates pallor with anemia.

        Args:
            image: Conjunctiva image

        Returns:
            Dictionary with color analysis results
        """
        pil_image = self.preprocess_image(image)
        img_array = np.array(pil_image)

        # Extract RGB channels
        r_channel = img_array[:, :, 0].astype(float)
        g_channel = img_array[:, :, 1].astype(float)
        b_channel = img_array[:, :, 2].astype(float)

        # Calculate color statistics
        mean_r = np.mean(r_channel)
        mean_g = np.mean(g_channel)
        mean_b = np.mean(b_channel)

        # Red ratio (higher in healthy, lower in anemic)
        total_intensity = mean_r + mean_g + mean_b
        red_ratio = mean_r / total_intensity if total_intensity > 0 else 0

        # Pallor index (higher means more pale/anemic)
        # Based on reduced red-to-green ratio in anemic conjunctiva
        pallor_index = 1 - (mean_r / (mean_g + 1e-6))
        pallor_index = max(0, min(1, (pallor_index + 0.5) / 1.5))

        # Hemoglobin estimation (rough approximation)
        # Normal Hb: 12-16 g/dL for women, 14-18 for men
        # This is a rough estimate based on color analysis
        estimated_hb = 8 + (red_ratio * 12)

        return {
            "mean_red": mean_r,
            "mean_green": mean_g,
            "mean_blue": mean_b,
            "red_ratio": red_ratio,
            "pallor_index": pallor_index,
            "estimated_hemoglobin": round(estimated_hb, 1),
            "interpretation": "Low hemoglobin" if pallor_index > 0.5 else "Normal hemoglobin",
        }


def test_detector():
    """Test the anemia detector with sample images."""
    print("Testing Anemia Detector...")

    detector = AnemiaDetector()

    # Test with sample images from dataset
    data_dir = Path(__file__).parent.parent.parent / "data" / "raw" / "eyes-defy-anemia"

    if data_dir.exists():
        # Find sample images
        sample_images = list(data_dir.rglob("*.jpg"))[:3]

        for img_path in sample_images:
            print(f"\nAnalyzing: {img_path.name}")
            result = detector.detect(img_path)
            print(f"  Anemia detected: {result['is_anemic']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Risk level: {result['risk_level']}")
            print(f"  Recommendation: {result['recommendation']}")

            # Color analysis
            color_info = detector.analyze_color_features(img_path)
            print(f"  Estimated Hb: {color_info['estimated_hemoglobin']} g/dL")
    else:
        print(f"Dataset not found at {data_dir}")
        print("Please run download_datasets.py first")


if __name__ == "__main__":
    test_detector()
