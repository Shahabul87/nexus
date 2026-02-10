#!/usr/bin/env python3
"""
Train Jaundice Classifier on MedSigLIP Embeddings

Extracts MedSigLIP embeddings from neonatal skin images with data augmentation,
then trains SVM/LogisticRegression for jaundice severity classification.

HAI-DEF Model: MedSigLIP (google/medsiglip-448)
Dataset: NeoJaundice (2,235 images with ground truth bilirubin)

Pipeline:
    1. Load neonatal skin images from NeoJaundice dataset
    2. Apply augmentation (rotation, brightness, contrast, flips)
    3. Extract MedSigLIP embeddings (1152-dim)
    4. Train SVM (RBF) + LogisticRegression with stratified 5-fold CV
    5. Save best classifier to models/linear_probes/jaundice_classifier.joblib

Note: Binary classification (jaundice vs normal) based on bilirubin > 5 mg/dL threshold.
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Bilirubin thresholds
JAUNDICE_THRESHOLD = 5.0  # mg/dL for binary classification


def load_neojaundice(data_dir: Path) -> List[Dict]:
    """Load images and labels from NeoJaundice dataset.

    Reads chd_jaundice_published_2.csv for ground truth bilirubin values.
    """
    csv_path = data_dir / "chd_jaundice_published_2.csv"
    images_dir = data_dir / "images"

    if not csv_path.exists():
        print(f"WARNING: Metadata CSV not found at {csv_path}")
        # Fallback: load all images with pseudo-labels
        samples = []
        if images_dir.exists():
            for img_path in sorted(images_dir.glob("*.jpg")):
                samples.append({
                    "path": img_path,
                    "bilirubin": 10.0,  # pseudo-label
                    "label": 1,
                })
        print(f"Loaded {len(samples)} images with pseudo-labels (no CSV)")
        return samples

    df = pd.read_csv(csv_path)
    samples = []

    for _, row in df.iterrows():
        img_name = row["image_idx"]
        img_path = images_dir / img_name
        if not img_path.exists():
            continue

        bilirubin = float(row["blood(mg/dL)"])
        label = 1 if bilirubin >= JAUNDICE_THRESHOLD else 0

        samples.append({
            "path": img_path,
            "bilirubin": bilirubin,
            "label": label,
        })

    n_jaundice = sum(s["label"] for s in samples)
    n_normal = len(samples) - n_jaundice
    print(f"Loaded {len(samples)} images (jaundice={n_jaundice}, normal={n_normal})")
    print(f"Bilirubin range: {min(s['bilirubin'] for s in samples):.1f} - {max(s['bilirubin'] for s in samples):.1f} mg/dL")

    return samples


def augment_image(image: Image.Image, seed: int) -> Image.Image:
    """Apply deterministic augmentation to a PIL image."""
    rng = np.random.RandomState(seed)

    angle = rng.uniform(-15, 15)
    image = image.rotate(angle, fillcolor=(128, 128, 128))

    factor = rng.uniform(0.8, 1.2)
    image = ImageEnhance.Brightness(image).enhance(factor)

    factor = rng.uniform(0.8, 1.2)
    image = ImageEnhance.Contrast(image).enhance(factor)

    factor = rng.uniform(0.8, 1.2)
    image = ImageEnhance.Color(image).enhance(factor)

    if rng.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    if rng.random() > 0.7:
        image = image.filter(ImageFilter.GaussianBlur(radius=1))

    return image


def extract_medsiglip_embeddings(
    images: List[Image.Image],
    model,
    processor,
    device: str,
    batch_size: int = 16,
) -> np.ndarray:
    """Extract MedSigLIP embeddings for a list of PIL images."""
    embeddings = []

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        with torch.no_grad():
            inputs = processor(images=batch, return_tensors="pt").to(device)

            if hasattr(model, 'get_image_features'):
                emb = model.get_image_features(**inputs)
            else:
                outputs = model(**inputs)
                if hasattr(outputs, 'image_embeds'):
                    emb = outputs.image_embeds
                elif hasattr(outputs, 'vision_model_output'):
                    emb = outputs.vision_model_output.pooler_output
                else:
                    vision_outputs = model.vision_model(**inputs)
                    emb = vision_outputs.pooler_output

            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy())

    return np.vstack(embeddings)


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("MedSigLIP Jaundice Classifier Training")
    print("=" * 60)

    data_dir = PROJECT_ROOT / "data" / "raw" / "neojaundice"
    output_dir = PROJECT_ROOT / "models" / "linear_probes"
    embedding_cache_dir = PROJECT_ROOT / "models" / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_cache_dir.mkdir(parents=True, exist_ok=True)

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Fewer augmentations since we have more data (2235 images)
    n_augmentations = 3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Step 1: Load dataset
    print("\n--- Step 1: Loading Dataset ---")
    samples = load_neojaundice(data_dir)
    if len(samples) == 0:
        print("ERROR: No samples found.")
        return

    # Step 2: Load MedSigLIP
    print("\n--- Step 2: Loading MedSigLIP ---")
    hf_token = os.environ.get("HF_TOKEN")
    model = None

    from transformers import AutoProcessor, AutoModel

    for model_id in ["google/medsiglip-448", "google/siglip-base-patch16-224"]:
        try:
            print(f"Loading {model_id}...")
            processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
            model = AutoModel.from_pretrained(model_id, token=hf_token).to(device)
            model.eval()
            print(f"Loaded: {model_id}")
            break
        except Exception as e:
            print(f"Could not load {model_id}: {e}")

    if model is None:
        print("ERROR: Could not load MedSigLIP.")
        return

    # Step 3: Extract embeddings with augmentation
    print("\n--- Step 3: Extracting Embeddings with Augmentation ---")
    embeddings_cache = embedding_cache_dir / f"jaundice_embeddings_aug{n_augmentations}.npy"
    labels_cache = embedding_cache_dir / f"jaundice_labels_aug{n_augmentations}.npy"

    if embeddings_cache.exists() and labels_cache.exists():
        print(f"Loading cached embeddings from {embeddings_cache}")
        X = np.load(embeddings_cache)
        y = np.load(labels_cache)
    else:
        all_images = []
        all_labels = []

        for sample in tqdm(samples, desc="Loading + augmenting"):
            try:
                img = Image.open(sample["path"]).convert("RGB")
            except Exception as e:
                print(f"  Error loading {sample['path'].name}: {e}")
                continue
            label = sample["label"]

            # Original image
            all_images.append(img)
            all_labels.append(label)

            # Augmented copies
            for aug_i in range(n_augmentations):
                aug_seed = hash((str(sample["path"]), aug_i)) % (2**31)
                aug_img = augment_image(img.copy(), aug_seed)
                all_images.append(aug_img)
                all_labels.append(label)

        print(f"Total images (original + augmented): {len(all_images)}")

        # Extract embeddings in batches
        print("Extracting MedSigLIP embeddings...")
        X = extract_medsiglip_embeddings(all_images, model, processor, device)
        y = np.array(all_labels)

        np.save(embeddings_cache, X)
        np.save(labels_cache, y)
        print(f"Saved embeddings to {embeddings_cache} (shape: {X.shape})")

    # Free model memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Step 4: Train classifiers
    print("\n--- Step 4: Training Classifiers ---")
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report
    import joblib

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    classifiers = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, C=1.0, solver="lbfgs", random_state=seed,
        ),
        "SVM_RBF": SVC(
            kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=seed,
        ),
        "SVM_Linear": SVC(
            kernel="linear", C=1.0, probability=True, random_state=seed,
        ),
    }

    best_name = None
    best_score = 0.0
    results = {}

    for name, clf in classifiers.items():
        print(f"\n  Training {name}...")
        scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring="accuracy")
        mean_acc = scores.mean()
        std_acc = scores.std()
        results[name] = {"mean_accuracy": float(mean_acc), "std_accuracy": float(std_acc)}
        print(f"  {name}: {mean_acc:.4f} +/- {std_acc:.4f}")

        if mean_acc > best_score:
            best_score = mean_acc
            best_name = name

    print(f"\nBest classifier: {best_name} ({best_score:.4f})")

    # Train final model on all data
    print(f"\nTraining final {best_name} on all data...")
    best_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", classifiers[best_name]),
    ])
    best_pipeline.fit(X, y)

    y_pred = best_pipeline.predict(X)
    print("\nClassification Report (full dataset):")
    print(classification_report(y, y_pred, target_names=["normal", "jaundice"]))

    # Step 5: Save model
    print("\n--- Step 5: Saving Model ---")
    model_path = output_dir / "jaundice_classifier.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"Saved classifier to {model_path}")

    metadata = {
        "model_type": best_name,
        "embedding_source": "MedSigLIP (google/medsiglip-448)",
        "embedding_dim": int(X.shape[1]),
        "num_classes": 2,
        "classes": {"normal": 0, "jaundice": 1},
        "bilirubin_threshold": JAUNDICE_THRESHOLD,
        "cv_accuracy_mean": float(best_score),
        "cv_accuracy_std": float(results[best_name]["std_accuracy"]),
        "num_original_samples": len(samples),
        "num_augmented_samples": len(y),
        "augmentations_per_image": n_augmentations,
        "all_results": results,
        "seed": seed,
    }

    metadata_path = output_dir / "jaundice_classifier_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print(f"DONE: Jaundice classifier trained")
    print(f"  Model: {best_name}")
    print(f"  Accuracy: {best_score:.4f}")
    print(f"  Original images: {len(samples)}, Total with augmentation: {len(y)}")
    print(f"  Saved to: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
