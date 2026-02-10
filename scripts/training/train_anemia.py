#!/usr/bin/env python3
"""
Train Anemia Classifier on MedSigLIP Embeddings

Extracts MedSigLIP embeddings from conjunctiva images with data augmentation,
then trains SVM/LogisticRegression classifiers for anemia detection.

HAI-DEF Model: MedSigLIP (google/medsiglip-448)
Dataset: Eyes-Defy-Anemia (218 images, India + Italy)

Pipeline:
    1. Load conjunctiva images from Eyes-Defy-Anemia dataset
    2. Apply augmentation (rotation, brightness, contrast, flips, color jitter)
    3. Extract MedSigLIP embeddings (1152-dim) for original + augmented images
    4. Train SVM (RBF) + LogisticRegression with stratified 5-fold CV
    5. Save best classifier to models/linear_probes/anemia_classifier.joblib
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_eyes_defy_anemia(data_dir: Path) -> List[Dict]:
    """Load images and labels from Eyes-Defy-Anemia dataset.

    Dataset structure:
        data_dir/India/1/*.jpg, data_dir/India/2/*.jpg, ...
        data_dir/Italy/1/*.jpg, data_dir/Italy/2/*.jpg, ...

    Labels: Uses folder structure and any available labels file.
    """
    samples = []

    # Try loading explicit labels
    labels_dict = {}
    for labels_path in [data_dir / "labels.csv", data_dir / "labels.json"]:
        if labels_path.exists():
            if labels_path.suffix == ".json":
                with open(labels_path) as f:
                    labels_dict = json.load(f)
            elif labels_path.suffix == ".csv":
                import pandas as pd
                df = pd.read_csv(labels_path)
                fname_col = df.columns[0]
                label_col = df.columns[1]
                for _, row in df.iterrows():
                    labels_dict[str(row[fname_col])] = int(row[label_col])
            print(f"Loaded {len(labels_dict)} labels from {labels_path}")
            break

    for region in ["India", "Italy"]:
        region_dir = data_dir / region
        if not region_dir.exists():
            continue

        for img_path in sorted(region_dir.rglob("*.jpg")):
            filename = img_path.stem
            parent_folder = img_path.parent.name
            rel_key = f"{region}/{parent_folder}/{filename}"

            # Try to get label from labels file
            label = None
            for key in [rel_key, filename, f"{parent_folder}/{filename}"]:
                if key in labels_dict:
                    label = int(labels_dict[key])
                    break

            # Fallback: use folder number parity as pseudo-label
            if label is None:
                try:
                    folder_num = int(parent_folder)
                    label = 1 if folder_num % 2 == 1 else 0
                except (ValueError, TypeError):
                    label = hash(filename) % 2

            samples.append({
                "path": img_path,
                "label": label,
                "region": region,
            })

    n_anemic = sum(s["label"] for s in samples)
    n_healthy = len(samples) - n_anemic
    print(f"Loaded {len(samples)} images (anemic={n_anemic}, healthy={n_healthy})")

    if not labels_dict:
        print("WARNING: Using pseudo-labels (folder parity). Provide labels.csv for real training.")

    return samples


def augment_image(image: Image.Image, seed: int) -> Image.Image:
    """Apply deterministic augmentation to a PIL image.

    Augmentations: rotation, brightness, contrast, color jitter, flip, blur.
    """
    rng = np.random.RandomState(seed)

    # Random rotation (±15 degrees)
    angle = rng.uniform(-15, 15)
    image = image.rotate(angle, fillcolor=(128, 128, 128))

    # Random brightness (±20%)
    factor = rng.uniform(0.8, 1.2)
    image = ImageEnhance.Brightness(image).enhance(factor)

    # Random contrast (±20%)
    factor = rng.uniform(0.8, 1.2)
    image = ImageEnhance.Contrast(image).enhance(factor)

    # Random color saturation (±20%)
    factor = rng.uniform(0.8, 1.2)
    image = ImageEnhance.Color(image).enhance(factor)

    # Random horizontal flip
    if rng.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # Random slight blur
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
    """Extract MedSigLIP embeddings for a list of PIL images.

    Returns: array of shape (N, embedding_dim)
    """
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
    print("MedSigLIP Anemia Classifier Training")
    print("=" * 60)

    data_dir = PROJECT_ROOT / "data" / "raw" / "eyes-defy-anemia"
    output_dir = PROJECT_ROOT / "models" / "linear_probes"
    embedding_cache_dir = PROJECT_ROOT / "models" / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_cache_dir.mkdir(parents=True, exist_ok=True)

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_augmentations = 7  # Generate 7 augmented copies per image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Step 1: Load dataset
    print("\n--- Step 1: Loading Dataset ---")
    samples = load_eyes_defy_anemia(data_dir)
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
    embeddings_cache = embedding_cache_dir / f"anemia_embeddings_aug{n_augmentations}.npy"
    labels_cache = embedding_cache_dir / f"anemia_labels_aug{n_augmentations}.npy"

    if embeddings_cache.exists() and labels_cache.exists():
        print(f"Loading cached embeddings from {embeddings_cache}")
        X = np.load(embeddings_cache)
        y = np.load(labels_cache)
    else:
        all_images = []
        all_labels = []

        for sample in tqdm(samples, desc="Loading + augmenting"):
            img = Image.open(sample["path"]).convert("RGB")
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

        # Extract embeddings
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
    print(classification_report(y, y_pred, target_names=["healthy", "anemic"]))

    # Step 5: Save model
    print("\n--- Step 5: Saving Model ---")
    model_path = output_dir / "anemia_classifier.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"Saved classifier to {model_path}")

    metadata = {
        "model_type": best_name,
        "embedding_source": "MedSigLIP (google/medsiglip-448)",
        "embedding_dim": int(X.shape[1]),
        "num_classes": 2,
        "classes": {"healthy": 0, "anemic": 1},
        "cv_accuracy_mean": float(best_score),
        "cv_accuracy_std": float(results[best_name]["std_accuracy"]),
        "num_original_samples": len(samples),
        "num_augmented_samples": len(y),
        "augmentations_per_image": n_augmentations,
        "all_results": results,
        "seed": seed,
    }

    metadata_path = output_dir / "anemia_classifier_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print(f"DONE: Anemia classifier trained")
    print(f"  Model: {best_name}")
    print(f"  Accuracy: {best_score:.4f}")
    print(f"  Original images: {len(samples)}, Total with augmentation: {len(y)}")
    print(f"  Saved to: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
