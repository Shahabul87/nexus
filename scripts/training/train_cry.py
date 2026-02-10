#!/usr/bin/env python3
"""
Train Cry Classifier on HeAR Embeddings

Extracts HeAR (Health Acoustic Representations) 512-dim embeddings from
infant cry audio and trains a linear classifier for cry type classification.

HAI-DEF Model: HeAR (google/hear-pytorch)
Dataset: donate-a-cry corpus (5 classes: belly_pain, burping, discomfort, hungry, tired)

Pipeline:
    1. Load cry audio files from donate-a-cry dataset
    2. Resample to 16kHz mono, split into 2-second clips
    3. Extract HeAR embeddings (512-dim per clip)
    4. Train SVM + LogisticRegression on embeddings
    5. Save best classifier to models/linear_probes/cry_classifier.joblib
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import librosa
from tqdm import tqdm

# Ensure src is on path for hear_preprocessing
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# HeAR constants
SAMPLE_RATE = 16000
CLIP_DURATION = 2  # seconds
CLIP_SAMPLES = SAMPLE_RATE * CLIP_DURATION  # 32000

CRY_CATEGORIES = {
    "belly_pain": 0,
    "burping": 1,
    "discomfort": 2,
    "hungry": 3,
    "tired": 4,
}
CATEGORY_NAMES = {v: k for k, v in CRY_CATEGORIES.items()}


def load_donate_a_cry(data_dir: Path) -> List[Dict]:
    """Load labeled audio files from donate-a-cry corpus.

    Args:
        data_dir: Path to donate-a-cry root directory.

    Returns:
        List of dicts with 'path', 'label', 'category' keys.
    """
    corpus_dir = data_dir / "donateacry_corpus_cleaned_and_updated_data"
    samples = []

    if not corpus_dir.exists():
        print(f"Warning: {corpus_dir} not found")
        return samples

    for category, label in CRY_CATEGORIES.items():
        cat_dir = corpus_dir / category
        if not cat_dir.exists():
            continue
        audio_files = list(cat_dir.glob("*.wav")) + list(cat_dir.glob("*.ogg"))
        for audio_path in audio_files:
            samples.append({
                "path": audio_path,
                "label": label,
                "category": category,
            })

    print(f"Loaded {len(samples)} labeled samples from donate-a-cry:")
    for cat, label in CRY_CATEGORIES.items():
        count = sum(1 for s in samples if s["label"] == label)
        print(f"  {cat}: {count} files")

    return samples


def load_and_preprocess_audio(path: Path) -> np.ndarray:
    """Load audio file and resample to 16kHz mono.

    Returns exactly CLIP_SAMPLES (32000) samples, padded or trimmed.
    """
    try:
        audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        # Fix to exactly 2 seconds
        audio = librosa.util.fix_length(audio, size=CLIP_SAMPLES)
        return audio
    except Exception as e:
        print(f"  Error loading {path.name}: {e}")
        return np.zeros(CLIP_SAMPLES, dtype=np.float32)


def extract_hear_embeddings_batch(
    audio_arrays: List[np.ndarray],
    hear_model: torch.nn.Module,
    device: str,
) -> np.ndarray:
    """Extract HeAR embeddings for a batch of audio arrays.

    Args:
        audio_arrays: List of audio arrays (each 32000 samples at 16kHz)
        hear_model: Loaded HeAR model
        device: torch device string

    Returns:
        Embeddings array of shape (N, embedding_dim)
    """
    from nexus.hear_preprocessing import preprocess_audio

    embeddings = []
    with torch.no_grad():
        for audio in audio_arrays:
            # Convert to tensor: (1, 32000)
            waveform = torch.tensor(
                audio.astype(np.float32)
            ).unsqueeze(0).to(device)

            # Preprocess: raw audio -> mel-PCEN spectrogram (1, 1, 192, 128)
            spectrogram = preprocess_audio(waveform)

            # Forward pass through HeAR ViT
            output = hear_model(
                pixel_values=spectrogram,
                return_dict=True,
            )

            # Extract embedding
            if hasattr(output, 'pooler_output') and output.pooler_output is not None:
                embedding = output.pooler_output
            elif hasattr(output, 'last_hidden_state'):
                embedding = output.last_hidden_state[:, 1:, :].mean(dim=1)
            elif isinstance(output, torch.Tensor):
                embedding = output
            else:
                embedding = list(output.values())[0] if hasattr(output, 'values') else output[0]

            embeddings.append(embedding.cpu().numpy().squeeze())

    return np.array(embeddings)


def extract_acoustic_features(audio: np.ndarray) -> np.ndarray:
    """Extract hand-crafted acoustic features as fallback when HeAR unavailable.

    Returns a feature vector of fixed dimensionality.
    """
    features = []

    # F0 (fundamental frequency)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'), sr=SAMPLE_RATE,
        )
    f0_valid = f0[~np.isnan(f0)]
    if len(f0_valid) > 0:
        features.extend([np.mean(f0_valid), np.std(f0_valid), np.min(f0_valid),
                         np.max(f0_valid), np.max(f0_valid) - np.min(f0_valid)])
    else:
        features.extend([0, 0, 0, 0, 0])

    features.append(float(np.mean(voiced_flag)))

    # MFCCs (13 coefficients, mean + std = 26 features)
    mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13)
    for i in range(13):
        features.extend([float(np.mean(mfccs[i])), float(np.std(mfccs[i]))])

    # Spectral features
    spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE)
    spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=SAMPLE_RATE)
    spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE)
    features.extend([
        float(np.mean(spec_centroid)), float(np.std(spec_centroid)),
        float(np.mean(spec_bw)), float(np.std(spec_bw)),
        float(np.mean(spec_rolloff)), float(np.std(spec_rolloff)),
    ])

    # ZCR and RMS
    zcr = librosa.feature.zero_crossing_rate(audio)
    rms = librosa.feature.rms(y=audio)
    features.extend([
        float(np.mean(zcr)), float(np.std(zcr)),
        float(np.mean(rms)), float(np.std(rms)),
    ])

    # Tempo
    onset_env = librosa.onset.onset_strength(y=audio, sr=SAMPLE_RATE)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=SAMPLE_RATE)
    features.append(float(tempo[0]) if len(tempo) > 0 else 0)

    return np.array(features, dtype=np.float32)


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("HeAR Cry Classifier Training")
    print("=" * 60)

    # Configuration
    data_dir = PROJECT_ROOT / "data" / "raw" / "donate-a-cry"
    output_dir = PROJECT_ROOT / "models" / "linear_probes"
    embedding_cache_dir = PROJECT_ROOT / "models" / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_cache_dir.mkdir(parents=True, exist_ok=True)

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Step 1: Load dataset
    print("\n--- Step 1: Loading Dataset ---")
    samples = load_donate_a_cry(data_dir)
    if len(samples) == 0:
        print("ERROR: No samples found. Check data/raw/donate-a-cry/")
        return

    # Step 2: Try loading HeAR model for real embeddings
    print("\n--- Step 2: Loading HeAR Model ---")
    hear_model = None
    use_hear = False
    hf_token = os.environ.get("HF_TOKEN")

    try:
        from transformers import AutoModel
        print("Loading HeAR model: google/hear-pytorch")
        hear_model = AutoModel.from_pretrained(
            "google/hear-pytorch",
            token=hf_token,
            trust_remote_code=True,
        ).to(device)
        hear_model.eval()
        use_hear = True
        print("HeAR model loaded successfully")
    except Exception as e:
        print(f"Could not load HeAR: {e}")
        print("Falling back to acoustic feature extraction")

    # Step 3: Extract embeddings
    print("\n--- Step 3: Extracting Embeddings ---")
    cache_suffix = "hear" if use_hear else "acoustic"
    embeddings_cache = embedding_cache_dir / f"cry_embeddings_{cache_suffix}.npy"
    labels_cache = embedding_cache_dir / f"cry_labels_{cache_suffix}.npy"

    if embeddings_cache.exists() and labels_cache.exists():
        print(f"Loading cached embeddings from {embeddings_cache}")
        X = np.load(embeddings_cache)
        y = np.load(labels_cache)
        print(f"Loaded {len(X)} cached embeddings, dim={X.shape[1]}")
    else:
        audio_arrays = []
        labels = []

        print(f"Processing {len(samples)} audio files...")
        for sample in tqdm(samples, desc="Loading audio"):
            audio = load_and_preprocess_audio(sample["path"])
            audio_arrays.append(audio)
            labels.append(sample["label"])

        if use_hear:
            print("Extracting HeAR embeddings...")
            # Process in batches to avoid memory issues
            batch_size = 32
            all_embeddings = []
            for i in tqdm(range(0, len(audio_arrays), batch_size), desc="HeAR extraction"):
                batch = audio_arrays[i:i + batch_size]
                batch_emb = extract_hear_embeddings_batch(batch, hear_model, device)
                all_embeddings.append(batch_emb)
            X = np.vstack(all_embeddings)
        else:
            print("Extracting acoustic features...")
            X = np.array([
                extract_acoustic_features(audio)
                for audio in tqdm(audio_arrays, desc="Acoustic features")
            ])

        y = np.array(labels)

        # Cache embeddings
        np.save(embeddings_cache, X)
        np.save(labels_cache, y)
        print(f"Saved embeddings to {embeddings_cache} (shape: {X.shape})")

    # Free HeAR model memory
    if hear_model is not None:
        del hear_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Step 4: Train classifiers
    print("\n--- Step 4: Training Classifiers ---")
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Stratified 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    classifiers = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", multi_class="multinomial",
            random_state=seed,
        ),
        "SVM_RBF": SVC(
            kernel="rbf", C=10.0, gamma="scale", probability=True,
            random_state=seed,
        ),
        "SVM_Linear": SVC(
            kernel="linear", C=1.0, probability=True,
            random_state=seed,
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
        results[name] = {"mean_accuracy": mean_acc, "std_accuracy": std_acc}
        print(f"  {name}: {mean_acc:.4f} +/- {std_acc:.4f}")

        if mean_acc > best_score:
            best_score = mean_acc
            best_name = name

    print(f"\nBest classifier: {best_name} ({best_score:.4f})")

    # Train final model on all data
    print(f"\nTraining final {best_name} on all data...")
    best_clf = classifiers[best_name]
    best_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", best_clf),
    ])
    best_pipeline.fit(X, y)

    # Full training set evaluation for classification report
    y_pred = best_pipeline.predict(X)
    target_names = [CATEGORY_NAMES[i] for i in sorted(CATEGORY_NAMES.keys())]
    print("\nClassification Report (full dataset):")
    print(classification_report(y, y_pred, target_names=target_names))

    # Step 5: Save model
    print("\n--- Step 5: Saving Model ---")
    model_path = output_dir / "cry_classifier.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"Saved classifier to {model_path}")

    # Save metadata
    embedding_dim = X.shape[1]
    metadata = {
        "model_type": best_name,
        "embedding_source": "HeAR (google/hear-pytorch)" if use_hear else "Acoustic Features",
        "embedding_dim": embedding_dim,
        "num_classes": len(CRY_CATEGORIES),
        "classes": CRY_CATEGORIES,
        "cv_accuracy_mean": float(best_score),
        "cv_accuracy_std": float(results[best_name]["std_accuracy"]),
        "num_samples": len(y),
        "all_results": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
        "seed": seed,
    }

    metadata_path = output_dir / "cry_classifier_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    print("\n" + "=" * 60)
    print(f"DONE: Cry classifier trained")
    print(f"  Model: {best_name}")
    print(f"  Embedding: {'HeAR 512-dim' if use_hear else f'Acoustic {embedding_dim}-dim'}")
    print(f"  Accuracy: {best_score:.4f}")
    print(f"  Saved to: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
