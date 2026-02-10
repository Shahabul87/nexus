#!/usr/bin/env python3
"""
Bilirubin Regression from MedSigLIP Embeddings

Novel Task: Predict continuous bilirubin levels (mg/dL) from neonatal
skin images using frozen MedSigLIP embeddings + trained regression head.

Architecture:
    Frozen MedSigLIP encoder -> 1152-dim (or 768-dim) embeddings
    -> Linear(D, 256) -> ReLU -> Dropout -> Linear(256, 1)

Loss: Huber loss (robust to outliers in bilirubin measurements)
Metrics: MAE, RMSE, Pearson correlation, Bland-Altman analysis

Dataset: NeoJaundice (data/raw/neojaundice/)
    - 2,235 images with ground truth blood(mg/dL) column
    - Split: 70/15/15 train/val/test

HAI-DEF Model: MedSigLIP (google/medsiglip-448) - frozen encoder
"""

# Fix SSL certificate issues on macOS
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.model_selection import train_test_split

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))


# ---------------------------------------------------------------------------
# Regression Head Model
# ---------------------------------------------------------------------------

class BilirubinRegressorHead(nn.Module):
    """3-layer MLP regression head with BatchNorm for bilirubin prediction.

    Deeper architecture (3 layers with BatchNorm) compared to original 2-layer.
    Improves MAE by learning more complex feature interactions.
    """

    def __init__(self, input_dim: int = 1152, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        # Wider first layer, then taper
        mid_dim = hidden_dim * 2  # 512
        self.net = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),  # Less dropout in later layers
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def load_jaundice_regression_data(
    data_dir: Path,
) -> Tuple[list, np.ndarray]:
    """
    Load NeoJaundice dataset with continuous bilirubin labels.

    Returns:
        image_paths: List of Path objects
        bilirubin_values: numpy array of bilirubin levels (mg/dL)
    """
    csv_path = data_dir / "chd_jaundice_published_2.csv"
    images_dir = data_dir / "images"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    df = pd.read_csv(csv_path)
    print(f"Loaded CSV with {len(df)} rows")
    print(f"Bilirubin range: {df['blood(mg/dL)'].min():.1f} - {df['blood(mg/dL)'].max():.1f} mg/dL")
    print(f"Bilirubin mean: {df['blood(mg/dL)'].mean():.1f}, std: {df['blood(mg/dL)'].std():.1f}")

    image_paths = []
    bilirubin_values = []

    for _, row in df.iterrows():
        img_path = images_dir / row["image_idx"]
        if img_path.exists():
            image_paths.append(img_path)
            bilirubin_values.append(float(row["blood(mg/dL)"]))

    print(f"Found {len(image_paths)} images with ground truth bilirubin values")
    return image_paths, np.array(bilirubin_values, dtype=np.float32)


# ---------------------------------------------------------------------------
# Embedding Extraction
# ---------------------------------------------------------------------------

def extract_or_load_embeddings(
    image_paths: list,
    cache_path: Path,
    batch_size: int = 8,
) -> np.ndarray:
    """Extract embeddings from MedSigLIP, or load from cache if available."""
    if cache_path.exists():
        print(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    from train_linear_probes import EmbeddingExtractor

    print("Extracting MedSigLIP embeddings (this may take a while)...")
    extractor = EmbeddingExtractor()
    embeddings = extractor.extract_batch_embeddings(image_paths, batch_size=batch_size)

    # Cache for future use
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    print(f"Embeddings cached to {cache_path}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    return embeddings


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_regressor(
    embeddings: np.ndarray,
    bilirubin: np.ndarray,
    hidden_dim: int = 256,
    dropout: float = 0.3,
    lr: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 64,
    patience: int = 15,
    random_state: int = 42,
) -> Tuple[BilirubinRegressorHead, Dict]:
    """
    Train bilirubin regression head on frozen MedSigLIP embeddings.

    Returns:
        model: Trained BilirubinRegressorHead
        metrics: Dictionary of evaluation metrics
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = embeddings.shape[1]

    # 70/15/15 split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        embeddings, bilirubin, test_size=0.15, random_state=random_state,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15 / 0.85, random_state=random_state,
    )

    print(f"\nSplit sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"Embedding dimension: {input_dim}")

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Create model
    model = BilirubinRegressorHead(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    criterion = nn.HuberLoss(delta=2.0)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_mae": []}

    for epoch in range(epochs):
        # Training
        model.train()
        indices = torch.randperm(len(X_train_t))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            x_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
            val_mae = torch.abs(val_pred - y_val_t).mean().item()

        scheduler.step(epoch + val_loss * 0)  # CosineAnnealingWarmRestarts uses epoch

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val MAE: {val_mae:.2f} mg/dL"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t).cpu().numpy()
        test_actual = y_test_t.cpu().numpy()

    mae = float(np.abs(test_pred - test_actual).mean())
    rmse = float(np.sqrt(np.mean((test_pred - test_actual) ** 2)))
    pearson_r, pearson_p = stats.pearsonr(test_pred, test_actual)

    # Bland-Altman analysis
    diff = test_pred - test_actual
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff))
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    metrics = {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "pearson_r": round(float(pearson_r), 4),
        "pearson_p": float(pearson_p),
        "bland_altman": {
            "mean_diff": round(mean_diff, 3),
            "std_diff": round(std_diff, 3),
            "loa_upper": round(loa_upper, 3),
            "loa_lower": round(loa_lower, 3),
        },
        "test_size": len(test_actual),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "epochs_trained": len(history["train_loss"]),
        "best_val_loss": round(best_val_loss, 4),
        "bilirubin_range": {
            "min": round(float(bilirubin.min()), 1),
            "max": round(float(bilirubin.max()), 1),
            "mean": round(float(bilirubin.mean()), 1),
            "std": round(float(bilirubin.std()), 1),
        },
        "history": {
            "train_loss": [round(v, 4) for v in history["train_loss"]],
            "val_loss": [round(v, 4) for v in history["val_loss"]],
            "val_mae": [round(v, 3) for v in history["val_mae"]],
        },
    }

    print(f"\n{'=' * 50}")
    print("TEST SET RESULTS")
    print(f"{'=' * 50}")
    print(f"MAE:       {mae:.3f} mg/dL")
    print(f"RMSE:      {rmse:.3f} mg/dL")
    print(f"Pearson r: {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"Bland-Altman: mean diff={mean_diff:.3f}, 95% LoA=[{loa_lower:.3f}, {loa_upper:.3f}]")

    return model, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train bilirubin regression from MedSigLIP embeddings")
    parser.add_argument("--data-dir", type=str, default="data/raw/neojaundice")
    parser.add_argument("--output-dir", type=str, default="models/linear_probes")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    args = parser.parse_args()

    print("=" * 60)
    print("BILIRUBIN REGRESSION FROM MEDSIGLIP EMBEDDINGS")
    print("Novel Task: Continuous bilirubin prediction from skin images")
    print("=" * 60)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load dataset
    print("\n[Step 1/4] Loading NeoJaundice dataset...")
    image_paths, bilirubin_values = load_jaundice_regression_data(data_dir)

    # Step 2: Extract embeddings
    print("\n[Step 2/4] Extracting MedSigLIP embeddings...")
    cache_path = output_dir / "jaundice_regression_embeddings.npy"
    embeddings = extract_or_load_embeddings(
        image_paths, cache_path, batch_size=args.embedding_batch_size,
    )

    # Also cache bilirubin values
    bili_cache = output_dir / "jaundice_regression_bilirubin.npy"
    if not bili_cache.exists():
        np.save(bili_cache, bilirubin_values)

    # Step 3: Train regression head
    print("\n[Step 3/4] Training bilirubin regression head...")
    model, metrics = train_regressor(
        embeddings, bilirubin_values,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )

    # Step 4: Save model and metrics
    print("\n[Step 4/4] Saving model and metrics...")
    model_path = output_dir / "bilirubin_regressor.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": metrics["input_dim"],
        "hidden_dim": metrics["hidden_dim"],
        "metrics": metrics,
    }, model_path)
    print(f"Model saved: {model_path}")

    metrics_path = output_dir / "bilirubin_regression_results.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("BILIRUBIN REGRESSION TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Model:         {model_path}")
    print(f"MAE:           {metrics['mae']:.3f} mg/dL")
    print(f"RMSE:          {metrics['rmse']:.3f} mg/dL")
    print(f"Pearson r:     {metrics['pearson_r']:.4f}")
    print(f"Epochs:        {metrics['epochs_trained']}")
    print(f"Input dim:     {metrics['input_dim']}")


if __name__ == "__main__":
    main()
