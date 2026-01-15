#!/usr/bin/env python3
"""
Train Cry Analysis Model

Trains a classifier on cry audio data for detecting
abnormal cry patterns and cry type classification.
"""

# Fix SSL certificate issues on macOS
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import librosa
from typing import Tuple, Dict, List, Optional


class CryDataset(Dataset):
    """Dataset for infant cry audio classification."""

    CRY_CATEGORIES = {
        "belly_pain": 0,
        "burping": 1,
        "discomfort": 2,
        "hungry": 3,
        "tired": 4,
    }

    def __init__(
        self,
        data_dirs: List[Path],
        sample_rate: int = 16000,
        duration: float = 3.0,
        n_mels: int = 128,
        task: str = "cry_type",  # "cry_type" or "is_cry"
    ):
        """
        Initialize dataset.

        Args:
            data_dirs: List of data directories
            sample_rate: Target sample rate
            duration: Audio clip duration in seconds
            n_mels: Number of mel bands
            task: Classification task
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.task = task
        self.samples = []

        for data_dir in data_dirs:
            data_dir = Path(data_dir)
            if not data_dir.exists():
                continue

            self._load_from_dir(data_dir)

        print(f"Loaded {len(self.samples)} samples")

    def _load_from_dir(self, data_dir: Path) -> None:
        """Load samples from directory."""
        # Check for donate-a-cry structure
        corpus_dir = data_dir / "donateacry_corpus_cleaned_and_updated_data"
        if corpus_dir.exists():
            for category in self.CRY_CATEGORIES.keys():
                cat_dir = corpus_dir / category
                if cat_dir.exists():
                    for audio_path in cat_dir.glob("*.wav"):
                        self.samples.append({
                            "path": audio_path,
                            "label": self.CRY_CATEGORIES[category],
                            "category": category,
                            "is_cry": True,
                        })

        # Check for infant-cry-dataset structure
        cry_dir = data_dir / "cry"
        not_cry_dir = data_dir / "not_cry"

        if cry_dir.exists():
            for audio_path in cry_dir.glob("*.wav"):
                self.samples.append({
                    "path": audio_path,
                    "label": 2,  # discomfort as default
                    "category": "cry",
                    "is_cry": True,
                })

        if not_cry_dir.exists():
            for audio_path in not_cry_dir.glob("*.wav"):
                self.samples.append({
                    "path": audio_path,
                    "label": -1,
                    "category": "not_cry",
                    "is_cry": False,
                })

        # Check for CryCeleb structure
        audio_dir = data_dir / "audio"
        if audio_dir.exists():
            for split_dir in audio_dir.iterdir():
                if split_dir.is_dir():
                    for audio_path in split_dir.glob("*.wav"):
                        self.samples.append({
                            "path": audio_path,
                            "label": 2,
                            "category": "cry",
                            "is_cry": True,
                        })

    def _load_audio(self, path: Path) -> np.ndarray:
        """Load and preprocess audio."""
        try:
            audio, _ = librosa.load(path, sr=self.sample_rate, duration=self.duration)

            # Pad if too short
            target_length = int(self.sample_rate * self.duration)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                audio = audio[:target_length]

            return audio
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return np.zeros(int(self.sample_rate * self.duration))

    def _extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel spectrogram features."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            fmax=8000,
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)

        return mel_spec_db

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]

        # Load audio
        audio = self._load_audio(sample["path"])

        # Extract mel spectrogram
        mel_spec = self._extract_mel_spectrogram(audio)

        # Convert to tensor [1, n_mels, time]
        mel_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)

        if self.task == "is_cry":
            label = 1 if sample["is_cry"] else 0
        else:
            label = max(0, sample["label"])  # Ensure non-negative

        return mel_tensor, label


class CryClassifier(nn.Module):
    """CNN classifier for cry audio."""

    def __init__(
        self,
        n_mels: int = 128,
        num_classes: int = 5,
    ):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_layers(x)
        output = self.classifier(features)
        return output

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification."""
        features = self.conv_layers(x)
        return features.view(features.size(0), -1)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for audio, labels in tqdm(dataloader, desc="Training"):
        audio = audio.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(audio)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), correct / total


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for audio, labels in tqdm(dataloader, desc="Evaluating"):
            audio = audio.to(device)
            labels = labels.to(device)

            outputs = model(audio)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), correct / total


def main():
    """Main training function."""
    # Configuration
    config = {
        "data_dirs": [
            "data/raw/donate-a-cry",
            "data/raw/infant-cry-dataset",
            "data/raw/cryceleb",
        ],
        "sample_rate": 16000,
        "duration": 3.0,
        "n_mels": 128,
        "batch_size": 16,  # Reduced for CPU training
        "epochs": 5,  # Reduced for faster initial testing
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "train_split": 0.8,
        "seed": 42,
        "task": "cry_type",
    }

    # Set seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print("Loading dataset...")
    full_dataset = CryDataset(
        [Path(d) for d in config["data_dirs"]],
        sample_rate=config["sample_rate"],
        duration=config["duration"],
        n_mels=config["n_mels"],
        task=config["task"],
    )

    if len(full_dataset) == 0:
        print("No samples found. Please check data directories.")
        return

    # Filter to only samples with valid labels for cry_type task
    if config["task"] == "cry_type":
        valid_samples = [s for s in full_dataset.samples if s["label"] >= 0]
        full_dataset.samples = valid_samples
        print(f"Filtered to {len(valid_samples)} valid samples")

    # Determine number of classes
    unique_labels = set(s["label"] for s in full_dataset.samples)
    num_classes = len(unique_labels)
    print(f"Number of classes: {num_classes}")

    # Split dataset
    train_size = int(config["train_split"] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config["seed"])
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,  # Use 0 for CPU training to avoid multiprocessing issues
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Model
    print("Creating model...")
    model = CryClassifier(
        n_mels=config["n_mels"],
        num_classes=num_classes,
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    # Training loop
    best_acc = 0.0
    history = []

    print("\nStarting training...")
    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        # Log
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc

            checkpoint_dir = Path("models/checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "config": config,
                "num_classes": num_classes,
            }, checkpoint_dir / "cry_best.pt")

            print(f"  Saved best model with accuracy: {best_acc:.4f}")

    # Save training history
    history_path = Path("models/checkpoints/cry_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Best Val Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
