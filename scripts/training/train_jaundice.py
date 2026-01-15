#!/usr/bin/env python3
"""
Train Jaundice Detection Model

Fine-tunes a vision model on the NeoJaundice dataset for
bilirubin estimation and jaundice severity classification.
"""

# Fix SSL certificate issues on macOS
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from typing import Tuple, Dict, Optional


class JaundiceDataset(Dataset):
    """Dataset for jaundice detection from neonatal images."""

    # Bilirubin thresholds for classification
    SEVERITY_THRESHOLDS = {
        "none": 5.0,
        "mild": 10.0,
        "moderate": 15.0,
        "severe": 20.0,
    }

    def __init__(
        self,
        data_dir: Path,
        csv_path: Optional[Path] = None,
        transform: Optional[transforms.Compose] = None,
        task: str = "classification",  # "classification" or "regression"
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Path to neojaundice dataset
            csv_path: Path to metadata CSV
            transform: Image transforms
            task: "classification" for severity, "regression" for bilirubin
        """
        self.data_dir = Path(data_dir)
        self.transform = transform or self._default_transform()
        self.task = task
        self.samples = []

        # Load metadata
        csv_path = csv_path or (self.data_dir / "chd_jaundice_published_2.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f"Loaded metadata with {len(df)} entries")

            # Process each row
            for _, row in df.iterrows():
                img_name = row["image_idx"]
                img_path = self.data_dir / "images" / img_name

                if not img_path.exists():
                    continue

                bilirubin = row["blood(mg/dL)"]
                treatment = row["Treatment"]

                # Determine severity class
                if bilirubin < self.SEVERITY_THRESHOLDS["none"]:
                    severity = 0  # None
                elif bilirubin < self.SEVERITY_THRESHOLDS["mild"]:
                    severity = 1  # Mild
                elif bilirubin < self.SEVERITY_THRESHOLDS["moderate"]:
                    severity = 2  # Moderate
                elif bilirubin < self.SEVERITY_THRESHOLDS["severe"]:
                    severity = 3  # Severe
                else:
                    severity = 4  # Critical

                self.samples.append({
                    "path": img_path,
                    "bilirubin": bilirubin,
                    "severity": severity,
                    "treatment": treatment,
                    "patient_id": row["patient_id"],
                })

        print(f"Loaded {len(self.samples)} samples")
        severity_counts = {}
        for s in self.samples:
            severity_counts[s["severity"]] = severity_counts.get(s["severity"], 0) + 1
        print(f"Severity distribution: {severity_counts}")

    def _default_transform(self) -> transforms.Compose:
        """Default image transforms."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.task == "regression":
            label = torch.tensor(sample["bilirubin"], dtype=torch.float32)
        else:
            label = sample["severity"]

        return image, label


class JaundiceClassifier(nn.Module):
    """Multi-task model for jaundice detection."""

    def __init__(
        self,
        backbone: str = "resnet18",
        num_severity_classes: int = 5,
        pretrained: bool = True,
    ):
        super().__init__()

        # Load pretrained backbone
        if backbone == "resnet18":
            self.backbone = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "resnet50":
            self.backbone = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.num_features = num_features

        # Classification head (severity)
        self.severity_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_severity_classes),
        )

        # Regression head (bilirubin)
        self.bilirubin_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        severity = self.severity_head(features)
        bilirubin = self.bilirubin_head(features).squeeze(-1)
        return severity, bilirubin


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion_cls: nn.Module,
    criterion_reg: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    alpha: float = 0.5,
) -> Dict[str, float]:
    """Train for one epoch with multi-task loss."""
    model.train()
    running_loss = 0.0
    cls_correct = 0
    total = 0
    mae_sum = 0.0

    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)

        # For multi-task, labels is severity, we need bilirubin too
        severity_labels = labels.to(device)

        optimizer.zero_grad()
        severity_pred, bilirubin_pred = model(images)

        # Classification loss
        cls_loss = criterion_cls(severity_pred, severity_labels)

        # Combined loss (classification only for now)
        loss = cls_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = severity_pred.max(1)
        total += severity_labels.size(0)
        cls_correct += predicted.eq(severity_labels).sum().item()

    return {
        "loss": running_loss / len(dataloader),
        "accuracy": cls_correct / total,
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion_cls: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            severity_labels = labels.to(device)

            severity_pred, _ = model(images)
            loss = criterion_cls(severity_pred, severity_labels)

            running_loss += loss.item()
            _, predicted = severity_pred.max(1)
            total += severity_labels.size(0)
            correct += predicted.eq(severity_labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(severity_labels.cpu().numpy())

    # Per-class accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    per_class_acc = {}
    for c in range(5):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class_acc[c] = (all_preds[mask] == all_labels[mask]).mean()

    return {
        "loss": running_loss / len(dataloader),
        "accuracy": correct / total,
        "per_class_accuracy": per_class_acc,
    }


def main():
    """Main training function."""
    # Configuration
    config = {
        "data_dir": "data/raw/neojaundice",
        "backbone": "resnet18",
        "batch_size": 16,  # Reduced for CPU training
        "epochs": 5,  # Reduced for faster initial testing
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "train_split": 0.8,
        "seed": 42,
    }

    # Set seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    print("Loading dataset...")
    full_dataset = JaundiceDataset(
        config["data_dir"],
        transform=train_transform,
        task="classification"
    )

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
    print(f"Creating model with {config['backbone']} backbone...")
    model = JaundiceClassifier(backbone=config["backbone"]).to(device)

    # Loss and optimizer
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
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
        train_metrics = train_epoch(
            model, train_loader, criterion_cls, criterion_reg, optimizer, device
        )

        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion_cls, device)

        scheduler.step()

        # Log
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
        })

        # Save best model
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]

            checkpoint_dir = Path("models/checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_metrics["accuracy"],
                "config": config,
            }, checkpoint_dir / "jaundice_best.pt")

            print(f"  Saved best model with accuracy: {best_acc:.4f}")

    # Save training history
    history_path = Path("models/checkpoints/jaundice_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Best Val Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
