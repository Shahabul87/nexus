#!/usr/bin/env python3
"""
Train Anemia Detection Model

Fine-tunes a vision model on the Eyes-Defy-Anemia dataset for
improved anemia detection from conjunctiva images.
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
from datetime import datetime
from typing import Tuple, Dict, Optional


class AnemiaDataset(Dataset):
    """Dataset for anemia detection from conjunctiva images."""

    def __init__(
        self,
        data_dir: Path,
        transform: Optional[transforms.Compose] = None,
        split: str = "train",
        labels_file: Optional[Path] = None,
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Path to eyes-defy-anemia dataset
            transform: Image transforms
            split: "train", "val", or "test"
            labels_file: Optional CSV/JSON file with columns: filename,is_anemic
        """
        self.data_dir = Path(data_dir)
        self.transform = transform or self._default_transform()
        self.samples = []
        self.labels_dict = {}

        # Try to load labels from file
        if labels_file and Path(labels_file).exists():
            self.labels_dict = self._load_labels_file(labels_file)
            print(f"Loaded {len(self.labels_dict)} labels from {labels_file}")
        else:
            # Check for default labels file locations
            for default_path in [
                self.data_dir / "labels.csv",
                self.data_dir / "labels.json",
                self.data_dir.parent / "anemia_labels.csv",
            ]:
                if default_path.exists():
                    self.labels_dict = self._load_labels_file(default_path)
                    print(f"Loaded {len(self.labels_dict)} labels from {default_path}")
                    break

        # Load images from India and Italy folders
        # Images are in numbered subdirectories
        for region in ["India", "Italy"]:
            region_dir = self.data_dir / region
            if not region_dir.exists():
                continue

            # Search recursively for jpg files
            for img_path in region_dir.rglob("*.jpg"):
                # Parse label from folder structure or filename
                filename = img_path.stem
                parent_folder = img_path.parent.name

                # Use folder number and filename patterns to determine label
                is_anemic = self._parse_label(filename, region, parent_folder)

                self.samples.append({
                    "path": img_path,
                    "label": 1 if is_anemic else 0,
                    "region": region,
                })

        print(f"Loaded {len(self.samples)} samples")
        print(f"  Anemic: {sum(s['label'] for s in self.samples)}")
        print(f"  Healthy: {sum(1 - s['label'] for s in self.samples)}")

    def _parse_label(self, filename: str, region: str, parent_folder: str = "") -> bool:
        """
        Parse anemia label from filename, folder structure, or labels file.

        Priority:
        1. Look up in self.labels_dict if loaded from CSV/JSON
        2. Check for explicit labels in filename
        3. WARN and use pseudo-labels as last resort

        WARNING: Training with pseudo-labels produces INVALID models!
        """
        filename_lower = filename.lower()
        full_path = f"{region}/{parent_folder}/{filename}"

        # Check labels dictionary first (loaded from CSV/JSON)
        if hasattr(self, 'labels_dict') and self.labels_dict:
            # Try various key formats
            for key in [full_path, filename, f"{parent_folder}/{filename}"]:
                if key in self.labels_dict:
                    return self.labels_dict[key]

        # Check for explicit labels in filename
        if 'anemic' in filename_lower or 'anemia' in filename_lower:
            return True
        if 'healthy' in filename_lower or 'normal' in filename_lower:
            return False

        # PSEUDO-LABELS: Only used when no real labels available
        # WARNING: This makes the model INVALID for actual use!
        if not hasattr(self, '_warned_pseudo_labels'):
            print("\n" + "="*70)
            print("WARNING: Using PSEUDO-LABELS (folder number based)")
            print("Models trained with pseudo-labels are NOT valid for real use!")
            print("Please provide a labels.csv file with columns: filename,is_anemic")
            print("="*70 + "\n")
            self._warned_pseudo_labels = True

        # Use folder number as pseudo-label (NOT valid for real training)
        try:
            folder_num = int(parent_folder)
            return folder_num % 2 == 1
        except (ValueError, TypeError):
            pass

        # Use hash for consistent but fake labels
        return hash(filename) % 2 == 1

    def _load_labels_file(self, filepath: Path) -> Dict[str, bool]:
        """
        Load labels from CSV or JSON file.

        Expected CSV format:
            filename,is_anemic
            1/image.jpg,1
            2/image.jpg,0

        Expected JSON format:
            {"1/image.jpg": true, "2/image.jpg": false}
        """
        filepath = Path(filepath)
        labels = {}

        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
                for k, v in data.items():
                    labels[k] = bool(v) if isinstance(v, (int, str)) else v
        elif filepath.suffix == '.csv':
            df = pd.read_csv(filepath)
            # Support various column names
            fname_col = next((c for c in df.columns if 'file' in c.lower() or 'name' in c.lower() or 'path' in c.lower()), df.columns[0])
            label_col = next((c for c in df.columns if 'anemic' in c.lower() or 'label' in c.lower() or 'class' in c.lower()), df.columns[1])

            for _, row in df.iterrows():
                filename = str(row[fname_col])
                label_val = row[label_col]
                # Convert various formats to bool
                if isinstance(label_val, str):
                    labels[filename] = label_val.lower() in ('1', 'true', 'yes', 'anemic')
                else:
                    labels[filename] = bool(label_val)

        return labels

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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, sample["label"]


class AnemiaClassifier(nn.Module):
    """Fine-tuned classifier for anemia detection."""

    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 2,
        pretrained: bool = True,
    ):
        super().__init__()

        # Load pretrained backbone
        if backbone == "resnet18":
            self.backbone = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
        elif backbone == "resnet50":
            self.backbone = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
        elif backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(num_features, num_classes)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


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

    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
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
) -> Tuple[float, float, Dict]:
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
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sensitivity": recall,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
    }

    return running_loss / len(dataloader), correct / total, metrics


def main():
    """Main training function."""
    # Configuration
    config = {
        "data_dir": "data/raw/eyes-defy-anemia",
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
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
    full_dataset = AnemiaDataset(config["data_dir"], transform=train_transform)

    # Split dataset
    train_size = int(config["train_split"] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config["seed"])
    )

    # Update val dataset transform
    val_dataset.dataset.transform = val_transform

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
    model = AnemiaClassifier(backbone=config["backbone"]).to(device)

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
    best_f1 = 0.0
    history = []

    print("\nStarting training...")
    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Evaluate
        val_loss, val_acc, metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        # Log
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            **metrics,
        })

        # Save best model
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_acc = val_acc

            checkpoint_dir = Path("models/checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "metrics": metrics,
                "config": config,
            }, checkpoint_dir / "anemia_best.pt")

            print(f"  Saved best model with F1: {best_f1:.4f}")

    # Save training history
    history_path = Path("models/checkpoints/anemia_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Best Val Accuracy: {best_acc:.4f}")
    print(f"Best F1 Score: {best_f1:.4f}")


if __name__ == "__main__":
    main()
