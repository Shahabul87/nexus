#!/usr/bin/env python3
"""
MedAssist CHW - HAI-DEF Model Downloader

Downloads and caches HAI-DEF models from Hugging Face.
Run: python models/download_models.py

Requirements:
- huggingface_hub
- transformers
- Accept HAI-DEF terms of use on Hugging Face
"""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

# Model configurations
MODELS = {
    "medgemma-4b": {
        "repo_id": "google/medgemma-4b-it",
        "description": "MedGemma 4B Instruction-tuned - Medical reasoning and text",
        "size_gb": 8.0,
        "required": True,
    },
    "medgemma-27b": {
        "repo_id": "google/medgemma-27b-text-it",
        "description": "MedGemma 27B Text - Advanced medical reasoning",
        "size_gb": 54.0,
        "required": False,
    },
    "medsiglib": {
        "repo_id": "google/medsiglib",
        "description": "MedSigLIP - Medical image encoder",
        "size_gb": 1.6,
        "required": True,
    },
    "hear": {
        "repo_id": "google/hear",
        "description": "HeAR - Health acoustic representations",
        "size_gb": 0.35,
        "required": True,
    },
    "derm-foundation": {
        "repo_id": "google/derm-foundation",
        "description": "Derm Foundation - Dermatology image encoder",
        "size_gb": 0.4,
        "required": True,
    },
    "cxr-foundation": {
        "repo_id": "google/cxr-foundation",
        "description": "CXR Foundation - Chest X-ray encoder",
        "size_gb": 0.5,
        "required": False,
    },
    "path-foundation": {
        "repo_id": "google/path-foundation",
        "description": "Path Foundation - Histopathology encoder",
        "size_gb": 0.4,
        "required": False,
    },
}


def display_models():
    """Display available models in a table."""
    table = Table(title="Available HAI-DEF Models")
    table.add_column("Model", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Size", style="yellow")
    table.add_column("Required", style="green")

    for name, config in MODELS.items():
        table.add_row(
            name,
            config["description"],
            f"{config['size_gb']:.1f} GB",
            "Yes" if config["required"] else "No",
        )

    console.print(table)


def download_model(model_name: str, cache_dir: Path, force: bool = False) -> bool:
    """Download a single model from Hugging Face."""
    if model_name not in MODELS:
        console.print(f"[red]Unknown model: {model_name}[/red]")
        return False

    config = MODELS[model_name]
    model_dir = cache_dir / model_name

    if model_dir.exists() and not force:
        console.print(f"[yellow]Model {model_name} already exists. Use --force to re-download.[/yellow]")
        return True

    console.print(f"\n[cyan]Downloading {model_name}...[/cyan]")
    console.print(f"  Repository: {config['repo_id']}")
    console.print(f"  Size: ~{config['size_gb']:.1f} GB")

    try:
        # Download the model
        snapshot_download(
            repo_id=config["repo_id"],
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
        console.print(f"[green]Successfully downloaded {model_name}[/green]")
        return True

    except Exception as e:
        console.print(f"[red]Failed to download {model_name}: {e}[/red]")
        console.print("[yellow]Make sure you have:")
        console.print("  1. Logged in: huggingface-cli login")
        console.print("  2. Accepted HAI-DEF terms of use on Hugging Face[/yellow]")
        return False


def download_all(cache_dir: Path, required_only: bool = False, force: bool = False):
    """Download all models."""
    models_to_download = [
        name for name, config in MODELS.items()
        if not required_only or config["required"]
    ]

    console.print(f"\n[cyan]Downloading {len(models_to_download)} models...[/cyan]")

    total_size = sum(MODELS[m]["size_gb"] for m in models_to_download)
    console.print(f"Total download size: ~{total_size:.1f} GB\n")

    success_count = 0
    for model_name in models_to_download:
        if download_model(model_name, cache_dir, force):
            success_count += 1

    console.print(f"\n[green]Downloaded {success_count}/{len(models_to_download)} models[/green]")


def main():
    parser = argparse.ArgumentParser(
        description="Download HAI-DEF models for MedAssist CHW"
    )
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of models to download (default: all required)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./models/cache",
        help="Directory to cache downloaded models",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all models (including optional)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if exists",
    )

    args = parser.parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.list:
        display_models()
        return

    if args.models:
        # Download specific models
        model_list = [m.strip() for m in args.models.split(",")]
        for model_name in model_list:
            download_model(model_name, cache_dir, args.force)
    else:
        # Download all required (or all if --all flag)
        download_all(cache_dir, required_only=not args.all, force=args.force)


if __name__ == "__main__":
    main()
