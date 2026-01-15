#!/usr/bin/env python3
"""
download_datasets.py - Download all NEXUS datasets

Usage:
    python scripts/download_datasets.py

Requirements:
    - Kaggle API configured (~/.kaggle/kaggle.json)
    - git-lfs installed (for HuggingFace datasets)
"""

import os
import subprocess
import sys
from pathlib import Path

# Base data directory
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def run_command(cmd: str, cwd: str = None) -> bool:
    """Run shell command."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.returncode == 0


def check_kaggle() -> bool:
    """Check if Kaggle CLI is configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("""
ERROR: Kaggle API not configured!

To configure:
1. Go to https://www.kaggle.com/settings
2. Click "Create New Token" under API section
3. Save kaggle.json to ~/.kaggle/kaggle.json
4. Run: chmod 600 ~/.kaggle/kaggle.json
""")
        return False
    return True


def download_kaggle_datasets():
    """Download Kaggle datasets."""
    if not check_kaggle():
        print("Skipping Kaggle datasets - API not configured")
        return False

    datasets = [
        ("harshwardhanfartale/eyes-defy-anemia", "eyes-defy-anemia"),
        ("nahiyan1402/anemiadataset", "anemia-dataset-2"),
    ]

    raw_dir = DATA_DIR / "raw"
    raw_dir.mkdir(exist_ok=True)

    for dataset_id, folder_name in datasets:
        dest = raw_dir / folder_name
        if dest.exists():
            print(f"Skipping {folder_name} (already exists)")
            continue

        print(f"\nDownloading {dataset_id}...")
        zip_name = dataset_id.split("/")[-1]

        if not run_command(f"kaggle datasets download -d {dataset_id} -p {raw_dir}"):
            continue

        # Unzip
        zip_file = raw_dir / f"{zip_name}.zip"
        if zip_file.exists():
            dest.mkdir(exist_ok=True)
            run_command(f"unzip -q '{zip_file}' -d '{dest}'")
            zip_file.unlink()
            print(f"Downloaded and extracted to {dest}")

    return True


def download_huggingface_datasets():
    """Download HuggingFace datasets."""
    raw_dir = DATA_DIR / "raw"
    cryceleb_dir = raw_dir / "cryceleb"

    if cryceleb_dir.exists():
        print("Skipping CryCeleb (already exists)")
        return True

    print("\nDownloading CryCeleb 2023 from HuggingFace...")

    # Check git-lfs
    if not run_command("git lfs version"):
        print("Installing git-lfs...")
        run_command("git lfs install")

    raw_dir.mkdir(exist_ok=True)
    success = run_command(
        f"git clone https://huggingface.co/datasets/Ubenwa/CryCeleb2023 {cryceleb_dir}"
    )

    if success:
        print(f"Downloaded to {cryceleb_dir}")
    return success


def download_github_datasets():
    """Download GitHub datasets."""
    raw_dir = DATA_DIR / "raw"
    icsd_dir = raw_dir / "icsd"

    if icsd_dir.exists():
        print("Skipping ICSD (already exists)")
        return True

    print("\nDownloading ICSD dataset from GitHub...")
    raw_dir.mkdir(exist_ok=True)
    success = run_command(f"git clone https://github.com/QingyuLiu0521/ICSD/ {icsd_dir}")

    if success:
        print(f"Downloaded to {icsd_dir}")
    return success


def print_manual_downloads():
    """Print instructions for manual downloads."""
    print("""
================================================================================
MANUAL DOWNLOADS REQUIRED
================================================================================

The following datasets require manual download:

1. NeoJaundice (REQUIRED - PRIMARY JAUNDICE DATASET):
   URL: https://springernature.figshare.com/articles/dataset/NeoJaundice/22302559
   Save to: data/raw/neojaundice/

2. NJN Dataset (Recommended):
   URL: https://sites.google.com/view/neonataljaundice
   Save to: data/raw/njn/

3. Harvard Conjunctiva (Optional):
   URL: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/L4MDKC
   Save to: data/raw/harvard-conjunctiva/

4. Baby Chillanto (CRITICAL - REQUEST ACCESS):
   Contact: CONACYT/INAOE Mexico
   This is the ONLY public dataset with birth asphyxia labels!
   Save to: data/raw/baby-chillanto/

   See DATASET_ACQUISITION_GUIDE.md for email template to request access.

================================================================================
""")


def verify_datasets():
    """Verify downloaded datasets."""
    print("\n" + "=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)

    raw_dir = DATA_DIR / "raw"

    datasets = [
        ("eyes-defy-anemia", "Anemia Detection (Primary)", True),
        ("neojaundice", "Jaundice Detection (Primary)", True),
        ("njn", "Jaundice Detection (Secondary)", False),
        ("cryceleb", "Cry Audio Samples", False),
        ("icsd", "Cry Detection", False),
        ("baby-chillanto", "Birth Asphyxia (CRITICAL)", True),
    ]

    missing_critical = []

    for folder, name, critical in datasets:
        path = raw_dir / folder
        exists = path.exists() and any(path.iterdir()) if path.exists() else False
        status = "FOUND" if exists else "MISSING"
        indicator = "  " if not critical else ("" if exists else "")

        print(f"  {indicator} {name}: {status}")
        print(f"       Path: {path}")

        if critical and not exists:
            missing_critical.append(name)

    if missing_critical:
        print(f"\n WARNING: Missing critical datasets:")
        for ds in missing_critical:
            print(f"   - {ds}")
        print("\n   Please download these manually before proceeding.")
        print("   See DATASET_ACQUISITION_GUIDE.md for instructions.")
    else:
        print("\n All critical datasets are available!")

    return len(missing_critical) == 0


def create_directory_structure():
    """Create expected directory structure."""
    dirs = [
        DATA_DIR / "raw",
        DATA_DIR / "prepared" / "anemia" / "anemic",
        DATA_DIR / "prepared" / "anemia" / "healthy",
        DATA_DIR / "prepared" / "jaundice" / "jaundiced",
        DATA_DIR / "prepared" / "jaundice" / "normal",
        DATA_DIR / "prepared" / "cry" / "asphyxia",
        DATA_DIR / "prepared" / "cry" / "normal",
        DATA_DIR / "test",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    print("Created directory structure in data/")


def main():
    """Main download function."""
    print("=" * 60)
    print("NEXUS Dataset Downloader")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}")

    # Create directory structure
    create_directory_structure()

    # Download automated datasets
    print("\n--- Automated Downloads ---")
    download_kaggle_datasets()
    download_huggingface_datasets()
    download_github_datasets()

    # Print manual download instructions
    print_manual_downloads()

    # Verify
    all_ready = verify_datasets()

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. Download manual datasets (see above)
2. Run: python scripts/prepare_datasets.py
3. Run: python scripts/validate_models.py
""")

    return 0 if all_ready else 1


if __name__ == "__main__":
    sys.exit(main())
