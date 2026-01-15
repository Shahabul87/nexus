# NEXUS Dataset Acquisition Guide

## Quick Reference

| Dataset | Purpose | Size | Access | Priority |
|---------|---------|------|--------|----------|
| **Eyes-Defy-Anemia** | Anemia detection | 218 images | Public (Kaggle) | P0 |
| **NeoJaundice** | Jaundice detection | 2,235 images | Public (Figshare) | P0 |
| **NJN Dataset** | Jaundice detection | 670 images | Public (Google Sites) | P1 |
| **Baby Chillanto** | Cry/asphyxia | 2,268 samples | Request access | P0 |
| **CryCeleb 2023** | Cry audio samples | 6+ hours | Public (HuggingFace) | P1 |
| **ICSD** | Infant cry detection | 1,391 clips | Public (GitHub) | P2 |
| **Mendeley Jaundice** | Jaundice (bilirubin) | 300 infants | Public (Mendeley) | P2 |

---

## 1. Anemia Detection Datasets

### 1.1 Eyes-Defy-Anemia (Primary - DOWNLOAD FIRST)

**Source**: Kaggle + IEEE DataPort
**Size**: 218 images with hemoglobin values
**Quality**: Excellent (segmented conjunctiva, Hb values included)

```bash
# Option 1: Kaggle CLI
pip install kaggle
kaggle datasets download -d harshwardhanfartale/eyes-defy-anemia
unzip eyes-defy-anemia.zip -d data/eyes-defy-anemia

# Option 2: Manual download
# Go to: https://www.kaggle.com/datasets/harshwardhanfartale/eyes-defy-anemia
# Click "Download" button
```

**Dataset Structure**:
```
eyes-defy-anemia/
├── Italy/                    # 123 Italian patients
│   ├── patient_001/
│   │   ├── original.jpg
│   │   ├── palpebral.jpg
│   │   └── forniceal.jpg
│   └── Italy.xlsx            # Hb values, age, sex
├── India/                    # 95 Indian patients
│   ├── patient_001/
│   │   └── ...
│   └── India.xlsx
└── README.md
```

**Key Fields in Excel**:
- `Hb` - Hemoglobin value (g/dL)
- `Age` - Patient age
- `Sex` - Patient sex
- Classification: Hb < 11 g/dL = anemic (WHO standard)

### 1.2 Harvard Dataverse Conjunctiva Dataset

**Source**: Harvard Dataverse
**Size**: ~500 images
**Link**: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/L4MDKC

```bash
# Manual download required
# 1. Go to the link above
# 2. Click "Access Dataset"
# 3. Download files
# 4. Extract to data/harvard-conjunctiva/
```

### 1.3 Other Kaggle Anemia Datasets

```bash
# Additional datasets for more training data
kaggle datasets download -d nahiyan1402/anemiadataset
kaggle datasets download -d guptajanavi/palpebral-conjunctiva-to-detect-anaemia
```

---

## 2. Jaundice Detection Datasets

### 2.1 NeoJaundice (Primary - DOWNLOAD FIRST)

**Source**: Springer Nature Figshare
**Size**: 2,235 images of 745 infants
**Quality**: Excellent (bilirubin values included)
**Link**: https://springernature.figshare.com/articles/dataset/NeoJaundice/22302559

```bash
# Download from Figshare
# 1. Go to the link above
# 2. Click "Download all" or individual files
# 3. Extract to data/neojaundice/

# Expected structure after extraction:
# data/neojaundice/
# ├── images/
# │   ├── img_001.jpg
# │   └── ...
# └── labels.csv  (with bilirubin values)
```

**Paper Reference**: "NeoJaundice: Neonatal Jaundice Evaluation in Demographic Images"

### 2.2 NJN Dataset (Normal and Jaundiced Newborns)

**Source**: Google Sites
**Size**: 670 images (560 normal, 200 jaundiced)
**Link**: https://sites.google.com/view/neonataljaundice

```bash
# Manual download required
# 1. Go to the link above
# 2. Find download section
# 3. Download dataset
# 4. Extract to data/njn/

# Expected structure:
# data/njn/
# ├── normal/
# │   └── *.jpg
# ├── jaundiced/
# │   └── *.jpg
# └── metadata.xlsx
```

### 2.3 Mendeley Forehead/Sternum Dataset

**Source**: Mendeley Data
**Size**: 300 infants with TSB and TcB values
**Link**: https://data.mendeley.com/datasets/yfsz6c36vc/1

```bash
# Download from Mendeley
# 1. Go to the link above
# 2. Click "Download"
# 3. Extract to data/mendeley-jaundice/
```

---

## 3. Infant Cry / Birth Asphyxia Datasets

### 3.1 Baby Chillanto Database (Primary - REQUEST ACCESS)

**Source**: CONACYT Mexico (National Institute of Astrophysics and Optical Electronics)
**Size**: 2,268 samples (340 asphyxia, 507 normal, 879 deaf, 350 hungry, 192 pain)
**Access**: Request required

**How to Request Access**:

```
EMAIL TEMPLATE:
--------------
To: [Contact CONACYT/INAOE Mexico - search for current contact]
Subject: Request for Baby Chillanto Database Access for Academic Research

Dear Sir/Madam,

I am writing to request access to the Baby Chillanto Database for academic
research purposes.

Project: I am participating in Google's MedGemma Impact Challenge, a
healthcare AI competition focused on developing tools for community health
workers. My project, NEXUS, aims to detect birth asphyxia in newborns
using cry analysis powered by Google's HeAR model.

Purpose: The Baby Chillanto Database will be used to train and validate
a classifier for detecting asphyxia from infant cries. This classifier
will be integrated into a mobile health application designed for
low-resource settings where birth asphyxia claims over 1 million
newborn lives annually.

I commit to:
1. Using the data solely for research purposes
2. Properly citing the database in all publications
3. Not redistributing the data
4. Following all data protection guidelines

Thank you for considering my request. This research has the potential
to save lives by bringing diagnostic capabilities to frontline
healthcare workers.

Best regards,
[Your Name]
[Your Institution/Affiliation]
[Your Email]
```

### 3.2 CryCeleb 2023 (Ubenwa)

**Source**: HuggingFace
**Size**: 6+ hours of infant cry audio from 786 newborns
**Access**: Public (Creative Commons)
**Note**: No asphyxia labels, but useful for cry detection and audio processing

```bash
# Install git-lfs first
git lfs install

# Clone dataset
git clone https://huggingface.co/datasets/Ubenwa/CryCeleb2023 data/cryceleb

# Alternative: Use HuggingFace datasets library
pip install datasets
python -c "from datasets import load_dataset; ds = load_dataset('Ubenwa/CryCeleb2023')"
```

### 3.3 ICSD Dataset (Infant Cry and Snoring Detection)

**Source**: GitHub
**Size**: 1,391 infant cry clips
**Access**: Public

```bash
git clone https://github.com/QingyuLiu0521/ICSD/ data/icsd
```

---

## 4. Download Script

```python
#!/usr/bin/env python3
"""
download_datasets.py - Download all NEXUS datasets

Run: python scripts/download_datasets.py
"""

import os
import subprocess
import sys
from pathlib import Path

# Base data directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def run_command(cmd, cwd=None):
    """Run shell command."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    return result.returncode == 0

def check_kaggle():
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
        return False

    datasets = [
        ("harshwardhanfartale/eyes-defy-anemia", "eyes-defy-anemia"),
        ("nahiyan1402/anemiadataset", "anemia-dataset-2"),
    ]

    for dataset_id, folder_name in datasets:
        dest = DATA_DIR / folder_name
        if dest.exists():
            print(f"Skipping {folder_name} (already exists)")
            continue

        print(f"\nDownloading {dataset_id}...")
        run_command(f"kaggle datasets download -d {dataset_id} -p {DATA_DIR}")

        # Unzip
        zip_file = DATA_DIR / f"{folder_name}.zip"
        if zip_file.exists():
            run_command(f"unzip -q {zip_file} -d {dest}")
            zip_file.unlink()

    return True

def download_huggingface_datasets():
    """Download HuggingFace datasets."""
    cryceleb_dir = DATA_DIR / "cryceleb"

    if cryceleb_dir.exists():
        print("Skipping CryCeleb (already exists)")
        return True

    print("\nDownloading CryCeleb 2023...")

    # Check git-lfs
    if not run_command("git lfs version"):
        print("Installing git-lfs...")
        run_command("git lfs install")

    return run_command(
        f"git clone https://huggingface.co/datasets/Ubenwa/CryCeleb2023 {cryceleb_dir}"
    )

def download_github_datasets():
    """Download GitHub datasets."""
    icsd_dir = DATA_DIR / "icsd"

    if icsd_dir.exists():
        print("Skipping ICSD (already exists)")
        return True

    print("\nDownloading ICSD...")
    return run_command(f"git clone https://github.com/QingyuLiu0521/ICSD/ {icsd_dir}")

def print_manual_downloads():
    """Print instructions for manual downloads."""
    print("""
================================================================================
MANUAL DOWNLOADS REQUIRED
================================================================================

The following datasets require manual download:

1. NeoJaundice (REQUIRED):
   URL: https://springernature.figshare.com/articles/dataset/NeoJaundice/22302559
   Save to: data/neojaundice/

2. NJN Dataset (Recommended):
   URL: https://sites.google.com/view/neonataljaundice
   Save to: data/njn/

3. Harvard Conjunctiva (Optional):
   URL: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/L4MDKC
   Save to: data/harvard-conjunctiva/

4. Baby Chillanto (CRITICAL - Request Access):
   Contact: CONACYT/INAOE Mexico
   This is the ONLY dataset with birth asphyxia labels!
   Save to: data/baby-chillanto/

================================================================================
""")

def verify_datasets():
    """Verify downloaded datasets."""
    print("\n" + "=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)

    datasets = [
        ("eyes-defy-anemia", "Anemia (Primary)", True),
        ("neojaundice", "Jaundice (Primary)", True),
        ("njn", "Jaundice (Secondary)", False),
        ("cryceleb", "Cry Audio", False),
        ("icsd", "Cry Detection", False),
        ("baby-chillanto", "Asphyxia (CRITICAL)", True),
    ]

    missing_critical = []

    for folder, name, critical in datasets:
        path = DATA_DIR / folder
        exists = path.exists()
        status = "FOUND" if exists else "MISSING"
        indicator = "  " if not critical else ("" if exists else "")

        print(f"  {indicator} {name}: {status} ({path})")

        if critical and not exists:
            missing_critical.append(name)

    if missing_critical:
        print(f"\n WARNING: Missing critical datasets: {', '.join(missing_critical)}")
        print("   Please download these manually before proceeding.")
    else:
        print("\n All critical datasets are available!")

    return len(missing_critical) == 0

def main():
    """Main download function."""
    print("=" * 60)
    print("NEXUS Dataset Downloader")
    print("=" * 60)

    # Download automated datasets
    print("\n--- Automated Downloads ---")
    download_kaggle_datasets()
    download_huggingface_datasets()
    download_github_datasets()

    # Print manual download instructions
    print_manual_downloads()

    # Verify
    all_ready = verify_datasets()

    return 0 if all_ready else 1

if __name__ == "__main__":
    sys.exit(main())
```

---

## 5. Dataset Preparation Scripts

### 5.1 Prepare Anemia Dataset

```python
# scripts/prepare_anemia_data.py
"""Prepare anemia dataset for training."""

import os
import pandas as pd
from pathlib import Path
import shutil

def prepare_eyes_defy_anemia(source_dir: str, output_dir: str):
    """
    Prepare Eyes-Defy-Anemia dataset.

    Creates:
    - output_dir/anemic/
    - output_dir/healthy/
    - output_dir/labels.csv
    """
    source = Path(source_dir)
    output = Path(output_dir)

    anemic_dir = output / "anemic"
    healthy_dir = output / "healthy"
    anemic_dir.mkdir(parents=True, exist_ok=True)
    healthy_dir.mkdir(parents=True, exist_ok=True)

    all_records = []

    # Process Italy folder
    italy_xlsx = source / "Italy" / "Italy.xlsx"
    if italy_xlsx.exists():
        df = pd.read_excel(italy_xlsx)
        process_region(source / "Italy", df, anemic_dir, healthy_dir, all_records, "Italy")

    # Process India folder
    india_xlsx = source / "India" / "India.xlsx"
    if india_xlsx.exists():
        df = pd.read_excel(india_xlsx)
        process_region(source / "India", df, anemic_dir, healthy_dir, all_records, "India")

    # Save combined labels
    labels_df = pd.DataFrame(all_records)
    labels_df.to_csv(output / "labels.csv", index=False)

    print(f"Prepared {len(all_records)} samples:")
    print(f"  Anemic: {len(list(anemic_dir.glob('*.jpg')))}")
    print(f"  Healthy: {len(list(healthy_dir.glob('*.jpg')))}")

def process_region(region_dir, df, anemic_dir, healthy_dir, records, region_name):
    """Process a region's data."""
    for _, row in df.iterrows():
        # Adjust column names based on actual Excel structure
        hb = row.get('Hb', row.get('hemoglobin', None))
        patient_id = row.get('ID', row.get('Patient', None))

        if hb is None or patient_id is None:
            continue

        # Find image file
        patient_folder = region_dir / str(patient_id)
        if not patient_folder.exists():
            continue

        # Look for conjunctiva image
        for img_name in ['palpebral.jpg', 'original.jpg', 'conjunctiva.jpg']:
            img_path = patient_folder / img_name
            if img_path.exists():
                break
        else:
            continue

        # Classify: WHO defines anemia as Hb < 11 g/dL for women
        is_anemic = hb < 11.0

        # Copy to appropriate folder
        dest_dir = anemic_dir if is_anemic else healthy_dir
        dest_name = f"{region_name}_{patient_id}.jpg"
        shutil.copy(img_path, dest_dir / dest_name)

        records.append({
            'filename': dest_name,
            'hemoglobin': hb,
            'is_anemic': is_anemic,
            'region': region_name
        })

if __name__ == "__main__":
    prepare_eyes_defy_anemia(
        source_dir="data/eyes-defy-anemia",
        output_dir="data/prepared/anemia"
    )
```

### 5.2 Prepare Jaundice Dataset

```python
# scripts/prepare_jaundice_data.py
"""Prepare jaundice dataset for training."""

import os
import pandas as pd
from pathlib import Path
import shutil

def prepare_neojaundice(source_dir: str, output_dir: str, bilirubin_threshold: float = 12.0):
    """
    Prepare NeoJaundice dataset.

    Creates:
    - output_dir/jaundiced/
    - output_dir/normal/
    - output_dir/labels.csv
    """
    source = Path(source_dir)
    output = Path(output_dir)

    jaundiced_dir = output / "jaundiced"
    normal_dir = output / "normal"
    jaundiced_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)

    # Load labels (adjust based on actual dataset structure)
    labels_file = source / "labels.csv"
    if labels_file.exists():
        df = pd.read_csv(labels_file)
    else:
        # Try to find label file
        for f in source.glob("*.csv"):
            df = pd.read_csv(f)
            break
        else:
            print("No labels file found!")
            return

    records = []

    for _, row in df.iterrows():
        # Adjust column names based on actual CSV structure
        img_name = row.get('image', row.get('filename', row.get('Image', None)))
        bilirubin = row.get('bilirubin', row.get('TSB', row.get('Bilirubin', None)))

        if img_name is None or bilirubin is None:
            continue

        # Find image
        img_path = source / "images" / img_name
        if not img_path.exists():
            img_path = source / img_name
        if not img_path.exists():
            continue

        # Classify based on bilirubin threshold
        is_jaundiced = bilirubin > bilirubin_threshold

        # Copy to appropriate folder
        dest_dir = jaundiced_dir if is_jaundiced else normal_dir
        shutil.copy(img_path, dest_dir / img_name)

        records.append({
            'filename': img_name,
            'bilirubin': bilirubin,
            'is_jaundiced': is_jaundiced
        })

    # Save labels
    labels_df = pd.DataFrame(records)
    labels_df.to_csv(output / "labels.csv", index=False)

    print(f"Prepared {len(records)} samples:")
    print(f"  Jaundiced: {len(list(jaundiced_dir.glob('*')))}")
    print(f"  Normal: {len(list(normal_dir.glob('*')))}")

if __name__ == "__main__":
    prepare_neojaundice(
        source_dir="data/neojaundice",
        output_dir="data/prepared/jaundice"
    )
```

---

## 6. Data Directory Structure

After downloading and preparing all datasets:

```
data/
├── raw/                          # Original downloads
│   ├── eyes-defy-anemia/
│   ├── neojaundice/
│   ├── njn/
│   ├── cryceleb/
│   ├── icsd/
│   └── baby-chillanto/          # When access granted
│
├── prepared/                     # Processed for training
│   ├── anemia/
│   │   ├── anemic/
│   │   ├── healthy/
│   │   └── labels.csv
│   ├── jaundice/
│   │   ├── jaundiced/
│   │   ├── normal/
│   │   └── labels.csv
│   └── cry/
│       ├── asphyxia/
│       ├── normal/
│       └── labels.csv
│
└── test/                         # Hold-out test set
    ├── anemia/
    ├── jaundice/
    └── cry/
```

---

## 7. Priority Checklist

### Today (Do Now)
- [ ] Configure Kaggle API (`~/.kaggle/kaggle.json`)
- [ ] Run `python scripts/download_datasets.py`
- [ ] Download NeoJaundice manually from Figshare
- [ ] **Send Baby Chillanto access request email**

### This Week
- [ ] Download NJN dataset
- [ ] Prepare all datasets using scripts
- [ ] Verify data quality

### If Baby Chillanto Access Denied
- [ ] Use CryCeleb for audio samples
- [ ] Implement rule-based cry analysis (see TECHNICAL_IMPLEMENTATION_GUIDE.md)
- [ ] Focus demo on anemia + jaundice

---

## 8. Dataset Licenses & Citations

When using these datasets, include proper citations:

```bibtex
@dataset{eyes_defy_anemia,
  title={Eyes-Defy-Anemia: Conjunctiva Image Dataset for Anemia Detection},
  author={Fartale, Harshwardhan},
  year={2024},
  publisher={IEEE DataPort / Kaggle}
}

@dataset{neojaundice,
  title={NeoJaundice: Neonatal Jaundice Evaluation in Demographic Images},
  year={2023},
  publisher={Springer Nature Figshare}
}

@dataset{cryceleb,
  title={CryCeleb: A Speaker Verification Dataset Based on Infant Cry Sounds},
  author={Ubenwa},
  year={2023},
  publisher={HuggingFace}
}

@article{baby_chillanto,
  title={Infant Cry Classification Using Baby Chillanto Database},
  author={CONACYT Mexico},
  journal={National Institute of Astrophysics and Optical Electronics}
}
```

---

**Document Version**: 1.0
**Created**: January 14, 2026
**For**: NEXUS - MedGemma Impact Challenge
