---
title: NEXUS
emoji: "\U0001FA7A"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: true
license: cc-by-4.0
tags:
  - medgemma
  - medical-ai
  - hai-def
  - maternal-health
  - neonatal-care
---

# NEXUS - AI-Powered Maternal-Neonatal Assessment Platform

> Non-invasive screening for maternal anemia, neonatal jaundice, and birth asphyxia using Google HAI-DEF models

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![MedGemma Impact Challenge](https://img.shields.io/badge/Kaggle-MedGemma%20Impact%20Challenge-20BEFF)](https://www.kaggle.com/competitions/med-gemma-impact-challenge)

## Overview

NEXUS transforms smartphones into diagnostic screening tools for Community Health Workers in low-resource settings. Using 3 Google HAI-DEF models in a 6-agent clinical workflow, it provides non-invasive assessment for:

- **Maternal anemia** from conjunctiva images (MedSigLIP)
- **Neonatal jaundice** from skin images with bilirubin regression (MedSigLIP)
- **Birth asphyxia** from cry audio analysis (HeAR)
- **Clinical synthesis** with WHO IMNCI protocol alignment (MedGemma)

## HAI-DEF Models

| Model | HuggingFace ID | Purpose |
|-------|----------------|---------|
| **MedSigLIP** | `google/medsiglip-448` | Anemia + jaundice detection, bilirubin regression |
| **HeAR** | `google/hear-pytorch` | Cry audio analysis for birth asphyxia |
| **MedGemma 4B** | `google/medgemma-4b-it` | Clinical reasoning and synthesis |

## Architecture

```
6-Agent Clinical Workflow:
  Triage -> Image Analysis (MedSigLIP) -> Audio Analysis (HeAR)
    -> WHO Protocol -> Referral Decision -> Clinical Synthesis (MedGemma)

Each agent produces structured reasoning traces for a full audit trail.
```

## Quick Start

### Prerequisites
- Python 3.10+
- HuggingFace token (for gated HAI-DEF models)

### Setup

```bash
# Clone and install
git clone <repo-url>
cd nexus
pip install -r requirements.txt

# Set HuggingFace token (required for MedSigLIP, MedGemma)
export HF_TOKEN=hf_your_token_here
```

### Run the Demo

```bash
# Streamlit interactive demo
PYTHONPATH=src streamlit run src/demo/streamlit_app.py

# FastAPI backend
PYTHONPATH=src uvicorn api.main:app --reload

# Run tests
PYTHONPATH=src python -m pytest tests/ -v
```

### Train Models

```bash
# Train linear probes (anemia + jaundice classifiers)
PYTHONPATH=src python scripts/training/train_linear_probes.py

# Train bilirubin regression head
PYTHONPATH=src python scripts/training/finetune_bilirubin_regression.py
```

### HuggingFace Spaces

```bash
# Local test of HF Spaces entry point
python app.py
```

## Project Structure

```
nexus/
├── src/nexus/                         # Core platform
│   ├── anemia_detector.py             # MedSigLIP anemia detection
│   ├── jaundice_detector.py           # MedSigLIP jaundice + bilirubin regression
│   ├── cry_analyzer.py                # HeAR cry analysis
│   ├── clinical_synthesizer.py        # MedGemma clinical synthesis
│   ├── agentic_workflow.py            # 6-agent workflow engine
│   └── pipeline.py                    # Unified assessment pipeline
├── src/demo/streamlit_app.py          # Interactive Streamlit demo
├── api/main.py                        # FastAPI backend
├── scripts/
│   ├── training/
│   │   ├── train_linear_probes.py     # MedSigLIP embedding classifiers
│   │   ├── finetune_bilirubin_regression.py  # Novel bilirubin regression
│   │   ├── train_anemia.py            # Anemia-specific training
│   │   ├── train_jaundice.py          # Jaundice-specific training
│   │   └── train_cry.py              # Cry classifier training
│   └── edge/
│       ├── quantize_models.py         # INT8 quantization
│       └── export_embeddings.py       # Pre-computed text embeddings
├── notebooks/
│   ├── 01_anemia_detection.ipynb
│   ├── 02_jaundice_detection.ipynb
│   ├── 03_cry_analysis.ipynb
│   └── 04_bilirubin_regression.ipynb  # Novel task reproducibility
├── tests/
│   ├── test_pipeline.py               # Pipeline tests
│   ├── test_agentic_workflow.py       # Agentic workflow tests (41 tests)
│   └── test_hai_def_integration.py    # HAI-DEF model compliance
├── models/
│   ├── linear_probes/                 # Trained classifiers + regressor
│   └── edge/                          # Quantized models + embeddings
├── data/
│   ├── raw/                           # Raw datasets (Eyes-Defy-Anemia, NeoJaundice, CryCeleb)
│   └── protocols/                     # WHO IMNCI protocols
├── submission/
│   ├── writeup.md                     # Competition writeup (3 pages)
│   └── video/                         # Demo video script and assets
├── app.py                             # HuggingFace Spaces entry point
├── requirements.txt                   # Full dependencies
└── requirements_spaces.txt            # HF Spaces minimal dependencies
```

## Key Results

| Task | Method | Performance |
|------|--------|-------------|
| Anemia zero-shot | MedSigLIP (max-similarity, 8 prompts/class) | Screening capability |
| Jaundice classification | MedSigLIP linear probe | 68.9% accuracy |
| **Bilirubin regression** | **MedSigLIP + MLP head** | **MAE: 2.667 mg/dL, r=0.77** |
| Cry analysis | HeAR + acoustic features | Qualitative assessment |
| Clinical synthesis | MedGemma + WHO IMNCI | Protocol-aligned recommendations |

### Novel Task: Bilirubin Regression
Frozen MedSigLIP embeddings -> 2-layer MLP -> continuous bilirubin (mg/dL) prediction.
Trained on 2,235 NeoJaundice images with ground truth serum bilirubin.
**MAE: 2.667 mg/dL, Pearson r: 0.7725 (p < 1e-67)**

### Edge AI
- INT8 dynamic quantization: 812.6 MB -> 111.2 MB (7.31x compression)
- Pre-computed text embeddings: 12 KB (no text encoder on device)
- Total on-device: ~289 MB

## Competition Tracks

- **Main Track**: Comprehensive maternal-neonatal assessment platform
- **Agentic Workflow Prize**: 6-agent pipeline with reasoning traces and audit trail

## Tests

```bash
# All tests
PYTHONPATH=src python -m pytest tests/ -v

# Agentic workflow only (41 tests)
PYTHONPATH=src python -m pytest tests/test_agentic_workflow.py -v
```

## License

[CC BY 4.0](LICENSE)

## Acknowledgments

- Google Health AI Developer Foundations team
- NeoJaundice dataset (Figshare)
- Eyes-Defy-Anemia dataset (Kaggle)
- WHO IMNCI protocol guidelines

---

Built with Google HAI-DEF for the MedGemma Impact Challenge 2026
