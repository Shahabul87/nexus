# NEXUS: AI-Powered Maternal-Neonatal Assessment Platform

## Project Name
**NEXUS** - Non-invasive EXamination Utility System

## Your Team

| Name | Specialty | Role |
|------|-----------|------|
| Md Shahabul Alam | Machine Learning | Model integration, agentic architecture, edge optimization |

---

## Problem Statement

Every day, **800 women** die from pregnancy-related complications and **7,400 newborns** do not survive their first month. 94% of these deaths occur in low-resource settings where diagnostic tools are scarce but smartphones are increasingly available.

**6.9 million Community Health Workers (CHWs)** form the backbone of primary healthcare delivery in these regions, yet they lack clinical decision support for three critical conditions:

- **Maternal anemia** affects 40% of pregnancies globally. Detection requires blood tests unavailable at community level.
- **Neonatal jaundice** affects 60% of newborns. Delayed detection causes irreversible brain damage (kernicterus).
- **Birth asphyxia** accounts for 23% of neonatal deaths. Early warning signs in infant cry patterns go unrecognized.

**Impact potential**: If deployed to 10% of CHWs globally, NEXUS would empower 690,000 health workers across ~500 million patient interactions annually.

---

## Overall Solution

NEXUS integrates **3 HAI-DEF models** in a multi-agent clinical workflow that transforms a smartphone into a diagnostic screening tool:

| HAI-DEF Model | Use Case | Integration |
|---------------|----------|-------------|
| **MedSigLIP** (`google/medsiglip-448`) | Anemia detection from conjunctiva images; jaundice detection and bilirubin regression from skin images | Zero-shot classification (max-similarity, 8 prompts/class) + trained SVM classifiers on embeddings with augmentation + novel 3-layer MLP bilirubin regression head with BatchNorm |
| **HeAR** (`google/hear-pytorch`) | Birth asphyxia screening and cry type classification from infant cry recordings | 512-dim HeAR embeddings + trained linear classifier (5-class: hungry, belly_pain, burping, discomfort, tired) with asphyxia risk derived from distress patterns |
| **MedGemma** (`google/medgemma-1.5-4b-it`) | Clinical reasoning and synthesis of multi-modal findings | 4-bit NF4 quantized inference (~2 GB VRAM), structured prompt with 6-agent reasoning traces, WHO IMNCI protocol alignment |

### Agentic Clinical Workflow

NEXUS uses a **6-agent sequential pipeline** where each agent produces step-by-step reasoning traces, creating a full audit trail:

```
Triage -> Image Analysis (MedSigLIP) -> Audio Analysis (HeAR)
  -> WHO Protocol -> Referral Decision -> Clinical Synthesis (MedGemma)
```

Each agent produces 3-5 step reasoning chains demonstrating genuine clinical thinking:
- **Triage**: Danger sign scoring with comorbidity detection, demographic risk assessment (birth weight, APGAR, gestational age), and clinical decision tree
- **Image Analysis**: MedSigLIP classification with trained SVM/LR classifiers and bilirubin regression
- **Audio Analysis**: HeAR embedding extraction with trained 5-class cry type classifier
- **WHO Protocol**: Condition-specific WHO IMNCI protocols with comorbidity analysis (e.g., hemolytic disease screening when anemia + jaundice co-occur) and conflict resolution
- **Referral**: Facility capability matching (blood bank, phototherapy, NICU), pre-referral stabilization actions, and structured referral notes
- **Synthesis**: MedGemma 1.5 (4-bit NF4) produces unified clinical recommendations from all agent findings

Critical danger signs trigger early-exit to immediate referral. The workflow follows WHO IMNCI protocols for severity classification (RED/YELLOW/GREEN).

### Novel Task: Bilirubin Regression from MedSigLIP Embeddings

We trained a **3-layer MLP regression head with BatchNorm** (1152->512->256->1, ~724K params) on frozen MedSigLIP embeddings to predict continuous bilirubin levels (mg/dL) from neonatal skin images -- a novel application that extends MedSigLIP beyond its original zero-shot classification design into quantitative clinical measurement. The deeper architecture with BatchNorm normalization and CosineAnnealingWarmRestarts learning rate scheduling improves training stability and feature learning.

Trained on 2,235 images from the NeoJaundice dataset with ground truth serum bilirubin measurements (Huber loss, early stopping, 70/15/15 split):

| Metric | Value |
|--------|-------|
| **MAE** | **2.564 mg/dL** |
| RMSE | 3.416 mg/dL |
| **Pearson r** | **0.7783** (p < 1e-69) |
| Bland-Altman bias | -0.506 mg/dL |
| 95% Limits of Agreement | [-7.1, 6.1] mg/dL |

This correlation (r=0.78) demonstrates that MedSigLIP's vision encoder captures clinically meaningful skin color features related to bilirubin concentration, enabling non-invasive screening that could reduce the need for blood draws in resource-limited settings.

---

## Technical Details

### Architecture

- **Frontend**: Streamlit interactive demo with 6 assessment modes (anemia, jaundice, cry, combined, agentic, model info)
- **Backend**: FastAPI with RESTful endpoints for all assessment types + agentic workflow
- **Mobile**: React Native scaffold (offline-first design)
- **Edge AI**: INT8 dynamic quantization (7.31x compression), pre-computed text embeddings for offline inference

### Model Performance

| Task | Method | Metric | Notes |
|------|--------|--------|-------|
| Anemia classification | MedSigLIP embeddings + trained SVM_RBF | **99.94%** accuracy | 218 images with 7x augmentation; SVM trained on augmented embeddings |
| Jaundice classification | MedSigLIP embeddings + trained SVM_RBF | **96.73%** accuracy | 2,235 images with 3x augmentation; binary (bilirubin > 5 mg/dL) |
| **Bilirubin regression** | **MedSigLIP embeddings + 3-layer MLP (BatchNorm)** | **MAE: 2.564, r=0.78** | **Novel quantitative task from frozen embeddings; 70/15/15 split** |
| Cry type classification | HeAR 512-dim embeddings + trained SVM_RBF | **83.81%** (5-fold CV) | donate-a-cry dataset (457 files, 5 classes); asphyxia risk from distress patterns |
| Clinical synthesis | MedGemma 1.5 4B (4-bit NF4) + WHO IMNCI | Protocol-aligned | 6-agent reasoning traces with 3-5 steps each, full audit trail |

### Edge Deployment

| Component | FP32 Memory | INT8 Memory | Compression |
|-----------|-----------|-----------|-------------|
| MedSigLIP vision encoder | 812.6 MB | 111.2 MB | **7.31x** |
| Acoustic model | 0.665 MB | 0.599 MB | 1.11x |
| Text embeddings | Computed at runtime | 12 KB (pre-computed binary) | Offline-ready |
| Total on-device footprint | - | ~289 MB (disk) | Target: 2 GB RAM devices |

---

## How NEXUS Addresses Each Judging Criterion

| Criterion (Weight) | How NEXUS Addresses It |
|--------------------|----------------------|
| **Execution & Communication (30%)** | Polished Streamlit demo with 6 interactive tabs, video walkthrough, clean public repository with reproducibility instructions |
| **Effective HAI-DEF Use (20%)** | All 3 HAI-DEF models integrated: MedSigLIP for image analysis (zero-shot + linear probes + novel bilirubin regression), HeAR for cry analysis, MedGemma for clinical synthesis |
| **Product Feasibility (20%)** | Working demo on HuggingFace Spaces, edge-optimized models (7.31x compression), React Native mobile scaffold, FastAPI backend |
| **Problem Domain (15%)** | Addresses 3 leading causes of maternal-neonatal mortality in low-resource settings; clear CHW user journey; WHO IMNCI protocol alignment |
| **Impact Potential (15%)** | 6.9M CHWs globally, 500M+ potential patient interactions; offline-capable for areas without reliable internet; smartphone-based (no special hardware) |

---

## Links

- **Video Demo**: [YouTube](https://youtu.be/J6_jPBnRfbU)
- **Public Code Repository**: [github.com/Shahabul87/nexus](https://github.com/Shahabul87/nexus)
- **Live Demo**: [huggingface.co/spaces/Shahabul/nexus](https://huggingface.co/spaces/Shahabul/nexus)

---

## Tracks

- [x] Main Track
- [x] Agentic Workflow Prize
