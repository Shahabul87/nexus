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
| **MedSigLIP** (`google/medsiglip-448`) | Anemia detection from conjunctiva images; jaundice detection and bilirubin regression from skin images | Zero-shot classification + trained linear probes + novel bilirubin regression head |
| **HeAR** (`google/hear-pytorch`) | Birth asphyxia screening from infant cry recordings | Audio embeddings with acoustic feature fallback |
| **MedGemma** (`google/medgemma-4b-it`) | Clinical reasoning and synthesis of multi-modal findings | Structured prompt with agent reasoning traces |

### Agentic Clinical Workflow

NEXUS uses a **6-agent sequential pipeline** where each agent produces step-by-step reasoning traces, creating a full audit trail:

```
Triage -> Image Analysis (MedSigLIP) -> Audio Analysis (HeAR)
  -> WHO Protocol -> Referral Decision -> Clinical Synthesis (MedGemma)
```

Each agent emits structured results with reasoning chains, confidence scores, and processing times. Critical danger signs trigger early-exit to immediate referral. The workflow follows WHO IMNCI protocols for severity classification (RED/YELLOW/GREEN).

### Novel Task: Bilirubin Regression

We fine-tuned a **2-layer MLP regression head** (1152→256→1, 295K params) on frozen MedSigLIP embeddings to predict continuous bilirubin levels (mg/dL) from neonatal skin images -- a novel application beyond MedSigLIP's original zero-shot classification design. Trained on 2,235 images from the NeoJaundice dataset with ground truth serum bilirubin measurements.

**Results**: MAE = **2.667 mg/dL**, RMSE = 3.402, Pearson r = **0.7725** (p < 1e-67). Bland-Altman analysis shows mean bias of 0.217 mg/dL with 95% limits of agreement [-6.4, 6.9].

---

## Technical Details

### Architecture

- **Frontend**: Streamlit interactive demo with 6 assessment modes
- **Backend**: FastAPI with RESTful endpoints for all assessment types + agentic workflow
- **Mobile**: React Native scaffold (offline-first design)
- **Edge AI**: INT8 dynamic quantization, pre-computed text embeddings, TorchScript export

### Model Performance

| Task | Method | Metric |
|------|--------|--------|
| Anemia classification | MedSigLIP zero-shot + linear probe | 52.27% accuracy (pseudo-labels) |
| Jaundice classification | MedSigLIP zero-shot + linear probe | 68.90% accuracy |
| Bilirubin regression | MedSigLIP embeddings + MLP | MAE: 2.667 mg/dL, r=0.77 |
| Cry analysis | HeAR embeddings + acoustic features | Qualitative assessment |
| Clinical synthesis | MedGemma 4B-it | WHO IMNCI-aligned recommendations |

### Edge Deployment

| Component | Cloud Size | Edge Size | Compression |
|-----------|-----------|-----------|-------------|
| MedSigLIP vision encoder | 812.6 MB | 111.2 MB (INT8) | **7.31x** |
| Acoustic model | 0.665 MB | 0.599 MB (INT8) | 1.11x |
| Text embeddings | Computed | 12 KB (pre-computed) | Offline-ready |
| Total on-device | - | ~289 MB | Offline-ready |

Target: Android 8.0+, ARM Cortex-A53, 2 GB RAM.

---

## Links

- **Video Demo (3 min)**: [YouTube Link]
- **Public Code Repository**: [GitHub Link]
- **Live Demo**: [HuggingFace Spaces Link]

---

## Tracks

- [x] Main Track
- [x] Agentic Workflow Prize
