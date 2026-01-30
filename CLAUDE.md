# NEXUS Project - MedGemma Impact Challenge

## CRITICAL COMPETITION REQUIREMENTS

### Deadline
**February 24, 2026, 11:59 PM UTC** - NO EXCEPTIONS

### Mandatory Requirements
1. **MUST use at least ONE HAI-DEF model** (MedGemma is MANDATORY per competition rules)
2. Video demo (3 minutes or less)
3. Technical writeup (3 pages or less)
4. Public code repository

---

## HAI-DEF MODELS - USE THESE ONLY

### Required Models for NEXUS

| Model | Task | HuggingFace ID | Status |
|-------|------|----------------|--------|
| **MedGemma** | Clinical reasoning & synthesis | `google/medgemma-4b-it` | INTEGRATED (HF token required) |
| **MedSigLIP** | Anemia & jaundice detection | `google/medsiglip-448` | INTEGRATED (HF token required) |
| **HeAR** | Cry analysis (birth asphyxia) | `google/hear-pytorch` | INTEGRATED (PyTorch, with acoustic fallback) |

### DO NOT USE
- `openai/clip-vit-base-patch32` - REPLACED
- Any non-HAI-DEF models for core functionality

---

## COMPLETED HAI-DEF INTEGRATION

### Completed: MedSigLIP for Image Analysis
**Files updated:**
- `src/nexus/anemia_detector.py` - Uses `google/medsiglip-448` (with `google/siglip-base-patch16-224` fallback)
- `src/nexus/jaundice_detector.py` - Uses `google/medsiglip-448` (with `google/siglip-base-patch16-224` fallback)

### Completed: HeAR for Cry Analysis
**File updated:**
- `src/nexus/cry_analyzer.py` - Uses `google/hear-pytorch` via HuggingFace (PyTorch, with acoustic feature fallback)

### Completed: MedGemma for Clinical Reasoning
**File updated:**
- `src/nexus/clinical_synthesizer.py` - Uses `google/medgemma-4b-it` with HF token auth and rule-based fallback

---

## EVALUATION CRITERIA (How We Win)

| Criterion | Weight | What Judges Look For |
|-----------|--------|---------------------|
| **Execution & Communication** | 30% | Polished video, clear demo, quality code |
| **Effective HAI-DEF Use** | 20% | MedGemma/MedSigLIP/HeAR used appropriately |
| **Product Feasibility** | 20% | Working demo, realistic deployment path |
| **Problem Domain** | 15% | Clear problem definition, user journey |
| **Impact Potential** | 15% | Quantified impact, scalability |

---

## PROJECT ARCHITECTURE

```
NEXUS Platform
├── Pregnancy Module
│   └── Anemia Screening (MedSigLIP - eye photos)
├── Neonatal Module
│   ├── Birth Asphyxia Detection (HeAR - cry audio)
│   └── Jaundice Detection (MedSigLIP - skin photos)
└── Clinical Reasoning
    └── Synthesis & Recommendations (MedGemma)
```

---

## SUBMISSION CHECKLIST

### Required Deliverables
- [ ] 3-minute video demo
- [ ] 3-page writeup (use Kaggle template)
- [ ] Public GitHub repository
- [ ] MedGemma usage demonstrated
- [ ] At least one other HAI-DEF model used

### Bonus (Helps Win)
- [ ] Live interactive demo app
- [ ] Open-weight HuggingFace model (if fine-tuned)
- [ ] Edge AI demonstration (offline capability)

---

## DATASETS ACQUIRED

| Dataset | Purpose | Count | Location |
|---------|---------|-------|----------|
| Eyes-Defy-Anemia | Anemia detection | 218 images | `data/raw/eyes-defy-anemia/` |
| NeoJaundice | Jaundice detection | 2,235 images | `data/raw/neojaundice/` |
| CryCeleb | Cry audio samples | 26,093 files | `data/raw/cryceleb/` |
| donate-a-cry | Labeled cry types | 457 files | `data/raw/donate-a-cry/` |
| infant-cry-dataset | Cry detection | 741 files | `data/raw/infant-cry-dataset/` |

---

## CURRENT STATUS

### HAI-DEF Models Integrated
- `src/nexus/anemia_detector.py` - MedSigLIP (`google/medsiglip-448`)
- `src/nexus/jaundice_detector.py` - MedSigLIP (`google/medsiglip-448`)
- `src/nexus/cry_analyzer.py` - HeAR (`google/hear-pytorch`) with acoustic feature fallback
- `src/nexus/clinical_synthesizer.py` - MedGemma 4B (`google/medgemma-4b-it`) with rule-based fallback

### Linear Probes Trained (Jan 14, 2026)
- **Anemia Linear Probe**: 52.27% accuracy (data labels are pseudo-labels)
- **Jaundice Linear Probe**: 68.90% accuracy
- Models saved in `models/linear_probes/`

### Applications Built
- **Streamlit Demo**: `src/demo/streamlit_app.py` - Full HAI-DEF integration
- **React Native Mobile App**: `mobile/` - Complete scaffold with all screens
- **FastAPI Backend**: `api/main.py` - Serves HAI-DEF models to mobile app

### Zero-Shot Validation Results (Jan 14, 2026)
- **Anemia Detector**: Working with 63-77% confidence
- **Jaundice Detector**: Working with 65-98% confidence
- **Cry Analyzer**: Working with acoustic features (HeAR fallback)
- **Clinical Synthesizer**: Working with rule-based WHO IMNCI protocols

### WEEK 1-2 COMPLETE
- [x] HAI-DEF model integration (MedSigLIP, HeAR, MedGemma)
- [x] Zero-shot validation on all datasets
- [x] Linear probe training on embeddings
- [x] Streamlit demo with Combined Assessment
- [x] React Native mobile app scaffold
- [x] FastAPI backend for mobile app

### Next Steps (Week 3)
1. Model quantization for edge deployment (INT8)
2. Full mobile app integration testing
3. Record 3-minute video demo
4. Write 3-page competition submission

---

## CODE STANDARDS

### TypeScript/React
- NO `any` types
- NO ESLint disable comments
- Proper error handling

### Python
- Type hints required
- Docstrings for all functions
- Validate inputs with proper types

### Testing
- Test HAI-DEF model integration
- Test edge cases
- Validate accuracy metrics

---

## KEY RESOURCES

### HAI-DEF Documentation
- MedGemma: https://developers.google.com/health-ai-developer-foundations/medgemma
- MedSigLIP: https://developers.google.com/health-ai-developer-foundations/medsiglip
- HeAR: https://github.com/Google-Health/google-health/blob/master/health_acoustic_representations/README.md

### Competition Page
- https://www.kaggle.com/competitions/med-gemma-impact-challenge

---

## REMINDER

**This competition REQUIRES HAI-DEF models.**

Using OpenAI CLIP instead of MedSigLIP could DISQUALIFY our submission.

**FIX THIS FIRST before any other work.**

---

*Target: First Prize ($30,000) + Edge AI Prize ($5,000)*
