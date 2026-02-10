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

### Linear Probes Trained (Feb 4, 2026)
- **Anemia SVM_RBF**: 99.94% accuracy (7x augmentation, pseudo-labels)
- **Jaundice SVM_RBF**: 96.73% accuracy (3x augmentation, real bilirubin labels)
- **Cry SVM_RBF**: 83.81% accuracy (HeAR 512-dim embeddings, 5-class)
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

### WEEK 3 COMPLETE (Jan 30, 2026)
- [x] 6-agent agentic workflow engine (`src/nexus/agentic_workflow.py`)
- [x] Agentic workflow integrated into pipeline, API, and Streamlit
- [x] 41 agentic workflow tests passing
- [x] Clinical synthesizer enhanced with agent reasoning traces
- [x] Novel task: bilirubin regression from MedSigLIP embeddings
- [x] Bilirubin regressor integrated into jaundice detector
- [x] Reproducibility notebook (`notebooks/04_bilirubin_regression.ipynb`)
- [x] Edge AI Mode toggle in Streamlit demo
- [x] Edge benchmarks document (`docs/edge_benchmarks.md`)
- [x] HuggingFace Spaces entry point (`app.py`)
- [x] Competition writeup rewritten (`submission/writeup.md`)
- [x] Video script updated (`submission/video/DEMO_VIDEO_SCRIPT.md`)
- [x] README.md updated with architecture, setup, and structure
- [x] Bilirubin regression trained: MAE=2.667 mg/dL, Pearson r=0.7725
- [x] Edge quantization complete: MedSigLIP 7.31x compression (812.6→111.2 MB)
- [x] Text embeddings exported: 4 categories, 768-dim binary files
- [x] All benchmarks updated with real data

### Bilirubin Regression Results (Feb 4, 2026)
- **MAE**: 2.564 mg/dL
- **RMSE**: 3.416 mg/dL
- **Pearson r**: 0.7783 (p < 1e-69)
- **Bland-Altman**: mean bias -0.506, 95% LoA [-7.1, 6.1]
- **Model**: `models/linear_probes/bilirubin_regressor.pt`

### Edge Quantization Results (Jan 30, 2026)
- **MedSigLIP INT8**: 812.6 MB → 111.2 MB memory (7.31x), 287.4 MB on disk
- **Acoustic INT8**: 0.665 MB → 0.599 MB (1.11x)
- **CPU latency**: 97.68 ms (FP32) vs 111.19 ms (INT8, macOS qnnpack)
- **Text embeddings**: 4 x 768-dim binary files (12 KB total)

### WEEK 4 - POLISH (Feb 4, 2026)
- [x] Expanded zero-shot prompts: 8 prompts per class (anemia + jaundice)
- [x] Max-similarity scoring: best-matching prompt per class instead of mean-pooled
- [x] Tuned logit temperature: 100 -> 30 for better probability calibration
- [x] Honest accuracy claims: removed inflated 85-98% numbers throughout
- [x] Improved Streamlit demo: error handling, ML bilirubin display, model status
- [x] Improved agentic workflow display: pipeline status bar, better reasoning traces, workflow summary
- [x] Fixed app.py: proper PYTHONPATH setup, environment defaults
- [x] Updated writeup: judging-criteria mapping table, honest metrics, stronger bilirubin narrative
- [x] Updated README: train models section, accurate results table
- [x] Updated requirements_spaces.txt: added plotly, joblib

### WEEK 4 - FIXES (Feb 4, 2026)
- [x] Fix 1: MedGemma 4-bit NF4 quantization (BitsAndBytes) — supports MedGemma 1.5
- [x] Fix 2: HeAR cry classifier — trained on donate-a-cry (5-class: hungry, belly_pain, burping, discomfort, tired)
- [x] Fix 3: Anemia accuracy improvement — SVM/LR on MedSigLIP embeddings with 7x augmentation
- [x] Fix 4: Jaundice accuracy improvement — SVM/LR on MedSigLIP embeddings with 3x augmentation
- [x] Fix 5: Deepened agentic workflow — 3-5 step reasoning per agent, comorbidity analysis, facility matching
- [x] Fix 6: Bilirubin regression upgrade — 3-layer MLP with BatchNorm, CosineAnnealingWarmRestarts
- [x] Fix 7: Demo polish — model badges, updated HAI-DEF info, writeup/video script refresh

### Training Results (Feb 4, 2026)
- **Anemia classifier**: SVM_RBF **99.94%** (5-fold CV, 218→1,744 augmented), `models/linear_probes/anemia_classifier.joblib`
- **Jaundice classifier**: SVM_RBF **96.73%** (5-fold CV, 2,235→8,940 augmented), `models/linear_probes/jaundice_classifier.joblib`
- **Cry classifier**: SVM_RBF **83.81%** (5-fold CV, 457 samples, 5-class), `models/linear_probes/cry_classifier.joblib`
- **Bilirubin regressor**: MAE 2.564, r=0.7783 (58 epochs), `models/linear_probes/bilirubin_regressor.pt`

### End-to-End Validation (Feb 5, 2026)
| Tab | Status | Result |
|-----|--------|--------|
| HAI-DEF Models Info | ✅ PASS | All 3 models displayed correctly |
| Maternal Anemia | ✅ PASS | No Anemia Detected, 100% confidence, Hb 12.1 g/dL |
| Neonatal Jaundice | ✅ PASS | MILD jaundice, 11.6 mg/dL (MedSigLIP Regressor) |
| Cry Analysis | ✅ PASS | Hungry cry identified, 8.3% asphyxia risk |
| Agentic Workflow | ✅ PASS | 6-agent pipeline (MedGemma requires HF_TOKEN) |

### Remaining Before Submission
1. ~~Re-run training scripts to generate model weights~~ DONE
2. ~~Validate end-to-end with real models~~ DONE (Feb 5)
3. Deploy to HuggingFace Spaces
4. Record 3-minute video demo
5. Submit on Kaggle
5. Submit on Kaggle

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

*Target: Main Track ($30,000) + Agentic Workflow Prize ($5,000+$5,000)*
