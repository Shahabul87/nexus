# NEXUS Demo Video Script

**Target Duration**: 3 minutes (max)
**Format**: Screen recording of live Streamlit demo with voiceover

---

## Timing Overview

| Section | Time | Content |
|---------|------|---------|
| Opening Hook | 0:00 - 0:12 | Mortality statistics |
| Problem | 0:12 - 0:35 | CHW challenge, 3 conditions |
| NEXUS Overview | 0:35 - 0:55 | Platform intro, 3 HAI-DEF models |
| Live Demo: Maternal | 0:55 - 1:25 | Anemia screening with real image |
| Live Demo: Newborn | 1:25 - 1:55 | Jaundice + cry analysis |
| Agentic Workflow | 1:55 - 2:30 | Reasoning trace walkthrough |
| Bilirubin Regression | 2:30 - 2:42 | Before/after fine-tuning metrics |
| Edge + Impact | 2:42 - 3:00 | Offline capability, closing |

---

## Detailed Script

### OPENING HOOK (0:00 - 0:12)

**[VISUAL: Statistics appearing on dark background]**

> "Every day, 800 mothers die from pregnancy complications. 7,400 newborns don't survive their first month. 94% of these deaths happen where smartphones exist but diagnostic tools don't."

---

### PROBLEM (0:12 - 0:35)

**[VISUAL: Streamlit app landing page]**

> "6.9 million community health workers are the only healthcare providers for billions of people. They need to screen for three conditions: maternal anemia -- invisible without blood tests. Neonatal jaundice -- where delayed detection causes brain damage. And birth asphyxia -- where infant cry patterns hold critical warning signs."

---

### NEXUS OVERVIEW (0:35 - 0:55)

**[VISUAL: Streamlit HAI-DEF Models Info tab showing 3 model cards with badges]**

> "NEXUS uses three Google HAI-DEF models working together. MedSigLIP analyzes conjunctiva and skin images with trained classifiers and a novel bilirubin regression head. HeAR produces 512-dimensional embeddings for cry type classification. And MedGemma 1.5, running in 4-bit quantized mode, provides clinical reasoning."

---

### LIVE DEMO: MATERNAL (0:55 - 1:25)

**[VISUAL: Navigate to Maternal Anemia Screening tab]**

> "Here's a live demo. A health worker uploads a conjunctiva photo."

**[ACTION: Upload sample conjunctiva image]**

> "MedSigLIP performs classification using trained SVM classifiers on its embeddings, enhanced with data augmentation. Within seconds: anemia risk score, estimated hemoglobin, severity level, and a clinical recommendation."

**[VISUAL: Show results with confidence scores]**

---

### LIVE DEMO: NEWBORN (1:25 - 1:55)

**[VISUAL: Navigate to Neonatal Jaundice Detection tab]**

> "For neonatal assessment, a skin photo is analyzed for jaundice."

**[ACTION: Upload jaundice skin image, show results]**

> "MedSigLIP detects jaundice severity and estimates bilirubin levels. Our trained regression model provides quantitative bilirubin predictions directly from image embeddings."

**[VISUAL: Navigate to Cry Analysis tab, upload audio]**

> "A 5-second cry recording is processed by HeAR, which extracts 512-dimensional embeddings. A trained classifier identifies the cry type -- hungry, pain, discomfort, burping, or tired -- and derives asphyxia risk from distress patterns."

---

### AGENTIC WORKFLOW (1:55 - 2:30)

**[VISUAL: Navigate to Agentic Workflow tab -- THIS IS KEY FOR SPECIAL PRIZE]**

> "What makes NEXUS different is the 6-agent clinical workflow. Watch as each agent processes the case and explains its reasoning."

**[ACTION: Configure patient, upload image, run workflow]**

> "The Triage Agent scores danger signs with comorbidity detection and demographic risk assessment. The Image Agent invokes MedSigLIP with trained classifiers. The Audio Agent uses HeAR embeddings for cry classification. The Protocol Agent applies WHO IMNCI guidelines with comorbidity analysis. The Referral Agent matches facility capabilities and plans pre-referral actions. And the Synthesis Agent uses MedGemma 1.5 to produce a unified clinical recommendation."

**[VISUAL: Expand reasoning traces -- show step-by-step logic per agent]**

> "Every decision is explainable. Each agent's reasoning trace creates a complete audit trail."

---

### BILIRUBIN REGRESSION (2:30 - 2:42)

**[VISUAL: Show metrics comparison or notebook output]**

> "We also trained a 3-layer MLP with BatchNorm on frozen MedSigLIP embeddings for continuous bilirubin regression -- a novel task. Trained on 2,235 images with ground truth serum bilirubin, it achieves MAE of 2.56 mg/dL and Pearson r of 0.78, replacing color heuristics with learned medical features."

---

### EDGE + IMPACT (2:42 - 3:00)

**[VISUAL: Toggle Edge AI Mode in sidebar, show metrics banner]**

> "For deployment, models are INT8 quantized to under 289 megabytes total. MedGemma runs in 4-bit mode using just 2 GB of VRAM. NEXUS runs offline on a $100 Android phone."

**[VISUAL: Edge AI metrics cards showing size reductions]**

> "690,000 health workers. 500 million patient encounters. Every mother deserves a specialist. Every baby deserves a chance. NEXUS makes it possible."

**[VISUAL: NEXUS logo]**

> "Built with Google HAI-DEF for the MedGemma Impact Challenge."

---

## Recording Notes

### Setup
- Record against the LIVE Streamlit demo (locally or HuggingFace Spaces)
- Resolution: 1920x1080 at 30fps
- Record voiceover separately for clean audio
- Use sample data from `submission/video/assets/`

### Demo Data
1. Conjunctiva image from `data/raw/eyes-defy-anemia/`
2. Neonatal skin image from `data/raw/neojaundice/images/`
3. Cry audio from `data/raw/donate-a-cry/`

### Post-Production
- [ ] Record all screen sections
- [ ] Record voiceover
- [ ] Add statistics overlay for opening
- [ ] Verify total duration is under 3:00
- [ ] Export at 1080p
- [ ] Upload to YouTube (unlisted) or Google Drive

---

*Script Version: 3.0*
*Last Updated: February 4, 2026*
