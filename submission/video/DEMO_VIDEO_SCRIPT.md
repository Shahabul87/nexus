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

**[VISUAL: Streamlit HAI-DEF Models Info tab]**

> "NEXUS uses three Google HAI-DEF models working together. MedSigLIP analyzes conjunctiva and skin images. HeAR processes infant cry audio. And MedGemma provides clinical reasoning, synthesizing findings into WHO protocol-aligned recommendations."

---

### LIVE DEMO: MATERNAL (0:55 - 1:25)

**[VISUAL: Navigate to Maternal Anemia Screening tab]**

> "Here's a live demo. A health worker uploads a conjunctiva photo."

**[ACTION: Upload sample conjunctiva image]**

> "MedSigLIP performs zero-shot classification using medical text prompts. Within seconds: anemia risk score, estimated hemoglobin, severity level, and a clinical recommendation."

**[VISUAL: Show results with confidence scores]**

---

### LIVE DEMO: NEWBORN (1:25 - 1:55)

**[VISUAL: Navigate to Neonatal Jaundice Detection tab]**

> "For neonatal assessment, a skin photo is analyzed for jaundice."

**[ACTION: Upload jaundice skin image, show results]**

> "MedSigLIP detects jaundice severity and estimates bilirubin levels. Our trained regression model provides quantitative bilirubin predictions directly from image embeddings."

**[VISUAL: Navigate to Cry Analysis tab, upload audio]**

> "A 5-second cry recording is processed by HeAR for birth asphyxia indicators."

---

### AGENTIC WORKFLOW (1:55 - 2:30)

**[VISUAL: Navigate to Agentic Workflow tab -- THIS IS KEY FOR SPECIAL PRIZE]**

> "What makes NEXUS different is the 6-agent clinical workflow. Watch as each agent processes the case and explains its reasoning."

**[ACTION: Configure patient, upload image, run workflow]**

> "The Triage Agent scores danger signs. The Image Analysis Agent invokes MedSigLIP. The Audio Agent uses HeAR. The Protocol Agent maps findings to WHO IMNCI classifications. The Referral Agent determines urgency. And the Synthesis Agent uses MedGemma to produce a unified clinical recommendation."

**[VISUAL: Expand reasoning traces -- show step-by-step logic per agent]**

> "Every decision is explainable. Each agent's reasoning trace creates a complete audit trail."

---

### BILIRUBIN REGRESSION (2:30 - 2:42)

**[VISUAL: Show metrics comparison or notebook output]**

> "We also fine-tuned MedSigLIP for continuous bilirubin regression -- a novel task. A lightweight 2-layer MLP trained on 2,235 images with ground truth serum bilirubin replaces color-based heuristics with learned medical features."

---

### EDGE + IMPACT (2:42 - 3:00)

**[VISUAL: Toggle Edge AI Mode in sidebar, show metrics banner]**

> "For deployment, models are INT8 quantized to 101 megabytes total. NEXUS runs offline on a $100 Android phone."

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

*Script Version: 2.0*
*Last Updated: January 30, 2026*
