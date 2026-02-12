# NEXUS Demo Video Script

**Target Duration**: 2 minutes 10 seconds
**Format**: Slideshow with voiceover

---

## Timing Overview

| Section | Time | Duration |
|---------|------|----------|
| Intro | 0:00 - 0:15 | 15s |
| Problem | 0:15 - 0:22 | 7s |
| HAI-DEF Models | 0:22 - 0:34 | 12s |
| Anemia Demo | 0:34 - 0:53 | 19s |
| Jaundice Demo | 0:53 - 1:09 | 16s |
| Cry Demo | 1:09 - 1:22 | 13s |
| Agentic Workflow | 1:22 - 1:49 | 27s |
| Edge AI | 1:49 - 1:56 | 7s |
| Closing | 1:56 - 2:09 | 13s |

---

## Voiceover Script

### INTRO (0:00 - 0:15)

> "NEXUS is an AI-powered screening platform for maternal and neonatal care. It uses three Google HAI-DEF models -- MedSigLIP, HeAR, and MedGemma -- in a 6-agent clinical workflow."

---

### PROBLEM (0:15 - 0:22)

> "800 mothers and 7,400 newborns die every day from preventable causes. NEXUS turns a smartphone into a diagnostic tool."

---

### HAI-DEF MODELS (0:22 - 0:34)

> "MedSigLIP handles image analysis. HeAR processes cry audio. MedGemma provides clinical reasoning. All three work together in every assessment."

---

### ANEMIA DEMO (0:34 - 0:53)

> "Upload a conjunctiva photo. MedSigLIP with a trained SVM classifier detects anemia at 91% confidence, estimates hemoglobin, and gives a clinical recommendation."

---

### JAUNDICE DEMO (0:53 - 1:09)

> "For jaundice, a skin photo is analyzed. Our novel bilirubin regression model -- a 3-layer MLP on frozen MedSigLIP embeddings -- predicts bilirubin at 11.6 mg/dL with Pearson r of 0.78."

---

### CRY DEMO (1:09 - 1:22)

> "HeAR extracts audio embeddings from a cry recording. A trained classifier identifies cry type and asphyxia risk in seconds."

---

### AGENTIC WORKFLOW (1:22 - 1:49)

> "The 6-agent pipeline runs Triage, Image Analysis, Audio Analysis, WHO Protocol, Referral Decision, and Clinical Synthesis. Each agent produces step-by-step reasoning traces -- a full audit trail. The final output: WHO classification, immediate actions, and referral guidance."

---

### EDGE AI (1:49 - 1:56)

> "Models are INT8 quantized to 289 megabytes total -- ready for offline use on low-cost phones."

---

### CLOSING (1:56 - 2:09)

> "NEXUS: three HAI-DEF models, six reasoning agents, one mission -- saving lives where it matters most. Built for the MedGemma Impact Challenge."

---

*Script Version: 4.0 -- February 10, 2026*
