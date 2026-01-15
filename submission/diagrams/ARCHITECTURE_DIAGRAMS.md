# NEXUS Architecture Diagrams

Visual representations of the NEXUS platform architecture for the MedGemma Impact Challenge submission.

---

## 1. System Architecture Overview

```
                          NEXUS PLATFORM ARCHITECTURE
    =========================================================================

    +-------------------------------------------------------------------------+
    |                           MOBILE APPLICATION                             |
    |                        (React Native + Expo)                             |
    +-------------------------------------------------------------------------+
    |                                                                          |
    |   +------------------+    +------------------+    +------------------+   |
    |   |   HOME SCREEN    |    |    MATERNAL      |    |    NEWBORN       |   |
    |   |                  |    |   ASSESSMENT     |    |   ASSESSMENT     |   |
    |   |  - Quick Tests   |    |  - Patient Info  |    |  - Patient Info  |   |
    |   |  - Full Assess   |    |  - Danger Signs  |    |  - Danger Signs  |   |
    |   |  - History       |    |  - Anemia Test   |    |  - Jaundice Test |   |
    |   +--------+---------+    |  - Synthesis     |    |  - Cry Analysis  |   |
    |            |              +--------+---------+    +--------+---------+   |
    |            |                       |                       |             |
    |            +-----------------------+-----------------------+             |
    |                                    |                                     |
    |                                    v                                     |
    |   +---------------------------------------------------------------------+|
    |   |                    AGENTIC WORKFLOW ENGINE                          ||
    |   |                                                                     ||
    |   |  +----------+   +----------+   +----------+   +----------+         ||
    |   |  |  TRIAGE  |-->|  IMAGE   |-->|  AUDIO   |-->| PROTOCOL |         ||
    |   |  |  AGENT   |   |  AGENT   |   |  AGENT   |   |  AGENT   |         ||
    |   |  +----------+   +----------+   +----------+   +----------+         ||
    |   |       |              |              |              |                ||
    |   |       v              v              v              v                ||
    |   |  Risk Score    MedSigLIP      HeAR Embed     WHO IMNCI             ||
    |   |                Analysis       Analysis       Guidelines            ||
    |   |       |              |              |              |                ||
    |   |       +-------+------+------+-------+------+-------+                ||
    |   |               |                            |                        ||
    |   |               v                            v                        ||
    |   |        +------------+            +----------------+                 ||
    |   |        |  REFERRAL  |            |   MedGemma     |                 ||
    |   |        |   AGENT    |<-----------|   Synthesis    |                 ||
    |   |        +------------+            +----------------+                 ||
    |   +---------------------------------------------------------------------+|
    |                                    |                                     |
    |                                    v                                     |
    |   +---------------------------------------------------------------------+|
    |   |                       OFFLINE LAYER                                 ||
    |   |                                                                     ||
    |   |   +----------------+    +----------------+    +----------------+    ||
    |   |   |    SQLite      |    |   Sync Queue   |    |   Edge AI      |    ||
    |   |   |   Database     |    |                |    |   (TFLite)     |    ||
    |   |   |                |    |  - Pending Ops |    |                |    ||
    |   |   | - Patients     |    |  - Retry Logic |    | - MedSigLIP    |    ||
    |   |   | - Assessments  |    |  - Conflict    |    | - HeAR         |    ||
    |   |   | - History      |    |    Resolution  |    | - Embeddings   |    ||
    |   |   +----------------+    +----------------+    +----------------+    ||
    |   +---------------------------------------------------------------------+|
    |                                                                          |
    +-------------------------------------------------------------------------+
                                         |
                                         | (When Online)
                                         v
    +-------------------------------------------------------------------------+
    |                           CLOUD BACKEND                                  |
    |                          (FastAPI + Python)                              |
    +-------------------------------------------------------------------------+
    |                                                                          |
    |   +----------------+    +----------------+    +----------------+         |
    |   |   MedSigLIP    |    |     HeAR       |    |   MedGemma     |         |
    |   |   Full Model   |    |   Full Model   |    |   4B Model     |         |
    |   +----------------+    +----------------+    +----------------+         |
    |                                                                          |
    +-------------------------------------------------------------------------+
```

---

## 2. Agentic Workflow Detail

```
                        5-AGENT WORKFLOW SYSTEM
    ================================================================

                              INPUT
                                |
                                v
    +-----------------------------------------------------------+
    |                      TRIAGE AGENT                          |
    |                                                            |
    |   Function: Initial risk stratification                    |
    |   Input: Patient info, danger signs                        |
    |   Output: Risk score (0-100), critical flags               |
    |                                                            |
    |   Algorithm:                                                |
    |   - Each danger sign = +10-20 points                       |
    |   - Age factors applied                                    |
    |   - Pregnancy factors applied                              |
    +-----------------------------------------------------------+
                                |
                                v
            +-------------------+-------------------+
            |                                       |
            v                                       v
    +---------------+                       +---------------+
    |  IMAGE AGENT  |                       |  AUDIO AGENT  |
    |               |                       |               |
    | Model:        |                       | Model:        |
    | MedSigLIP     |                       | HeAR          |
    |               |                       |               |
    | Tasks:        |                       | Tasks:        |
    | - Anemia      |                       | - Cry pattern |
    | - Jaundice    |                       | - Asphyxia    |
    |               |                       |   detection   |
    | Method:       |                       |               |
    | Zero-shot     |                       | Method:       |
    | classification|                       | Embedding +   |
    |               |                       | Linear probe  |
    +---------------+                       +---------------+
            |                                       |
            +-------------------+-------------------+
                                |
                                v
    +-----------------------------------------------------------+
    |                     PROTOCOL AGENT                         |
    |                                                            |
    |   Function: Apply WHO IMNCI clinical guidelines            |
    |   Input: All agent results                                 |
    |   Output: Classification, treatment protocol               |
    |                                                            |
    |   Classifications:                                         |
    |   +-------+---------------------------+-----------------+  |
    |   | Color |        Meaning            |     Action      |  |
    |   +-------+---------------------------+-----------------+  |
    |   |  RED  | Critical - Life threat    | Immediate refer |  |
    |   +-------+---------------------------+-----------------+  |
    |   |YELLOW | Warning - Needs attention | Treatment/refer |  |
    |   +-------+---------------------------+-----------------+  |
    |   | GREEN | Normal - Low risk         | Home care       |  |
    |   +-------+---------------------------+-----------------+  |
    +-----------------------------------------------------------+
                                |
                                v
    +-----------------------------------------------------------+
    |                     REFERRAL AGENT                         |
    |                                                            |
    |   Function: Synthesize all findings, final decision        |
    |   Input: Protocol output, all agent results                |
    |   Output: Clinical summary, referral decision, urgency     |
    |                                                            |
    |   Uses: MedGemma 4B for natural language synthesis         |
    |                                                            |
    |   Output Structure:                                        |
    |   - Primary diagnosis                                      |
    |   - Secondary findings                                     |
    |   - Recommended actions                                    |
    |   - Referral urgency (1-4 hours / 24-48 hours / None)     |
    |   - Follow-up schedule                                     |
    +-----------------------------------------------------------+
                                |
                                v
                             OUTPUT
```

---

## 3. HAI-DEF Model Integration

```
                    HAI-DEF MODEL INTEGRATION MAP
    ================================================================

    +-------------------+     +-------------------+     +-------------------+
    |                   |     |                   |     |                   |
    |     MedSigLIP     |     |       HeAR        |     |     MedGemma      |
    |                   |     |                   |     |                   |
    +-------------------+     +-------------------+     +-------------------+
    |                   |     |                   |     |                   |
    | Type: Vision      |     | Type: Audio       |     | Type: Language    |
    | Model             |     | Embeddings        |     | Model             |
    |                   |     |                   |     |                   |
    +-------------------+     +-------------------+     +-------------------+
    |                   |     |                   |     |                   |
    | Input:            |     | Input:            |     | Input:            |
    | - 224x224 images  |     | - 16kHz audio     |     | - Structured      |
    | - RGB normalized  |     | - 2-10 sec clips  |     |   text prompt     |
    |                   |     |                   |     |                   |
    +-------------------+     +-------------------+     +-------------------+
    |                   |     |                   |     |                   |
    | Method:           |     | Method:           |     | Method:           |
    | Zero-shot with    |     | Extract 768-dim   |     | Prompt            |
    | medical prompts   |     | embeddings, then  |     | engineering       |
    |                   |     | linear classifier |     | with context      |
    |                   |     |                   |     |                   |
    +-------------------+     +-------------------+     +-------------------+
    |                   |     |                   |     |                   |
    | Tasks:            |     | Tasks:            |     | Tasks:            |
    | - Anemia from     |     | - Asphyxia from   |     | - Synthesize      |
    |   conjunctiva     |     |   cry patterns    |     |   findings        |
    | - Jaundice from   |     | - Cry type        |     | - Generate        |
    |   skin/sclera     |     |   classification  |     |   recommendations |
    |                   |     |                   |     | - Referral        |
    +-------------------+     +-------------------+     +-------------------+
    |                   |     |                   |     |                   |
    | Edge Deploy:      |     | Edge Deploy:      |     | Edge Deploy:      |
    | INT8 quantized    |     | INT8 quantized    |     | Cloud-only or     |
    | ~100MB TFLite     |     | ~12MB TFLite      |     | 4B quantized      |
    |                   |     |                   |     |                   |
    +-------------------+     +-------------------+     +-------------------+
             |                         |                         |
             |                         |                         |
             v                         v                         v
    +-------------------------------------------------------------------+
    |                                                                    |
    |                      UNIFIED CLINICAL OUTPUT                       |
    |                                                                    |
    |   +--------------------+  +--------------------+  +-------------+  |
    |   | Anemia Assessment  |  | Asphyxia Risk     |  | Clinical    |  |
    |   | - Probability      |  | - Probability      |  | Summary     |  |
    |   | - Est. Hemoglobin  |  | - Cry type         |  | - Diagnosis |  |
    |   | - Severity         |  | - Abnormal signs   |  | - Actions   |  |
    |   +--------------------+  +--------------------+  +-------------+  |
    |                                                                    |
    +-------------------------------------------------------------------+
```

---

## 4. Data Flow Diagram

```
                         DATA FLOW ARCHITECTURE
    ================================================================

    USER INPUT                           PROCESSING                    OUTPUT
    =========                            ==========                    ======

    +-------------+
    | Patient     |
    | Information |----+
    +-------------+    |
                       |    +------------------+
    +-------------+    +--->|                  |
    | Danger      |-------->|   TRIAGE AGENT   |-----> Risk Score
    | Signs       |    +--->|                  |       Critical Flags
    +-------------+    |    +------------------+
                       |
    +-------------+    |    +------------------+    +------------------+
    | Conjunctiva |----+--->|                  |    |                  |
    | Photo       |         |   IMAGE AGENT    |--->| Anemia Result    |
    +-------------+         |   (MedSigLIP)    |    | - Probability    |
                            |                  |    | - Hemoglobin     |
    +-------------+         +------------------+    +------------------+
    | Skin Photo  |----+--->|                  |    |                  |
    +-------------+    |    |   IMAGE AGENT    |--->| Jaundice Result  |
                       |    |   (MedSigLIP)    |    | - Probability    |
                       |    |                  |    | - Bilirubin      |
                       |    +------------------+    +------------------+
                       |
    +-------------+    |    +------------------+    +------------------+
    | Cry Audio   |----+--->|                  |    |                  |
    | Recording   |         |   AUDIO AGENT    |--->| Asphyxia Result  |
    +-------------+         |   (HeAR)         |    | - Risk Score     |
                            |                  |    | - Cry Type       |
                            +------------------+    +------------------+
                                                            |
                                                            |
                            +------------------+            |
                            |                  |<-----------+
                            | PROTOCOL AGENT   |
                            | (WHO IMNCI)      |
                            |                  |
                            +------------------+
                                    |
                                    v
                            +------------------+    +------------------+
                            |                  |    |                  |
                            | REFERRAL AGENT   |--->| FINAL OUTPUT     |
                            | (MedGemma)       |    |                  |
                            |                  |    | - Classification |
                            +------------------+    | - Recommendation |
                                                    | - Referral       |
                                                    | - Follow-up      |
                                                    +------------------+
                                                            |
                                                            v
                            +------------------+    +------------------+
                            |                  |    |                  |
                            | LOCAL DATABASE   |<---| RESULTS SCREEN   |
                            | (SQLite)         |    |                  |
                            |                  |    +------------------+
                            +------------------+
                                    |
                                    | (When Online)
                                    v
                            +------------------+
                            |                  |
                            | SYNC TO CLOUD    |
                            |                  |
                            +------------------+
```

---

## 5. Offline-First Architecture

```
                     OFFLINE-FIRST ARCHITECTURE
    ================================================================

                            DEVICE STATE
                                 |
            +--------------------+--------------------+
            |                                         |
            v                                         v
        OFFLINE                                   ONLINE
            |                                         |
            v                                         v
    +---------------+                         +---------------+
    |               |                         |               |
    |   Edge AI     |                         |   Cloud API   |
    |   (TFLite)    |                         |   (FastAPI)   |
    |               |                         |               |
    | - MedSigLIP   |                         | - Full Models |
    |   INT8        |                         | - MedGemma    |
    | - HeAR INT8   |                         | - Higher Acc  |
    | - Pre-computed|                         |               |
    |   Embeddings  |                         |               |
    +---------------+                         +---------------+
            |                                         |
            v                                         v
    +---------------+                         +---------------+
    |               |                         |               |
    | Local Results |                         | Cloud Results |
    |               |                         |               |
    +---------------+                         +---------------+
            |                                         |
            +--------------------+--------------------+
                                 |
                                 v
                        +---------------+
                        |               |
                        |   SQLite DB   |
                        |               |
                        | - Patients    |
                        | - Assessments |
                        | - Sync Queue  |
                        |               |
                        +---------------+
                                 |
                                 v
                        +---------------+
                        |               |
                        |  SYNC SERVICE |
                        |               |
                        | If Online:    |
                        | - Process     |
                        |   pending     |
                        | - Resolve     |
                        |   conflicts   |
                        | - Update      |
                        |   cloud       |
                        |               |
                        +---------------+


    SYNC QUEUE STATES
    =================

    +----------+     +----------+     +----------+     +----------+
    | PENDING  | --> |PROCESSING| --> | SUCCESS  | or | FAILED   |
    +----------+     +----------+     +----------+     +----------+
                                                            |
                                                            v
                                                      Retry with
                                                      exponential
                                                      backoff
```

---

## 6. User Flow Diagrams

### Maternal Assessment Flow

```
    START
      |
      v
    +------------------+
    | Enter Patient    |
    | Information      |
    | - Name, Age      |
    | - Gestational wk |
    | - Gravida/Para   |
    +------------------+
      |
      v
    +------------------+
    | WHO IMNCI        |
    | Danger Signs     |
    | Checklist        |
    | (8 items)        |
    +------------------+
      |
      v
    +------------------+
    | Capture          |
    | Conjunctiva      |
    | Photo            |
    +------------------+
      |
      v
    +------------------+     +------------------+
    | MedSigLIP        |---->| Anemia           |
    | Analysis         |     | - Probability    |
    |                  |     | - Est. Hgb       |
    +------------------+     +------------------+
      |
      v
    +------------------+     +------------------+
    | MedGemma         |---->| Clinical         |
    | Synthesis        |     | Summary          |
    +------------------+     +------------------+
      |
      v
    +------------------+
    | Results Screen   |
    | - Classification |
    | - Recommendation |
    | - Referral       |
    +------------------+
      |
      v
     END
```

### Newborn Assessment Flow

```
    START
      |
      v
    +------------------+
    | Enter Patient    |
    | Information      |
    | - Name           |
    | - Age (hours)    |
    | - Birth weight   |
    | - APGAR score    |
    +------------------+
      |
      v
    +------------------+
    | WHO IMNCI        |
    | Danger Signs     |
    | Checklist        |
    | (10 items)       |
    +------------------+
      |
      v
    +------------------+
    | Capture Skin     |
    | Photo for        |
    | Jaundice         |
    +------------------+
      |
      v
    +------------------+     +------------------+
    | MedSigLIP        |---->| Jaundice         |
    | Analysis         |     | - Probability    |
    | (Kramer zones)   |     | - Est. Bilirubin |
    +------------------+     +------------------+
      |
      v
    +------------------+
    | Record Cry       |
    | Audio (5-10 sec) |
    +------------------+
      |
      v
    +------------------+     +------------------+
    | HeAR Analysis    |---->| Asphyxia         |
    |                  |     | - Risk score     |
    |                  |     | - Cry type       |
    +------------------+     +------------------+
      |
      v
    +------------------+     +------------------+
    | MedGemma         |---->| Clinical         |
    | Multi-modal      |     | Summary          |
    | Synthesis        |     |                  |
    +------------------+     +------------------+
      |
      v
    +------------------+
    | Results Screen   |
    | - Classification |
    | - Recommendation |
    | - Referral       |
    | - Follow-up      |
    +------------------+
      |
      v
     END
```

---

## 7. Deployment Architecture

```
                     DEPLOYMENT ARCHITECTURE
    ================================================================

    DEVELOPMENT                 STAGING                  PRODUCTION
    ===========                 =======                  ==========

    +-------------+         +-------------+          +-------------+
    |   Local     |         |   Expo      |          |   App       |
    |   Machine   |         |   Preview   |          |   Store     |
    |             |         |             |          |             |
    | npm start   |  --->   | EAS Build   |   --->   | APK/IPA     |
    |             |         | --preview   |          | Release     |
    +-------------+         +-------------+          +-------------+
          |                       |                        |
          v                       v                        v
    +-------------+         +-------------+          +-------------+
    |   Local     |         |   Staging   |          | Production  |
    |   Backend   |         |   API       |          | API         |
    |             |         |             |          |             |
    | localhost   |         | Railway     |          | Railway/    |
    | :8000       |         | staging     |          | GCP/AWS     |
    +-------------+         +-------------+          +-------------+


    MOBILE APP BUNDLE CONTENTS
    ==========================

    +--------------------------------------------------+
    |                    APK / IPA                      |
    +--------------------------------------------------+
    |                                                   |
    |   React Native Bundle          ~5 MB             |
    |   +---------------------------------------------+|
    |   | JavaScript Bundle                           ||
    |   | Assets (Icons, Images)                      ||
    |   +---------------------------------------------+|
    |                                                   |
    |   TFLite Models                ~115 MB           |
    |   +---------------------------------------------+|
    |   | MedSigLIP INT8            ~100 MB           ||
    |   | HeAR INT8                 ~12 MB            ||
    |   | Text Embeddings           ~3 KB             ||
    |   +---------------------------------------------+|
    |                                                   |
    |   Native Dependencies          ~20 MB            |
    |   +---------------------------------------------+|
    |   | expo-camera                                 ||
    |   | expo-av                                     ||
    |   | expo-sqlite                                 ||
    |   | expo-network                                ||
    |   +---------------------------------------------+|
    |                                                   |
    |   TOTAL APK SIZE:             ~140 MB            |
    |                                                   |
    +--------------------------------------------------+
```

---

## Usage Notes

These diagrams can be:
1. **Exported as images** using ASCII-to-image converters
2. **Converted to SVG** using tools like asciiflow
3. **Used in presentations** with monospace fonts
4. **Embedded in documentation** as code blocks

For the demo video, consider animating these diagrams to show:
- Data flow progression
- Agent activation sequence
- Offline/online transitions

---

*Created for the MedGemma Impact Challenge 2026*
*NEXUS: AI-Powered Maternal-Neonatal Care Platform*
