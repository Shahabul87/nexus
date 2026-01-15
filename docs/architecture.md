# System Architecture

## Overview

MedAssist CHW is built as an offline-first mobile application with optional cloud sync capabilities. The architecture prioritizes:

1. **Edge-first inference** - All AI models run on-device
2. **Offline functionality** - Full feature set without internet
3. **Agentic workflow** - Multi-agent orchestration for clinical workflows
4. **Privacy by design** - Patient data stays on device

## High-Level Architecture

```
+================================================================+
|                     MedAssist CHW Mobile App                    |
+================================================================+
|                                                                  |
|  +------------------+  +------------------+  +------------------+ |
|  |   UI Layer       |  |   State Mgmt     |  |   Offline DB     | |
|  |   (React Native) |  |   (Zustand)      |  |   (WatermelonDB) | |
|  +------------------+  +------------------+  +------------------+ |
|                              |                                   |
|  +-----------------------------------------------------------+  |
|  |                    Agent Orchestrator                      |  |
|  |  - Workflow Management   - Context Tracking                |  |
|  |  - Tool Selection        - Response Generation             |  |
|  +-----------------------------------------------------------+  |
|                              |                                   |
|  +-----------------------------------------------------------+  |
|  |                    HAI-DEF Model Layer                     |  |
|  +-----------------------------------------------------------+  |
|  |  +-------------+  +-------------+  +-------------+         |  |
|  |  | MedGemma    |  | MedSigLIP   |  | HeAR        |         |  |
|  |  | 4B (INT4)   |  | (INT8)      |  | (INT8)      |         |  |
|  |  +-------------+  +-------------+  +-------------+         |  |
|  |  +-------------+  +-------------+                          |  |
|  |  | Derm        |  | CXR         |                          |  |
|  |  | Foundation  |  | Foundation  |                          |  |
|  |  +-------------+  +-------------+                          |  |
|  +-----------------------------------------------------------+  |
|                              |                                   |
|  +-----------------------------------------------------------+  |
|  |              On-Device Inference Engine                    |  |
|  |  - TensorFlow Lite Runtime                                 |  |
|  |  - Model Quantization (INT4/INT8)                          |  |
|  |  - Dynamic Model Loading                                   |  |
|  +-----------------------------------------------------------+  |
|                                                                  |
+==================================================================+
```

## Component Details

### 1. UI Layer (React Native)

**Technology:** React Native with TypeScript

**Key Components:**
- `HomeScreen` - Dashboard and quick actions
- `AssessmentScreen` - Patient evaluation workflow
- `PatientListScreen` - Offline patient records
- `ProtocolScreen` - Clinical guidelines reference

**Design Principles:**
- Large touch targets for field use
- High contrast for outdoor visibility
- Minimal text input (prefer selections)
- Support for local languages

### 2. State Management (Zustand)

**Stores:**
- `patientStore` - Current patient context
- `assessmentStore` - Ongoing assessment state
- `workflowStore` - Agent workflow state
- `syncStore` - Sync queue and status

### 3. Offline Database (WatermelonDB)

**Schema:**
```typescript
// Patient record
Patient {
  id: string
  name: string
  age: number
  gender: string
  village: string
  createdAt: Date
  syncStatus: 'pending' | 'synced'
}

// Assessment record
Assessment {
  id: string
  patientId: string
  type: 'skin' | 'respiratory' | 'general'
  images: string[]  // Base64 or local paths
  audioFiles: string[]
  symptoms: object
  aiAnalysis: object
  recommendation: string
  referralNeeded: boolean
  createdAt: Date
}
```

### 4. Agent Orchestrator

The orchestrator manages multi-agent workflows for patient encounters.

**Agents:**
1. **TriageAgent** - Initial assessment and routing
2. **ImageAgent** - Visual analysis (skin, wounds, eyes)
3. **AudioAgent** - Respiratory sound analysis
4. **SymptomAgent** - Guided symptom collection
5. **SynthesizerAgent** - Multi-modal reasoning
6. **ProtocolAgent** - Treatment protocol navigation
7. **ReferralAgent** - Severity and referral decisions

**Workflow Example:**
```
Patient presents with cough and skin rash

1. TriageAgent: Identifies respiratory + dermatological concerns
2. AudioAgent: Records and analyzes cough (possible TB pattern)
3. ImageAgent: Analyzes skin rash (possible fungal infection)
4. SymptomAgent: Collects duration, fever, weight loss
5. SynthesizerAgent: Combines findings, considers co-morbidity
6. ProtocolAgent: Retrieves TB screening + antifungal protocols
7. ReferralAgent: Recommends urgent TB testing referral
```

### 5. HAI-DEF Model Layer

**Model Specifications:**

| Model | Original | Optimized | Latency Target |
|-------|----------|-----------|----------------|
| MedGemma 4B | 8GB | 2GB (INT4) | <3s |
| MedSigLIP | 1.6GB | 400MB (INT8) | <500ms |
| HeAR | 350MB | 100MB (INT8) | <200ms |
| Derm Foundation | 400MB | 150MB (INT8) | <500ms |
| CXR Foundation | 500MB | 200MB (INT8) | <500ms |

**Loading Strategy:**
- Core models (MedGemma, MedSigLIP) loaded at startup
- Specialized models (HeAR, Derm, CXR) loaded on-demand
- LRU cache for model memory management

### 6. Inference Engine

**Runtime:** TensorFlow Lite with GPU delegate

**Optimizations:**
- INT4/INT8 quantization
- GPU acceleration where available
- Batched inference for multiple images
- Streaming inference for audio

## Data Flow

### Patient Assessment Flow

```
[CHW Opens App]
       |
       v
[Select/Create Patient]
       |
       v
[TriageAgent: "What brings the patient today?"]
       |
       +---> Skin complaint ----> [ImageAgent] ---+
       |                                          |
       +---> Cough/Breathing ---> [AudioAgent] ---+
       |                                          |
       +---> General illness ---> [SymptomAgent] -+
                                                  |
                                                  v
                                    [SynthesizerAgent]
                                    (Multi-modal reasoning)
                                          |
                                          v
                                    [ProtocolAgent]
                                    (Treatment guidance)
                                          |
                                          v
                                    [ReferralAgent]
                                    (Severity assessment)
                                          |
                                          v
                                 [Documentation & Sync]
```

## Security & Privacy

### On-Device Data Protection
- All patient data encrypted at rest (SQLCipher)
- No data transmitted without explicit user action
- Biometric/PIN lock for app access

### Sync Security
- TLS 1.3 for all network communication
- End-to-end encryption for patient data
- Consent-based data sharing

### Model Security
- Model weights stored encrypted
- Inference runs in isolated process
- No model outputs leave device without consent

## Deployment Architecture

### Mobile Deployment
```
App Store / Play Store
       |
       v
[MedAssist CHW APK/IPA]
       |
       +---> Core App Bundle (~50MB)
       |
       +---> Model Pack (downloaded separately, ~3GB)
```

### Optional Backend (Sync Server)
```
[Firebase / Custom Backend]
       |
       +---> Authentication Service
       |
       +---> Sync Service (patient records)
       |
       +---> Analytics Service (anonymized)
       |
       +---> Model Update Service
```

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| App startup | <5s | Cold start to usable |
| Image analysis | <2s | Capture to result |
| Audio analysis | <1s | Recording to result |
| Full assessment | <5min | Complete patient encounter |
| Battery usage | <10%/hour | Active use |
| Storage | <5GB | All models + 1000 patients |

## Future Considerations

1. **Model Updates** - OTA model updates without app update
2. **Federated Learning** - Learn from field data while preserving privacy
3. **Multi-language** - Support for 10+ languages common in CHW settings
4. **Wearable Integration** - Connect to pulse oximeters, thermometers
5. **Offline Maps** - Referral facility locations without internet
