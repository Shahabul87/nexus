# NEXUS Mobile App

AI-Powered Maternal-Neonatal Care Platform built with React Native and Expo.

> **Offline-First**: Full functionality without internet connectivity

## Features

### Comprehensive Assessments (WHO IMNCI Based)
- **Maternal Assessment**: Complete prenatal checkup with danger signs screening
- **Newborn Assessment**: Full neonatal evaluation including jaundice and cry analysis

### Quick Screenings
- **Anemia Screening**: Analyze conjunctiva (inner eyelid) photos
- **Jaundice Detection**: Analyze skin photos with Kramer zone reference
- **Cry Analysis**: Record and analyze cry patterns for asphyxia signs
- **Combined Assessment**: Multi-modal evaluation with MedGemma synthesis

### Offline Capability
- **Local Database**: SQLite storage for all patient data and assessments
- **Sync Queue**: Automatic background synchronization when online
- **Edge AI**: On-device inference for core screenings (TFLite models)

## HAI-DEF Models Integration

| Model | Use Case | Implementation |
|-------|----------|----------------|
| **MedSigLIP** | Image analysis for anemia/jaundice | Zero-shot classification (INT8) |
| **HeAR** | Infant cry analysis for asphyxia | Acoustic embeddings + linear probe |
| **MedGemma** | Clinical synthesis | WHO IMNCI protocol integration |

## Agentic Workflow Engine

The app includes a 5-agent workflow system for comprehensive assessments:

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGENTIC WORKFLOW ENGINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐                    │
│   │ TRIAGE  │ →  │  IMAGE  │ →  │  AUDIO  │                    │
│   │  AGENT  │    │  AGENT  │    │  AGENT  │                    │
│   │         │    │         │    │         │                    │
│   │Risk     │    │MedSigLIP│    │  HeAR   │                    │
│   │Scoring  │    │Analysis │    │Analysis │                    │
│   └────┬────┘    └────┬────┘    └────┬────┘                    │
│        │              │              │                          │
│        └──────────────┼──────────────┘                          │
│                       ▼                                          │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              PROTOCOL AGENT (WHO IMNCI)                  │  │
│   │  • Apply clinical guidelines                             │  │
│   │  • Generate treatment recommendations                    │  │
│   │  • Determine follow-up schedule                          │  │
│   └─────────────────────┬───────────────────────────────────┘  │
│                         ▼                                        │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              REFERRAL AGENT                              │  │
│   │  • Synthesize all findings                               │  │
│   │  • Determine referral urgency                            │  │
│   │  • Generate clinical summary                             │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Node.js 18+ and npm
- iOS Simulator (Mac) or Android Emulator
- Expo Go app (for physical device testing)

## Quick Start

```bash
# Navigate to mobile directory
cd mobile

# Install dependencies
npm install

# Start Expo development server
npm start

# Run on iOS simulator (macOS only)
npm run ios

# Run on Android emulator
npm run android
```

## Project Structure

```
mobile/
├── App.tsx                          # Main app with navigation
├── app.json                         # Expo configuration
├── package.json                     # Dependencies
├── tsconfig.json                    # TypeScript config
│
└── src/
    ├── screens/
    │   ├── HomeScreen.tsx           # Assessment selection hub
    │   ├── PregnantWomanScreen.tsx  # Maternal assessment flow
    │   ├── NewbornScreen.tsx        # Neonatal assessment flow
    │   ├── AnemiaScreen.tsx         # Quick anemia screening
    │   ├── JaundiceScreen.tsx       # Quick jaundice detection
    │   ├── CryAnalysisScreen.tsx    # Quick cry analysis
    │   ├── CombinedAssessmentScreen.tsx  # Multi-modal assessment
    │   └── ResultsScreen.tsx        # Enhanced results display
    │
    ├── services/
    │   ├── index.ts                 # Service exports
    │   ├── edgeAI.ts                # On-device TFLite inference
    │   ├── nexusApi.ts              # Cloud API client
    │   ├── agenticWorkflow.ts       # 5-agent workflow engine
    │   ├── database.ts              # SQLite offline storage
    │   └── syncService.ts           # Background sync queue
    │
    ├── hooks/
    │   ├── index.ts                 # Hook exports
    │   └── useOffline.ts            # Offline status management
    │
    └── assets/                      # Images and icons
```

## API Configuration

Configure the backend API URL:

```bash
# Create .env file
echo "EXPO_PUBLIC_API_URL=http://localhost:8000" > .env
```

Or update `app.json`:
```json
{
  "expo": {
    "extra": {
      "apiUrl": "http://localhost:8000"
    }
  }
}
```

## Offline Mode

The app operates fully offline with these features:

### Local Database (SQLite)
- Patient records stored locally
- Assessment history with all results
- Automatic schema migrations

### Sync Queue
- Changes queued when offline
- Automatic sync when connectivity returns
- Retry logic with exponential backoff
- Conflict resolution

### Edge AI
- TFLite models for on-device inference
- INT8 quantization for low-end devices
- Pre-computed text embeddings for zero-shot

## Assessment Flows

### Maternal Assessment (4 Steps)
1. **Patient Info**: Gestational weeks, gravida, para
2. **Danger Signs**: WHO IMNCI maternal danger signs checklist
3. **Anemia Screening**: Conjunctiva photo capture and analysis
4. **Analysis**: AI synthesis with recommendations

### Newborn Assessment (5 Steps)
1. **Patient Info**: Age, birth weight, APGAR score
2. **Danger Signs**: WHO IMNCI newborn danger signs checklist
3. **Jaundice Screening**: Skin photo with Kramer zone reference
4. **Cry Analysis**: Audio recording for asphyxia detection
5. **Analysis**: Multi-modal AI synthesis

## Building for Production

```bash
# Install EAS CLI
npm install -g eas-cli

# Login to Expo
eas login

# Build for iOS
eas build --platform ios

# Build for Android
eas build --platform android

# Build APK for testing
eas build --platform android --profile preview
```

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| expo | ~52.0.0 | React Native framework |
| expo-camera | ~15.0.0 | Camera access |
| expo-av | ~14.0.0 | Audio recording |
| expo-sqlite | ~14.0.0 | Local database |
| expo-network | ~6.0.0 | Network state |
| expo-file-system | ~17.0.0 | File operations |
| zustand | ^4.5.0 | State management |

## Competition Info

This app is being developed for the **MedGemma Impact Challenge 2026** on Kaggle.

| Detail | Value |
|--------|-------|
| **Prize Pool** | $100,000 |
| **Main Prize** | $30,000 |
| **Edge AI Prize** | $5,000 |
| **Agentic Workflow Prize** | $5,000 |
| **Deadline** | February 24, 2026 |

## License

MIT License - See LICENSE file in the project root.

---

Built with Google HAI-DEF for the MedGemma Impact Challenge 2026
