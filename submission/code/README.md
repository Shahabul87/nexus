# NEXUS - Submission Code Package

This package contains all source code for the NEXUS AI-Powered Maternal-Neonatal Care Platform.

## Package Contents

```
submission/code/
├── README.md                    # This file
├── mobile/                      # React Native mobile application
│   ├── App.tsx                  # Main app entry
│   ├── src/
│   │   ├── screens/             # All UI screens
│   │   ├── services/            # API, AI, Database services
│   │   ├── hooks/               # Custom React hooks
│   │   └── components/          # Reusable UI components
│   └── package.json             # Dependencies
├── backend/                     # FastAPI backend (optional cloud deployment)
│   ├── main.py                  # API endpoints
│   ├── models/                  # ML model wrappers
│   └── requirements.txt         # Python dependencies
├── scripts/                     # Utility scripts
│   ├── download_datasets.py     # Dataset acquisition
│   ├── prepare_datasets.py      # Data preprocessing
│   └── edge/                    # Edge AI tools
│       ├── quantize_models.py   # INT8 quantization
│       └── convert_to_tflite.py # TFLite conversion
└── notebooks/                   # Jupyter notebooks
    ├── 01_anemia_detection.ipynb
    ├── 02_jaundice_detection.ipynb
    └── 03_cry_analysis.ipynb
```

## Quick Start

### Mobile App

```bash
cd mobile
npm install
npm start
```

### Backend (Optional)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

## HAI-DEF Model Integration

| Model | Usage | File |
|-------|-------|------|
| MedSigLIP | Image analysis (anemia, jaundice) | `mobile/src/services/edgeAI.ts` |
| HeAR | Cry audio analysis | `mobile/src/services/edgeAI.ts` |
| MedGemma | Clinical synthesis | `mobile/src/services/nexusApi.ts` |

## Agentic Workflow

The 5-agent system is implemented in:
- `mobile/src/services/agenticWorkflow.ts`

Agents:
1. **TriageAgent** - Initial risk assessment
2. **ImageAnalysisAgent** - MedSigLIP processing
3. **AudioAnalysisAgent** - HeAR processing
4. **ProtocolAgent** - WHO IMNCI guidelines
5. **ReferralAgent** - Final decision synthesis

## Offline Capability

- **Database**: `mobile/src/services/database.ts` (SQLite)
- **Sync**: `mobile/src/services/syncService.ts`
- **Edge AI**: `mobile/src/services/edgeAI.ts` (TFLite)

## Key Dependencies

### Mobile
- React Native + Expo
- expo-camera, expo-av, expo-sqlite
- zustand (state management)

### Backend
- FastAPI
- PyTorch, TensorFlow
- transformers (HAI-DEF models)

## License

CC BY 4.0 - See LICENSE in project root.

---

*Built for the MedGemma Impact Challenge 2026*
