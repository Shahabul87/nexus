# Entry Points

This document describes how to run the MedAssist CHW project.

## Prerequisites

1. Python 3.10+ installed
2. Node.js 18+ installed
3. React Native CLI installed
4. Android Studio or Xcode configured

## Environment Setup

```bash
# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set up mobile app
cd src/mobile
npm install
```

## Entry Points

### 1. Download Models

Downloads all HAI-DEF models from Hugging Face.

```bash
python models/download_models.py
```

**Options:**
- `--models medgemma,hear` - Download specific models only
- `--cache-dir ./cache` - Custom cache directory
- `--quantize` - Apply quantization after download

### 2. Optimize Models for Mobile

Converts and quantizes models for on-device inference.

```bash
python models/optimize/quantize_medgemma.py
python models/optimize/quantize_medsigLIP.py
python models/optimize/quantize_hear.py
```

**Output:** TFLite/ONNX models in `models/converted/`

### 3. Run Mobile App (Development)

```bash
cd src/mobile

# Start Metro bundler
npm start

# Run on Android (separate terminal)
npm run android

# Run on iOS (separate terminal)
npm run ios
```

### 4. Run Backend Server (Optional)

For sync functionality when internet is available.

```bash
cd src/backend
npm install
npm run dev
```

Server runs at `http://localhost:3000`

### 5. Run Tests

```bash
# Unit tests
npm run test:unit

# Integration tests
npm run test:integration

# E2E tests
npm run test:e2e

# Python tests
pytest tests/
```

### 6. Build for Production

```bash
# Android APK
cd src/mobile
npm run build:android

# iOS IPA
npm run build:ios
```

### 7. Generate Submission Package

```bash
./scripts/generate_submission.sh
```

Creates:
- `submission/final_package.zip`
- `directory_structure.txt`
- Validates all requirements

## Quick Start (Full Pipeline)

```bash
# One-command setup and run
./scripts/setup_environment.sh
python models/download_models.py
cd src/mobile && npm start
```

## Troubleshooting

### Model Download Issues
- Ensure you have accepted HAI-DEF terms of use
- Check Hugging Face authentication: `huggingface-cli login`

### Mobile Build Issues
- Android: Run `cd android && ./gradlew clean`
- iOS: Run `cd ios && pod install --repo-update`

### Memory Issues
- Use `--quantize` flag when downloading models
- Ensure device has 4GB+ RAM for inference
