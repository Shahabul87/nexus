#!/bin/bash

# NEXUS Submission Package Script
# Creates a clean submission package for Kaggle

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/packaged"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PACKAGE_NAME="nexus_submission_${TIMESTAMP}"

echo "================================================"
echo "  NEXUS Submission Package Generator"
echo "  MedGemma Impact Challenge 2026"
echo "================================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR/$PACKAGE_NAME"

echo "[1/6] Copying mobile app source..."
mkdir -p "$OUTPUT_DIR/$PACKAGE_NAME/mobile"
cp -r "$PROJECT_ROOT/mobile/src" "$OUTPUT_DIR/$PACKAGE_NAME/mobile/"
cp "$PROJECT_ROOT/mobile/App.tsx" "$OUTPUT_DIR/$PACKAGE_NAME/mobile/"
cp "$PROJECT_ROOT/mobile/package.json" "$OUTPUT_DIR/$PACKAGE_NAME/mobile/"
cp "$PROJECT_ROOT/mobile/tsconfig.json" "$OUTPUT_DIR/$PACKAGE_NAME/mobile/"
cp "$PROJECT_ROOT/mobile/app.json" "$OUTPUT_DIR/$PACKAGE_NAME/mobile/"
cp "$PROJECT_ROOT/mobile/README.md" "$OUTPUT_DIR/$PACKAGE_NAME/mobile/"

echo "[2/6] Copying Python scripts..."
mkdir -p "$OUTPUT_DIR/$PACKAGE_NAME/scripts"
if [ -d "$PROJECT_ROOT/scripts" ]; then
    cp -r "$PROJECT_ROOT/scripts"/* "$OUTPUT_DIR/$PACKAGE_NAME/scripts/" 2>/dev/null || true
fi

echo "[2.5/6] Copying API server..."
mkdir -p "$OUTPUT_DIR/$PACKAGE_NAME/api"
if [ -d "$PROJECT_ROOT/api" ]; then
    cp -r "$PROJECT_ROOT/api"/* "$OUTPUT_DIR/$PACKAGE_NAME/api/" 2>/dev/null || true
fi

echo "[2.6/6] Copying edge models..."
mkdir -p "$OUTPUT_DIR/$PACKAGE_NAME/models/edge"
if [ -d "$PROJECT_ROOT/models/edge" ]; then
    cp -r "$PROJECT_ROOT/models/edge"/* "$OUTPUT_DIR/$PACKAGE_NAME/models/edge/" 2>/dev/null || true
fi

echo "[3/6] Copying source modules..."
mkdir -p "$OUTPUT_DIR/$PACKAGE_NAME/src"
if [ -d "$PROJECT_ROOT/src" ]; then
    cp -r "$PROJECT_ROOT/src"/* "$OUTPUT_DIR/$PACKAGE_NAME/src/" 2>/dev/null || true
fi

echo "[4/6] Copying notebooks..."
mkdir -p "$OUTPUT_DIR/$PACKAGE_NAME/notebooks"
if [ -d "$PROJECT_ROOT/notebooks" ]; then
    cp -r "$PROJECT_ROOT/notebooks"/*.ipynb "$OUTPUT_DIR/$PACKAGE_NAME/notebooks/" 2>/dev/null || true
fi

echo "[5/6] Copying documentation..."
cp "$PROJECT_ROOT/README.md" "$OUTPUT_DIR/$PACKAGE_NAME/"
cp "$PROJECT_ROOT/NEXUS_MASTER_PLAN.md" "$OUTPUT_DIR/$PACKAGE_NAME/" 2>/dev/null || true
cp "$PROJECT_ROOT/TECHNICAL_IMPLEMENTATION_GUIDE.md" "$OUTPUT_DIR/$PACKAGE_NAME/" 2>/dev/null || true
cp "$SCRIPT_DIR/README.md" "$OUTPUT_DIR/$PACKAGE_NAME/SUBMISSION_README.md"

# Copy submission materials
mkdir -p "$OUTPUT_DIR/$PACKAGE_NAME/submission"
cp -r "$PROJECT_ROOT/submission/writeup" "$OUTPUT_DIR/$PACKAGE_NAME/submission/" 2>/dev/null || true
cp -r "$PROJECT_ROOT/submission/diagrams" "$OUTPUT_DIR/$PACKAGE_NAME/submission/" 2>/dev/null || true
cp -r "$PROJECT_ROOT/submission/video" "$OUTPUT_DIR/$PACKAGE_NAME/submission/" 2>/dev/null || true

echo "[6/6] Creating archive..."
cd "$OUTPUT_DIR"
zip -r "${PACKAGE_NAME}.zip" "$PACKAGE_NAME" -x "*.DS_Store" -x "*node_modules*" -x "*.git*"

# Calculate size
SIZE=$(du -h "${PACKAGE_NAME}.zip" | cut -f1)

echo ""
echo "================================================"
echo "  Package Created Successfully!"
echo "================================================"
echo ""
echo "  Location: $OUTPUT_DIR/${PACKAGE_NAME}.zip"
echo "  Size: $SIZE"
echo ""
echo "  Contents:"
echo "  - Mobile App (React Native + Expo)"
echo "  - Python Scripts (HAI-DEF integration)"
echo "  - Jupyter Notebooks"
echo "  - Documentation"
echo "  - Submission Materials"
echo ""
echo "  Ready for Kaggle submission!"
echo "================================================"
