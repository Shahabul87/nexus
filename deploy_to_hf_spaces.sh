#!/bin/bash
# Deploy NEXUS to HuggingFace Spaces
# Run this after creating the Space at https://huggingface.co/new-space
# Space name: nexus, SDK: Docker

set -e

HF_SPACE="https://huggingface.co/spaces/Shahabul/nexus"
echo "Deploying to $HF_SPACE ..."

# Clone the HF Space repo
git clone "$HF_SPACE" /tmp/nexus-hf-space 2>/dev/null || true
cd /tmp/nexus-hf-space

# Copy required files from project
PROJECT_DIR="${1:-D:/Shahabul/mydl/nexus}"

cp "$PROJECT_DIR/Dockerfile" .
cp "$PROJECT_DIR/README.md" .
cp "$PROJECT_DIR/requirements_spaces.txt" .
cp "$PROJECT_DIR/app.py" .
cp -r "$PROJECT_DIR/src/" src/

# Copy model metadata (not weights - they're gitignored)
mkdir -p models/linear_probes
cp "$PROJECT_DIR/models/linear_probes/"*.json models/linear_probes/ 2>/dev/null || true

# Push to HF
git add -A
git commit -m "Deploy NEXUS to HuggingFace Spaces"
git push

echo "Deployed! Visit: $HF_SPACE"
