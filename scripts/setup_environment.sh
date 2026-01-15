#!/bin/bash
# MedAssist CHW - Environment Setup Script
# Run: ./scripts/setup_environment.sh

set -e

echo "=========================================="
echo "MedAssist CHW - Environment Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo -e "\n${YELLOW}Checking prerequisites...${NC}"

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    echo -e "${GREEN}Python: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}Python 3 not found. Please install Python 3.10+${NC}"
    exit 1
fi

# Check Node.js
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo -e "${GREEN}Node.js: $NODE_VERSION${NC}"
else
    echo -e "${RED}Node.js not found. Please install Node.js 18+${NC}"
    exit 1
fi

# Check npm
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    echo -e "${GREEN}npm: $NPM_VERSION${NC}"
else
    echo -e "${RED}npm not found.${NC}"
    exit 1
fi

# Create Python virtual environment
echo -e "\n${YELLOW}Setting up Python virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created${NC}"
else
    echo -e "${YELLOW}Virtual environment already exists${NC}"
fi

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
echo -e "\n${YELLOW}Installing Python dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}Python dependencies installed${NC}"

# Install Hugging Face CLI
echo -e "\n${YELLOW}Setting up Hugging Face CLI...${NC}"
pip install huggingface_hub
echo -e "${GREEN}Hugging Face CLI installed${NC}"

# Check if logged in to Hugging Face
if ! huggingface-cli whoami &> /dev/null; then
    echo -e "${YELLOW}Please log in to Hugging Face to download models:${NC}"
    echo "Run: huggingface-cli login"
fi

# Create necessary directories
echo -e "\n${YELLOW}Creating project directories...${NC}"
mkdir -p models/cache models/converted models/checkpoints
mkdir -p data/raw data/processed
mkdir -p output
mkdir -p logs
echo -e "${GREEN}Directories created${NC}"

# Set up mobile app
echo -e "\n${YELLOW}Setting up mobile app...${NC}"
if [ -d "src/mobile" ]; then
    cd src/mobile
    if [ ! -d "node_modules" ]; then
        npm install
        echo -e "${GREEN}Mobile app dependencies installed${NC}"
    else
        echo -e "${YELLOW}Mobile app dependencies already installed${NC}"
    fi
    cd ../..
else
    echo -e "${YELLOW}Mobile app directory not found. Will be created during development.${NC}"
fi

# Summary
echo -e "\n=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo -e "\nNext steps:"
echo "1. Log in to Hugging Face: huggingface-cli login"
echo "2. Accept HAI-DEF terms at: https://huggingface.co/google/medgemma-4b-it"
echo "3. Download models: python models/download_models.py"
echo "4. Start mobile development: cd src/mobile && npm start"
echo ""
echo "For more details, see: README.md and entry_points.md"
