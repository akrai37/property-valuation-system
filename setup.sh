#!/bin/bash
# Smart Housing Valuation System - Quick Start Script
# This script automates the setup and running of the project

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Smart Housing Valuation System - Quick Start Setup       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if in correct directory
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}✗ Error: requirements.txt not found!${NC}"
    echo "Please run this script from the smart_housing_valuation directory"
    exit 1
fi

echo -e "${BLUE}Step 1: Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠ Virtual environment already exists${NC}"
fi

echo ""
echo -e "${BLUE}Step 2: Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

echo ""
echo -e "${BLUE}Step 3: Installing dependencies...${NC}"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo -e "${GREEN}✓ Dependencies installed${NC}"

echo ""
echo -e "${BLUE}Step 4: Training the model...${NC}"
python train_model.py
echo -e "${GREEN}✓ Model training completed${NC}"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo -e "${GREEN}  Setup Complete! Ready to run the web application  ${NC}"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. To start the web application, run:"
echo -e "   ${BLUE}streamlit run app.py${NC}"
echo ""
echo "2. To make batch predictions, run:"
echo -e "   ${BLUE}python predict.py${NC}"
echo ""
echo "3. The app will open at: http://localhost:8501"
echo ""
echo -e "${YELLOW}Note:${NC} Make sure your virtual environment is activated:"
echo -e "   ${BLUE}source venv/bin/activate${NC}"
echo ""
