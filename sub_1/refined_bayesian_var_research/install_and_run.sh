#!/bin/bash
# Windows/Unix compatible installation script

echo "REFINED BAYESIAN VAR RESEARCH - Installation"
echo "=============================================="
echo ""

# Step 1: Virtual Environment
echo "Step 1: Creating virtual environment..."
python -m venv venv

# Activate (Unix-like)
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

# Activate (Windows)
if [ -f venv\Scripts\activate.bat ]; then
    venv\Scripts\activate.bat
fi

echo "Virtual environment activated"
echo ""

# Step 2: Install dependencies
echo "Step 2: Installing dependencies..."
pip install --upgrade pip
pip install -r config/requirements.txt

echo ""
echo "Step 3: Running pipeline..."
cd src
python run_pipeline_refined.py

echo ""
echo "=============================================="
echo "EXECUTION COMPLETE!"
echo "Check ../results/ and ../figures/ for outputs"
echo "=============================================="
