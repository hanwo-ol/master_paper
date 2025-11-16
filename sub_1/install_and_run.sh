#!/bin/bash
# ============================================================================
# install_and_run.sh
# Automated Installation & Execution Script
# ============================================================================

echo "=========================================================================="
echo "REFINED BAYESIAN VAR RESEARCH - INSTALLATION & EXECUTION"
echo "=========================================================================="

# 1. Environment Setup
echo ""
echo "【Step 1: Setting up Python virtual environment】"
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

echo "✓ Virtual environment activated"

# 2. Install Dependencies
echo ""
echo "【Step 2: Installing dependencies】"
pip install --upgrade pip
pip install -r config/requirements.txt

echo "✓ Dependencies installed"

# 3. Create Data Directories
echo ""
echo "【Step 3: Creating data directories】"
mkdir -p data
mkdir -p results
mkdir -p figures

echo "✓ Directories created"

# 4. Run Pipeline
echo ""
echo "【Step 4: Running research pipeline】"
echo ""

cd src

# Run all stages
echo "Starting pipeline execution..."
python run_pipeline_refined.py

cd ..

# 5. Results Summary
echo ""
echo "=========================================================================="
echo "✅ EXECUTION COMPLETE!"
echo "=========================================================================="
echo ""
echo "📊 Results Location:"
echo "  - Data: ./data/"
echo "  - Results: ./results/"
echo "  - Figures: ./figures/"
echo ""
echo "📋 Output Files:"
echo "  - benchmark_results.csv"
echo "  - summary_report.txt"
echo "  - 5 visualization PNGs"
echo ""
echo "🎯 Next Steps:"
echo "  1. Review results in ./results/"
echo "  2. Analyze visualizations in ./figures/"
echo "  3. Use 7-question checklist: docs/RESEARCH_CHECKLIST.md"
echo "  4. Write research paper using the findings"
echo ""
echo "=========================================================================="
