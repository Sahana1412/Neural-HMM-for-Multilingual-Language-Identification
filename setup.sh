#!/bin/bash
# Quick setup script for Neural-HMM Language Identification project

set -e

echo "🚀 Neural-HMM Language Identification - Setup Script"
echo "=================================================="

# Check Python version
if ! command -v python3.11 &> /dev/null; then
    echo "❌ Python 3.11+ not found. Please install Python 3.11+"
    exit 1
fi

echo "✅ Python 3.11 found: $(python3.11 --version)"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3.11 -m venv venv
source venv/bin/activate || . venv/Scripts/activate 2>/dev/null || true

# Upgrade pip
echo "🔄 Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install package
echo "📥 Installing neural-hmm-lang-id..."
pip install -e .

# Install optional dependencies
echo "📚 Installing optional dependencies..."
pip install -e ".[dev,notebooks]" 2>/dev/null || echo "⚠️  Optional dependencies install skipped"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate venv: source venv/bin/activate"
echo "  2. Preprocess data: python main.py preprocess --config config/default.yaml"
echo "  3. Train models: make train-all"
echo "  4. View help: make help"
echo ""
echo "For quick start, see QUICKSTART.md"
