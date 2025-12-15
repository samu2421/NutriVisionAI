#!/bin/bash

# NutriVision AI - Quick Start Script
# This script activates the virtual environment and starts the Flask server

echo "============================================================"
echo "NutriVision AI - Quick Start"
echo "============================================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please create it first: python3 -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
echo "Checking dependencies..."
python -c "import flask" 2>/dev/null || {
    echo "❌ Flask not installed!"
    echo "Installing dependencies..."
    pip install -r requirements.txt
}

# Check if model exists
if [ -f "models/efficientnet_b1_indian.pth" ]; then
    echo "✓ Model found"
else
    echo "⚠️  Model not found - Demo mode will be available"
    echo "To train the model, run: python src/train.py"
fi

# Check if food data exists
if [ -f "data/food_data.json" ]; then
    echo "✓ Food data found"
else
    echo "❌ Food data not found!"
    exit 1
fi

# Create upload folder if it doesn't exist
mkdir -p static/uploads

echo "============================================================"
echo "Starting Flask server..."
echo "============================================================"

# Run the application
python app.py

