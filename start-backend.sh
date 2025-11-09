#!/bin/bash

# Start Backend Script for Balder Trading App

echo "🚀 Starting Balder Backend..."

cd "$(dirname "$0")/backend"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Start server
echo "✅ Starting FastAPI server on http://localhost:8000"
echo "📖 API Docs available at http://localhost:8000/docs"
echo ""

# Try to use uvicorn directly first (more reliable)
if command -v uvicorn &> /dev/null; then
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
else
    python main.py
fi

