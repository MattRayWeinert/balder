#!/bin/bash

# Simple backend start script

cd "$(dirname "$0")"

# Activate venv
source venv/bin/activate

# Start with uvicorn command directly
echo "🚀 Starting Balder Backend API..."
echo "📍 http://localhost:8000"
echo "📖 Docs: http://localhost:8000/docs"
echo ""

uvicorn main:app --host 0.0.0.0 --port 8000 --reload

