#!/bin/bash

# Start Frontend Script for Balder Trading App

echo "🚀 Starting Balder Frontend..."

cd "$(dirname "$0")/frontend"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Start development server
echo "✅ Starting React dev server on http://localhost:5173"
echo "🔐 Login passcode: 000"
echo ""
npm run dev

