#!/bin/bash

# Start Both Backend and Frontend for Balder Trading App

echo "🚀 Starting Balder Full-Stack Application..."
echo ""

# Start backend in background
echo "Starting backend..."
./start-backend.sh &
BACKEND_PID=$!

# Wait a bit for backend to start
sleep 3

# Start frontend in foreground
echo "Starting frontend..."
./start-frontend.sh

# Kill background processes on exit
trap "kill $BACKEND_PID 2>/dev/null" EXIT

