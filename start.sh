#!/bin/bash

echo "🚀 Starting Adversarial Testing Playground..."

# Check if Python and Node.js are installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

# Install dependencies if needed
echo "📦 Installing dependencies..."
npm run install:all

# Start both servers
echo "🔥 Starting servers..."
echo "   Backend: http://localhost:5000"
echo "   Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Run both servers concurrently
npm run dev
