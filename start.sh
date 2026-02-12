#!/bin/bash

# OpenAgentTrace - Quick Start Script
# This script starts the backend server and frontend dashboard

set -e

echo "🚀 Starting OpenAgentTrace..."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Please run this script from the OpenAgentTrace root directory"
    exit 1
fi

# Create data directory
mkdir -p .oat

echo -e "${BLUE}Step 1: Installing Python dependencies...${NC}"
pip install -e ".[server]" --quiet

echo -e "${BLUE}Step 2: Starting backend server on port 8787...${NC}"
cd server
uvicorn main:app --port 8787 --host 0.0.0.0 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 2

# Check if backend is running
if ! curl -s http://localhost:8787/health > /dev/null; then
    echo "❌ Backend failed to start"
    exit 1
fi

echo -e "${GREEN}✓ Backend running at http://localhost:8787${NC}"

echo -e "${BLUE}Step 3: Installing frontend dependencies...${NC}"
cd ui
npm install --silent 2>/dev/null || npm install

echo -e "${BLUE}Step 4: Starting frontend dashboard on port 3000...${NC}"
npm run dev &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
sleep 3

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   OpenAgentTrace is running! 🎉${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "   Dashboard:  ${BLUE}http://localhost:3000${NC}"
echo -e "   API:        ${BLUE}http://localhost:8787${NC}"
echo -e "   Health:     ${BLUE}http://localhost:8787/health${NC}"
echo ""
echo -e "${YELLOW}   Run the demo agent:${NC}"
echo -e "   ${GREEN}python examples/demo_agent.py${NC}"
echo ""
echo -e "   Press ${YELLOW}Ctrl+C${NC} to stop all services"
echo ""

# Trap Ctrl+C to clean up
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for processes
wait
