#!/bin/bash

echo "Starting Chat System"
echo "=========================="

# Ensure logs directory exists
mkdir -p logs

# Function to check and free ports
check_port() {
    local port=$1
    if lsof -i:$port > /dev/null 2>&1; then
        echo "Port $port is already in use, terminating process..."
        lsof -ti:$port | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
}

# Clean up old processes
echo "Cleaning old processes..."
pkill -f "python.*backend.py" 2>/dev/null || true
pkill -f "npm.*dev" 2>/dev/null || true

# Frontend port (default 4000, can override via FRONTEND_PORT)
FRONTEND_PORT=${FRONTEND_PORT:-4000}

# Check and clean ports
check_port 8000
check_port 8100
check_port 8200
check_port $FRONTEND_PORT

echo "Cleanup completed."
echo ""

# Start backend API (ports 8200, 8100)
echo "Starting backend API..."
nohup python3 backend.py > logs/backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"
echo "API running on: http://localhost:8200"
echo "File service running on: http://localhost:8100"

# Wait for backend to initialize
sleep 3

# Start frontend (React, default port: $FRONTEND_PORT)
echo ""
echo "Starting React frontend..."
cd chat || exit
nohup npm run dev -- -p $FRONTEND_PORT > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
echo "Frontend PID: $FRONTEND_PID"
echo "Frontend running on: http://localhost:$FRONTEND_PORT"

# Save PIDs
echo $BACKEND_PID > logs/backend.pid
echo $FRONTEND_PID > logs/frontend.pid

echo ""
echo "All services started successfully."
echo ""
echo "Service URLs:"
echo "  Mock API:     http://localhost:8000"
echo "  Backend API:  http://localhost:8200"
echo "  Frontend:     http://localhost:$FRONTEND_PORT"
echo "  File Service: http://localhost:8100"
echo ""
echo "Log files:"
echo "  Backend: logs/backend.log"
echo "  Frontend: logs/frontend.log"
echo ""
echo "Stop services: ./stop.sh"
