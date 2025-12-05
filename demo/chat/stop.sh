#!/bin/bash

echo "Stopping AI Chat System"
echo "======================="

# Stop service by PID file
stop_service() {
    local service_name=$1
    local pid_file=$2
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping $service_name (PID: $pid)..."
            kill "$pid" 2>/dev/null || true
            sleep 1
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                echo "   Force stopping $service_name..."
                kill -9 "$pid" 2>/dev/null || true
            fi
            echo "   $service_name stopped."
        else
            echo "   $service_name process not found."
        fi
        rm -f "$pid_file"
    else
        echo "   PID file for $service_name not found."
    fi
}

# Stop services
stop_service "Backend API" "logs/backend.pid"
stop_service "React Frontend" "logs/frontend.pid"

echo ""
echo "Cleaning up remaining processes..."

# Kill by process name (just in case)
pkill -f "python.*backend.py" 2>/dev/null && echo "   Cleaned up backend.py process." || true
pkill -f "npm.*dev" 2>/dev/null && echo "   Cleaned up npm dev process." || true
pkill -f "next.*dev" 2>/dev/null && echo "   Cleaned up next dev process." || true
pkill -f "next-server" 2>/dev/null && echo "   Cleaned up next-server process." || true
pkill -f "vite.*serve" 2>/dev/null && echo "   Cleaned up vite serve process." || true
pkill -f "node.*vite" 2>/dev/null && echo "   Cleaned up node-vite process." || true
pkill -f "react-scripts.*start" 2>/dev/null && echo "   Cleaned up react-scripts start process." || true

echo ""
echo "Releasing ports..."

# Release ports (sync with start.sh)
FRONTEND_PORT=${FRONTEND_PORT:-4000}
for port in 8100 8200 "$FRONTEND_PORT"; do
    # Only kill TCP LISTENers
    pids=$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "   Releasing port $port (PIDs: $pids)..."
        kill $pids 2>/dev/null || true
        sleep 1
        pids2=$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
        if [ -n "$pids2" ]; then
            echo "   Force releasing port $port (PIDs: $pids2)..."
            kill -9 $pids2 2>/dev/null || true
        fi
    fi
done

echo ""
echo "Checking for remaining processes..."
remaining=$(ps aux | grep -E "(api\.py|backend\.py|npm.*dev|next.*dev|next-server|vite.*serve|react-scripts.*start|node.*vite)" | grep -v grep | wc -l)
if [ "$remaining" -eq 0 ]; then
    echo "   All processes have been stopped."
else
    echo "   Warning: $remaining related processes are still running:"
    ps aux | grep -E "(api\.py|backend\.py|npm.*dev|next.*dev|next-server|vite.*serve|react-scripts.*start|node.*vite)" | grep -v grep
fi

echo ""
echo "System stopped successfully."
echo ""
echo "Log files are kept in the logs/ directory."
echo "To restart the system: ./start.sh"
