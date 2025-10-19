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

echo ""
echo "Releasing ports..."

# Release ports (sync with launch.sh)
FRONTEND_PORT=${FRONTEND_PORT:-4000}
for port in 8000 8100 8200 $FRONTEND_PORT; do
    if lsof -i:$port > /dev/null 2>&1; then
        echo "   Releasing port $port..."
        lsof -ti:$port | xargs kill -9 2>/dev/null || true
    fi
done

echo ""
echo "Checking for remaining processes..."
remaining=$(ps aux | grep -E "(api\.py|backend\.py|npm.*dev)" | grep -v grep | wc -l)
if [ "$remaining" -eq 0 ]; then
    echo "   All processes have been stopped."
else
    echo "   Warning: $remaining related processes are still running:"
    ps aux | grep -E "(api\.py|backend\.py|npm.*dev)" | grep -v grep
fi

echo ""
echo "System stopped successfully."
echo ""
echo "Log files are kept in the logs/ directory."
echo "To restart the system: ./launch.sh"
