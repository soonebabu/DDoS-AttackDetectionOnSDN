#!/bin/bash

# Start all components of SDN DDoS Detection System

echo "Starting SDN DDoS Detection System..."
echo "======================================"

# Kill existing processes
echo "Cleaning up old processes..."
pkill -f "ryu-manager" 2>/dev/null
sudo pkill -f "mininet" 2>/dev/null
pkill -f "python.*app.py" 2>/dev/null
sudo mn -c 2>/dev/null

# Start Ryu controller
echo "Starting Ryu controller..."
ryu-manager ryu_controller/ddos_detector.py \
    --log-file logs/controller.log \
    --verbose \
    > logs/ryu_output.log 2>&1 &
RYU_PID=$!
sleep 3

if ps -p $RYU_PID > /dev/null; then
    echo "✓ Ryu controller started (PID: $RYU_PID)"
else
    echo "✗ Failed to start Ryu controller"
    exit 1
fi

# Start Mininet
echo "Starting Mininet..."
sudo python mininet_scripts/network.py \
    > logs/mininet.log 2>&1 &
MININET_PID=$!
sleep 5

if ps -p $MININET_PID > /dev/null; then
    echo "✓ Mininet started (PID: $MININET_PID)"
else
    echo "✗ Failed to start Mininet"
    kill $RYU_PID 2>/dev/null
    exit 1
fi

# Start dashboard
echo "Starting web dashboard..."
python web_dashboard/app.py \
    --host 0.0.0.0 \
    --port 5000 \
    > logs/dashboard.log 2>&1 &
DASHBOARD_PID=$!
sleep 2

if ps -p $DASHBOARD_PID > /dev/null; then
    echo "✓ Dashboard started (PID: $DASHBOARD_PID)"
    echo "  Dashboard URL: http://localhost:5000"
else
    echo "✗ Failed to start dashboard"
    kill $RYU_PID $MININET_PID 2>/dev/null
    exit 1
fi

echo ""
echo "System started successfully!"
echo "Components:"
echo "  - Ryu Controller: PID $RYU_PID"
echo "  - Mininet: PID $MININET_PID"
echo "  - Dashboard: PID $DASHBOARD_PID"
echo ""
echo "Dashboard: http://localhost:5000"
echo "Logs: logs/"
echo ""
echo "Press Ctrl+C to stop all components"

trap 'cleanup' INT

cleanup() {
    echo ""
    echo "Stopping system..."
    kill $RYU_PID $MININET_PID $DASHBOARD_PID 2>/dev/null
    sudo mn -c 2>/dev/null
    echo "System stopped"
    exit 0
}

while true; do
    sleep 10
done