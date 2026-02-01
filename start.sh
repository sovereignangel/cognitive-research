#!/bin/bash
# Start both the API server and React dashboard

cd "$(dirname "$0")"

echo "Starting Consciousness Observatory..."
echo ""

# Start API server in background
echo "Starting API server on http://localhost:5000"
./venv/bin/python api/server.py &
API_PID=$!

# Give API a moment to start
sleep 2

# Start React dev server
echo "Starting dashboard on http://localhost:5173"
cd dashboard-app && npm run dev &
DASH_PID=$!

echo ""
echo "Dashboard: http://localhost:5173"
echo "API:       http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop both servers"

# Cleanup on exit
trap "kill $API_PID $DASH_PID 2>/dev/null" EXIT

wait
