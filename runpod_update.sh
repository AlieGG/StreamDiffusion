#!/bin/bash
# Run this on the RunPod pod to pull latest changes and restart
set -e

cd /app

echo "=== Pulling latest changes ==="
git pull origin main

cd demo/realtime-img2img

echo "=== Rebuilding frontend ==="
cd frontend
npm install
npm run build
cd ..

echo "=== Restarting server ==="
pkill -f "python main.py" 2>/dev/null || true
sleep 2
nohup python main.py --port 7860 --host 0.0.0.0 --taesd --acceleration sfast > /tmp/streamdiffusion.log 2>&1 &
echo "Done! Server restarting..."
sleep 3
tail -20 /tmp/streamdiffusion.log
