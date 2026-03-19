#!/bin/bash
# RunPod setup script for StreamDiffusion
# Run once on a fresh RunPod PyTorch template pod
set -e

echo "=== Installing Node.js ==="
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs

echo "=== Cloning repo ==="
git clone https://github.com/AlieGG/StreamDiffusion.git /app
cd /app/demo/realtime-img2img

echo "=== Installing Python deps (preserving template torch/xformers) ==="
# Do NOT install torch/torchvision/xformers - the RunPod template already has the right versions
pip install --no-cache-dir \
    "diffusers==0.24.0" \
    "transformers==4.35.2" \
    "huggingface_hub==0.19.4" \
    "fastapi==0.104.1" \
    "uvicorn[standard]==0.24.0.post1" \
    "Pillow==10.1.0" \
    "peft==0.6.0" \
    "compel==2.0.2" \
    "markdown2" \
    "pydantic"

echo "=== Installing StreamDiffusion ==="
pip install --no-cache-dir --no-deps \
    git+https://github.com/AlieGG/StreamDiffusion.git@main#egg=streamdiffusion

echo "=== Installing stable-fast ==="
pip install --no-cache-dir \
    "stable_fast @ https://github.com/chengzeyi/stable-fast/releases/download/v0.0.15.post1/stable_fast-0.0.15.post1+torch211cu121-cp310-cp310-manylinux2014_x86_64.whl" \
    || echo "stable-fast install failed, continuing without it"

echo "=== Patching huggingface_hub compatibility ==="
PATCH_FILE="/usr/local/lib/python3.10/dist-packages/diffusers/utils/dynamic_modules_utils.py"
if [ -f "$PATCH_FILE" ]; then
    sed -i 's/from huggingface_hub import HfFolder, cached_download, hf_hub_download, model_info/from huggingface_hub import HfFolder, hf_hub_download, model_info\ntry:\n    from huggingface_hub import cached_download\nexcept ImportError:\n    cached_download = hf_hub_download/' "$PATCH_FILE" || true
fi

echo "=== Building frontend ==="
cd /app/demo/realtime-img2img/frontend
npm install
npm run build

echo "=== Starting server ==="
cd /app/demo/realtime-img2img
nohup python main.py --port 7860 --host 0.0.0.0 --taesd --acceleration sfast \
    > /tmp/streamdiffusion.log 2>&1 &
echo "Server PID: $!"
sleep 8
tail -20 /tmp/streamdiffusion.log
echo "=== Setup complete! ==="
