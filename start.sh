#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "========================================================="
echo "🚀 RunPod GPU Pod Setup & Startup Script Starting"
echo "========================================================="

# 1. Setup cache directories on persistent /workspace volume
# RunPod GPU Pods only persist data inside /workspace. 
# Caching here prevents re-downloading 4.5GB of models on every pod restart.
export HF_HOME="/workspace/.cache/huggingface"
export TORCH_HOME="/workspace/.cache/torch"
export HF_HUB_DISABLE_PROGRESS_BARS=1

mkdir -p "$HF_HOME" "$TORCH_HOME"

echo "📍 HF Cache Path: $HF_HOME"
echo "📍 Torch Cache Path: $TORCH_HOME"

# 2. System dependencies check (ffmpeg, fluidsynth for pretty_midi)
if ! command -v ffmpeg &> /dev/null; then
    echo "📦 System dependencies (ffmpeg, etc.) not found. Installing..."
    # RunPod pods run as root, so sudo is generally not needed/available
    apt-get update && apt-get install -y \
        git \
        ffmpeg \
        fluidsynth \
        fluid-soundfont-gm \
        pkg-config \
        libavformat-dev \
        libavcodec-dev \
        libavdevice-dev \
        libavutil-dev \
        libswscale-dev \
        libswresample-dev \
        libavfilter-dev \
        && rm -rf /var/lib/apt/lists/*
else
    echo "✅ System dependencies (ffmpeg) are already installed."
fi

# 3. Python virtual environment setup
# NOTE: Do NOT use --system-site-packages to avoid torchaudio/torch version conflicts
# between the system-installed packages and the venv packages.
if [ -d "venv" ]; then
    echo "🐍 Virtual environment 'venv' already exists. Activating..."
    source venv/bin/activate
else
    echo "🐍 Creating new python virtual environment 'venv' (isolated, no system-site-packages)..."
    python3 -m venv venv
    source venv/bin/activate
fi

# Ensure system-site-packages are disabled in the venv configuration to prevent bleed-in
python3 -c "
import os
cfg = 'venv/pyvenv.cfg'
if os.path.exists(cfg):
    with open(cfg, 'r') as f:
        lines = f.readlines()
    with open(cfg, 'w') as f:
        for line in lines:
            if line.strip().startswith('include-system-site-packages'):
                f.write('include-system-site-packages = false\n')
            else:
                f.write(line)
"

# Prevent the system-installed torchaudio/torch from bleeding into the venv
# by unsetting PYTHONPATH so only the venv site-packages are used.
unset PYTHONPATH

# 4. Python packages installation
echo "📦 Installing python dependencies..."
pip install --upgrade pip

# Install torch + torchaudio together first to guarantee version compatibility.
# This installs them into the venv (isolated from /usr/local/lib/python3.12/dist-packages).
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
TORCHAUDIO_VER=$(python3 -c "import torchaudio; print(torchaudio.__version__)" 2>/dev/null || echo "")

if [ -z "$TORCH_VER" ] || [ -z "$TORCHAUDIO_VER" ]; then
    echo "🔧 Installing torch + torchaudio into venv (this ensures version compatibility)..."
    # Install the CUDA 12.1 build of torch/torchaudio (matches RunPod's CUDA 12.x images).
    # If your pod uses a different CUDA version, change the index URL accordingly.
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "✅ torch ($TORCH_VER) and torchaudio ($TORCHAUDIO_VER) already present in venv."
fi

# Install remaining requirements, excluding torch/torchaudio/xformers (already handled above)
grep -vE "^(torch|torchaudio|xformers)" requirements.txt > temp_requirements.txt
pip install -r temp_requirements.txt
rm temp_requirements.txt

# Install audiocraft without pulling in conflicting torch builds
pip install --no-deps audiocraft

# Check if xformers is available and fully compatible with PyTorch
if ! python3 -c "import xformers; from xformers import ops" &> /dev/null; then
    echo "🔧 xformers is missing or incompatible. Attempting to install compatible version (0.0.28.post3)..."
    pip install xformers==0.0.28.post3 || pip install xformers || echo "⚠️ Failed to install xformers. Proceeding using native PyTorch attention."
else
    echo "✅ Compatible xformers is already installed."
fi

# 5. Pre-download weights to persistent cache
echo "📥 Pre-downloading model weights to persistent volume..."
python3 -c "
from huggingface_hub import snapshot_download
print('Downloading facebook/musicgen-melody...')
snapshot_download(repo_id='facebook/musicgen-melody')
print('Downloading facebook/encodec_32khz...')
snapshot_download(repo_id='facebook/encodec_32khz')
print('Downloading google-t5/t5-base...')
snapshot_download(repo_id='t5-base')
print('✨ All models pre-downloaded successfully!')
"

# 6. Start FastAPI application
echo "========================================================="
echo "🔥 Starting FastAPI Server on http://0.0.0.0:8000"
echo "========================================================="
exec uvicorn app:app --host 0.0.0.0 --port 8000
