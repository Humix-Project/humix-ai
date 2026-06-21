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
if [ -d "venv" ]; then
    echo "🐍 Virtual environment 'venv' already exists. Activating..."
    source venv/bin/activate
else
    echo "🐍 Creating new python virtual environment 'venv'..."
    python3 -m venv venv
    source venv/bin/activate
fi

# 4. Python packages installation
echo "📦 Installing python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install --no-deps audiocraft

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
