#!/bin/bash

# 에러 발생 시 즉시 종료 설정
set -e

echo "========================================================="
echo "🚀 RunPod GPU Pod Setup & Startup Script Starting"
echo "========================================================="

# 1. 영구 볼륨(/workspace)에 캐시 디렉토리 설정
export HF_HOME="/workspace/.cache/huggingface"
export TORCH_HOME="/workspace/.cache/torch"
export HF_HUB_DISABLE_PROGRESS_BARS=1

mkdir -p "$HF_HOME" "$TORCH_HOME"

echo "📍 HF Cache Path: $HF_HOME"
echo "📍 Torch Cache Path: $TORCH_HOME"

# 2. 시스템 의존성 체크 및 설치 (ffmpeg 등)
if ! command -v ffmpeg &> /dev/null; then
    echo "📦 System dependencies not found. Installing..."
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

# 3. 파이썬 가상환경(venv) 생성 및 활성화
if [ -d "venv" ]; then
    echo "🐍 Virtual environment 'venv' already exists. Activating..."
    source venv/bin/activate
else
    echo "🐍 Creating new python virtual environment 'venv'..."
    python3 -m venv venv
    source venv/bin/activate
fi

# 시스템 패키지가 venv로 유입되는 것 차단
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
unset PYTHONPATH

# 4. 파이썬 패키지 설치
echo "📦 Installing python dependencies..."
pip install --upgrade pip

# 임시로 에러 종료를 끄고 torch 설치 여부 확인
set +e
HAS_TORCH=$(python3 -c "import torch, torchaudio; print('OK')" 2>/dev/null || echo "NO")
set -e

if [ "$HAS_TORCH" = "NO" ]; then
    echo "🔧 Installing torch + torchaudio into venv..."
    # --no-cache-dir 옵션을 주어 설치 중 OOM(메모리 부족) 방지
    # RTX 5090 (Blackwell) 지원을 위해 CUDA 12.6 인덱스를 사용하며, 실패 시 CUDA 12.4나 기본 버전을 시도합니다.
    pip install --no-cache-dir torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126 || \
    pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 || \
    pip install --no-cache-dir torch torchaudio
else
    echo "✅ torch and torchaudio already present in venv."
fi

# 나머지 패키지 설치
if [ -f "requirements.txt" ]; then
    grep -vE "^(torch|torchaudio|xformers)" requirements.txt > temp_requirements.txt || true
    pip install --no-cache-dir -r temp_requirements.txt
    rm temp_requirements.txt
fi

# audiocraft 및 xformers 안전하게 설치
pip install --no-deps audiocraft

# xformers 존재 여부 및 호환성(임포트 에러 여부) 검사
set +e
HAS_XFORMERS=$(python3 -c "import xformers; from xformers import ops" 2>/dev/null || echo "NO")
set -e

if [ "$HAS_XFORMERS" = "NO" ]; then
    echo "🔧 Installing compatible xformers..."
    # PyTorch 2.6.0 호환 빌드인 0.0.29.post3 또는 PyTorch 2.5.1 호환 빌드인 0.0.28.post3를 설치합니다.
    pip install --no-cache-dir xformers==0.0.29.post3 || \
    pip install --no-cache-dir xformers==0.0.28.post3 || \
    pip install --no-cache-dir xformers || \
    echo "⚠️ xformers 설치 실패. 기본 어텐션을 사용합니다."
else
    echo "✅ 호환되는 xformers가 이미 설치되어 있습니다."
fi

# 5. 모델 가중치 미리 다운로드
echo "📥 Pre-downloading model weights to persistent volume..."
python3 -c "
from huggingface_hub import snapshot_download
import sys

try:
    print('Downloading facebook/musicgen-melody...')
    snapshot_download(repo_id='facebook/musicgen-melody')
    print('Downloading facebook/encodec_32khz...')
    snapshot_download(repo_id='facebook/encodec_32khz')
    print('Downloading google-t5/t5-base...')
    snapshot_download(repo_id='google-t5/t5-base')
    print('✨ All models pre-downloaded successfully!')
except Exception as e:
    print(f'❌ Model download failed: {e}')
    sys.exit(1)
"

# 6. FastAPI 애플리케이션 시작 (RunPod 외부 프록시 맵핑을 위해 8000 포트 유지)
echo "========================================================="
echo "🔥 Starting FastAPI Server on http://0.0.0.0:8000"
echo "========================================================="
exec uvicorn app:app --host 0.0.0.0 --port 8000
