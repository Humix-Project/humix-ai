FROM python:3.11-slim AS smoke

WORKDIR /

COPY app.py /app.py
COPY handler.py /handler.py
COPY vector_processor.py /vector_processor.py
COPY vector_service.py /vector_service.py
COPY prompt_builder.py /prompt_builder.py
COPY gemini_client.py /gemini_client.py
COPY melody_synth.py /melody_synth.py
COPY audio_output.py /audio_output.py
COPY section_regen.py /section_regen.py
COPY melody_extend.py /melody_extend.py
COPY melody_correct.py /melody_correct.py

CMD [ "python", "-c", "print('Docker smoke image built successfully')" ]

FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime AS runtime

WORKDIR /

# Suppress interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies, including fluidsynth and soundfont for pretty_midi synthesis
RUN apt-get update && apt-get install -y \
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

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
RUN pip install --no-cache-dir xformers==0.0.29.post3
RUN pip install --no-cache-dir --no-deps audiocraft==1.3.0
# AMT melody extension (#25): the package only pins transformers in its metadata and never imports it,
# so it is installed without dependencies (mido, tqdm, numpy are already present)
RUN pip install --no-cache-dir --no-deps git+https://github.com/jthickstun/anticipation.git

# Set up cache directory environment variables inside the container
ENV HF_HOME=/cache/huggingface
ENV TORCH_HOME=/cache/torch
ENV HF_HUB_DISABLE_PROGRESS_BARS=1
RUN mkdir -p /cache/huggingface /cache/torch

# Pre-download MusicGen Melody, EnCodec 32kHz, and T5-base text encoder weights during build
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='facebook/musicgen-melody'); snapshot_download(repo_id='facebook/encodec_32khz'); snapshot_download(repo_id='google-t5/t5-base'); snapshot_download(repo_id='stanford-crfm/music-medium-800k')"

COPY app.py /app.py
COPY handler.py /handler.py
COPY vector_processor.py /vector_processor.py
COPY vector_service.py /vector_service.py
COPY prompt_builder.py /prompt_builder.py
COPY gemini_client.py /gemini_client.py
COPY melody_synth.py /melody_synth.py
COPY audio_output.py /audio_output.py
COPY section_regen.py /section_regen.py
COPY melody_extend.py /melody_extend.py
COPY melody_correct.py /melody_correct.py

EXPOSE 8000

CMD [ "python", "-u", "/handler.py" ]
