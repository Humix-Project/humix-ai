FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /

# Install system dependencies, including fluidsynth and soundfont for pretty_midi synthesis
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    fluidsynth \
    fluid-soundfont-gm \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Pre-download the Meta MusicGen Melody model to reduce cold start latency
RUN python -c "from audiocraft.models import MusicGen; MusicGen.get_pretrained('facebook/musicgen-melody')"

COPY app.py /app.py
COPY handler.py /handler.py
COPY vector_processor.py /vector_processor.py
COPY vector_service.py /vector_service.py

EXPOSE 8000

CMD [ "python", "-u", "/handler.py" ]
