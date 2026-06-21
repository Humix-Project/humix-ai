FROM python:3.11-slim AS smoke

WORKDIR /

COPY app.py /app.py
COPY handler.py /handler.py
COPY vector_processor.py /vector_processor.py
COPY vector_service.py /vector_service.py

CMD [ "python", "-c", "print('Docker smoke image built successfully')" ]

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS runtime

WORKDIR /

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
RUN pip install --no-cache-dir --no-deps audiocraft

COPY app.py /app.py
COPY handler.py /handler.py
COPY vector_processor.py /vector_processor.py
COPY vector_service.py /vector_service.py

EXPOSE 8000

CMD [ "python", "-u", "/handler.py" ]
