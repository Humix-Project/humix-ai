# Use a PyTorch official image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /

# Install system dependencies (git for code references, ffmpeg for audio processing)
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install python packages
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Pre-download the Meta MusicGen Melody model to avoid download overhead during container start/cold-starts
RUN python -c "from audiocraft.models import MusicGen; MusicGen.get_pretrained('facebook/musicgen-melody')"

# Copy the application files
COPY app.py /app.py
COPY handler.py /handler.py

# Expose port (useful when running FastAPI on persistent GPU Pod)
EXPOSE 8000

# Default command runs the RunPod Serverless handler.
# When running on a persistent GPU Pod, this command can be overridden to run the FastAPI app:
# "uvicorn app:app --host 0.0.0.0 --port 8000"
CMD [ "python", "-u", "/handler.py" ]
