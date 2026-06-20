from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
import torchaudio
import pretty_midi
import boto3
import os
import requests
from audiocraft.models import MusicGen

app = FastAPI(title="HuMix AI MusicGen Pod Service")

# Initialize model globally (loaded on first request or startup)
model = None

def load_model():
    global model
    if model is None:
        print("Loading MusicGen Melody model...")
        model = MusicGen.get_pretrained('facebook/musicgen-melody')
        print("Model loaded successfully.")

class MelodyVector(BaseModel):
    pitch: int
    onset_seconds: Optional[float] = None
    start_time_seconds: Optional[float] = None
    duration_seconds: float

class GenerationRequest(BaseModel):
    task_id: str
    melody_vectors: List[MelodyVector]
    genre: str
    mood: str
    reference_track: Optional[str] = None
    callback_url: Optional[str] = None

def convert_vectors_to_wav_tensor(melody_vectors: List[MelodyVector], sample_rate=16000):
    """
    Converts MIDI-like vectors (notes) into a synthesized mono audio waveform tensor at 16kHz
    as expected by MusicGen's melody conditioning.
    """
    pm = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    
    for vector in melody_vectors:
        pitch = vector.pitch
        # Support both 'onset_seconds' and 'start_time_seconds'
        onset = vector.onset_seconds if vector.onset_seconds is not None else vector.start_time_seconds
        if onset is None:
            onset = 0.0
        duration = vector.duration_seconds
        end = onset + duration
        
        note = pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=onset,
            end=end
        )
        piano.notes.append(note)
        
    pm.instruments.append(piano)
    # Synthesize MIDI to raw audio amplitude (numpy array)
    audio_data = pm.synthesize(fs=sample_rate)
    
    # Reshape to [B, C, T] (MusicGen expectation is 3D tensor: batch, channel, time)
    melody_wav = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return melody_wav

def upload_to_s3(local_path, bucket_name, s3_key):
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name=os.environ.get('AWS_DEFAULT_REGION', 'ap-northeast-2')
    )
    s3.upload_file(local_path, bucket_name, s3_key)
    region = os.environ.get('AWS_DEFAULT_REGION', 'ap-northeast-2')
    return f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_key}"

def process_music_generation(req: GenerationRequest):
    try:
        load_model()
        
        # 1. Convert melody vectors to audio tensor
        melody_wav = convert_vectors_to_wav_tensor(req.melody_vectors, sample_rate=16000)
        
        # Move tensor to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        melody_wav = melody_wav.to(device)
        
        # 2. Setup generation parameters
        model.set_generation_params(duration=30)  # Default max 30s as per MusicGen spec
        
        # Combine genre, mood as prompt builder
        prompt = f"{req.genre}, {req.mood}"
        if req.reference_track:
            prompt += f", in the style of {req.reference_track}"
            
        # 3. Generate music
        outputs = model.generate_with_chroma([prompt], melody_wav, 16000)
        
        # 4. Save output locally (MusicGen outputs at 32kHz sampling rate)
        local_output_path = f"/tmp/{req.task_id}.wav"
        output_wav = outputs[0].cpu()
        torchaudio.save(local_output_path, output_wav, 32000)
        
        # 5. Upload to S3
        bucket = os.environ.get('AWS_S3_BUCKET')
        s3_key = f"generated/songs/{req.task_id}.wav"
        generated_audio_url = upload_to_s3(local_output_path, bucket, s3_key)
        
        # Clean up local file
        if os.path.exists(local_output_path):
            os.remove(local_output_path)
            
        # 6. Callback backend
        if req.callback_url:
            callback_payload = {
                "generated_audio_url": generated_audio_url
            }
            print(f"Calling backend callback: {req.callback_url}")
            requests.post(req.callback_url, json=callback_payload, timeout=10)
            
    except Exception as e:
        print(f"Error generating music for task {req.task_id}: {str(e)}")

@app.post("/internal/v1/ai/generation/songs")
async def generate_songs(req: GenerationRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_music_generation, req)
    return {"task_id": req.task_id, "status": "PROCESSING"}

@app.get("/health")
def health():
    return {"status": "ok"}
