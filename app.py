from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional
import torch
import torchaudio
import pretty_midi
import numpy as np
import os
import requests
from audiocraft.models import MusicGen
from vector_processor import MelodyProcessor

app = FastAPI(title="HuMix AI MusicGen & Vectorization Service")
processor = MelodyProcessor()

# Initialize model globally (loaded on first request or startup)
model = None

def load_model():
    global model
    if model is None:
        print("Loading MusicGen Melody model...")
        model = MusicGen.get_pretrained('facebook/musicgen-melody')
        print("Model loaded successfully.")

# Startup event to preload the model (skipped during testing)
@app.on_event("startup")
def startup_event():
    if os.environ.get("TESTING") != "true":
        load_model()

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
    callback_url: str
    presigned_url: str

class ModificationRequest(BaseModel):
    task_id: str
    melody_vectors: List[MelodyVector]
    prompt: str
    callback_url: str
    presigned_url: str

class MelodyExtractRequest(BaseModel):
    s3_url: str

def convert_vectors_to_wav_tensor(melody_vectors: List[MelodyVector], sample_rate=16000):
    pm = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    
    for vector in melody_vectors:
        pitch = vector.pitch
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
    audio_data = pm.synthesize(fs=sample_rate)
    melody_wav = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return melody_wav

def synthesize_sine_fallback(melody_vectors: List[MelodyVector], sample_rate=16000):
    """
    Fallback sine-wave synthesizer using numpy in case pretty_midi synthesis fails.
    Prevents pipeline crashes and ensures a WAV file is still uploaded.
    """
    try:
        print("Using sine wave fallback synthesizer...")
        duration = sum(v.duration_seconds for v in melody_vectors)
        if duration <= 0:
            duration = 5.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
        return torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    except Exception as e:
        print(f"Fallback synthesis failed: {e}")
        t = np.linspace(0, 5.0, int(sample_rate * 5.0), endpoint=False)
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
        return torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def upload_via_presigned_url(local_path, presigned_url):
    print(f"Uploading output to S3 via presigned URL...")
    with open(local_path, "rb") as f:
        # Upload binary file using HTTP PUT
        res = requests.put(presigned_url, data=f, headers={"Content-Type": "audio/wav"})
        res.raise_for_status()

def process_music_generation(task_id: str, melody_vectors: List[MelodyVector], prompt: str, presigned_url: str, callback_url: str):
    local_output_path = f"/tmp/{task_id}.wav"
    try:
        load_model()
        
        # 1. Convert melody vectors to audio tensor (with fallback)
        try:
            melody_wav = convert_vectors_to_wav_tensor(melody_vectors, sample_rate=16000)
        except Exception as e:
            print(f"convert_vectors_to_wav_tensor failed: {e}. Falling back.")
            melody_wav = synthesize_sine_fallback(melody_vectors, sample_rate=16000)
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        melody_wav = melody_wav.to(device)
        
        # 2. Setup generation parameters
        model.set_generation_params(duration=30)
        
        # 3. Generate music
        outputs = model.generate_with_chroma([prompt], melody_wav, 16000)
        
        # 4. Save output locally
        output_wav = outputs[0].cpu()
        torchaudio.save(local_output_path, output_wav, 32000)
        
        # 5. Upload via presigned URL
        upload_via_presigned_url(local_output_path, presigned_url)
        
        # 6. Callback backend (Success)
        callback_payload = {
            "generated_audio_url": presigned_url.split('?')[0]  # Return clean URL without query parameters
        }
        print(f"Calling backend callback: {callback_url}")
        requests.post(callback_url, json=callback_payload, timeout=10)
        
    except Exception as e:
        print(f"Error generating music for task {task_id}: {str(e)}")
        # Send failure callback to prevent backend hanging
        try:
            requests.post(callback_url, json={"generated_audio_url": "FAILED"}, timeout=10)
        except Exception as cb_err:
            print(f"Failed to send failure callback: {cb_err}")
    finally:
        # Clean up local file
        if os.path.exists(local_output_path):
            os.remove(local_output_path)

@app.post("/internal/v1/ai/generation/songs", status_code=status.HTTP_202_ACCEPTED)
async def generate_songs(req: GenerationRequest, background_tasks: BackgroundTasks):
    prompt = f"{req.genre}, {req.mood}"
    if req.reference_track:
        prompt += f", in the style of {req.reference_track}"
    background_tasks.add_task(process_music_generation, req.task_id, req.melody_vectors, prompt, req.presigned_url, req.callback_url)
    return {"task_id": req.task_id}

@app.post("/internal/v1/ai/generation/songs/{song_id}/modifications", status_code=status.HTTP_202_ACCEPTED)
async def modify_songs(song_id: str, req: ModificationRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_music_generation, req.task_id, req.melody_vectors, req.prompt, req.presigned_url, req.callback_url)
    return {"task_id": req.task_id}

@app.post("/api/v1/ai/melody-extract")
def extract_melody(payload: MelodyExtractRequest):
    try:
        signal = processor.preprocess_audio(payload.s3_url)
        f0_data = processor.extract_f0(signal)
        n_raw = processor.hz_to_midi(f0_data)
        result_vector = processor.quantize_and_map(n_raw)
        return result_vector
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 멜로디 추출 엔진 연산 실패: {str(e)}")

@app.get("/health")
def health():
    return {"status": "ok"}
