import os
import sys

# Use the baked-in cache if available, otherwise fallback to RunPod's workspace cache
if os.path.exists("/cache/huggingface"):
    os.environ["HF_HOME"] = "/cache/huggingface"
elif os.path.exists("/workspace"):
    os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
    os.environ["XDG_CACHE_HOME"] = "/workspace/.cache"

if os.path.exists("/cache/torch"):
    os.environ["TORCH_HOME"] = "/cache/torch"
elif os.path.exists("/workspace"):
    os.environ["TORCH_HOME"] = "/workspace/.cache/torch"

from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional
import torch
import torchaudio
import pretty_midi
import numpy as np
import os
import requests
from audiocraft.models import MusicGen
from vector_processor import MelodyProcessor

import threading

# MusicGen switches to extended generation beyond 30 seconds
MAX_DURATION_SECONDS = 30
MODIFY_DURATION_SECONDS = 30
OUTPUT_SAMPLE_RATE = 32000

app = FastAPI(title="HuMix AI MusicGen & Vectorization Service")
processor = MelodyProcessor()

# Global lock to serialize access to the stateful MusicGen model in concurrent threads
model_lock = threading.Lock()

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

import uuid

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
    prompt: Optional[str] = None
    duration_seconds: float = Field(gt=0, le=MAX_DURATION_SECONDS)
    callback_url: str
    presigned_url: str

class ModificationRequest(BaseModel):
    task_id: str
    melody_vectors: List[MelodyVector]
    genre: str = ""
    mood: str = ""
    prompt: Optional[str] = None
    callback_url: str
    presigned_url: str

class MelodyExtractRequest(BaseModel):
    s3_url: str

class RunpodInput(BaseModel):
    action: Optional[str] = "generate"
    task_id: Optional[str] = None
    melody_vectors: Optional[List[MelodyVector]] = None
    genre: Optional[str] = None
    mood: Optional[str] = None
    callback_url: Optional[str] = None
    presigned_url: Optional[str] = None
    prompt: Optional[str] = None
    duration_seconds: Optional[float] = None
    s3_url: Optional[str] = None

class RunpodRequestEnvelope(BaseModel):
    input: RunpodInput

def convert_vectors_to_wav_tensor(melody_vectors: List[MelodyVector], sample_rate=16000):
    pm = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    melody_end = 0.0

    for vector in melody_vectors:
        pitch = vector.pitch
        onset = vector.onset_seconds if vector.onset_seconds is not None else vector.start_time_seconds
        if onset is None:
            onset = 0.0
        duration = vector.duration_seconds
        end = onset + duration
        # Rests still count toward the melody length
        melody_end = max(melody_end, end)

        # pitch 0 is a rest; synthesizing it would produce an 8.18Hz full-amplitude tone
        if pitch == 0:
            continue

        note = pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=onset,
            end=end
        )
        piano.notes.append(note)

    if not piano.notes:
        raise ValueError("melody_vectors has no pitched notes.")

    pm.instruments.append(piano)
    audio_data = pm.synthesize(fs=sample_rate)
    # Cut the ~1s release tail so the melody loops with the humming length
    melody_samples = round(melody_end * sample_rate)
    audio_data = audio_data[:melody_samples]
    if len(audio_data) < melody_samples:
        audio_data = np.pad(audio_data, (0, melody_samples - len(audio_data)))
    melody_wav = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return melody_wav

def upload_via_presigned_url(local_path, presigned_url):
    print(f"Uploading output to S3 via presigned URL...")
    with open(local_path, "rb") as f:
        # Upload binary file using HTTP PUT
        res = requests.put(presigned_url, data=f, headers={"Content-Type": "audio/wav"})
        res.raise_for_status()

def build_description(genre: Optional[str], mood: Optional[str], prompt: Optional[str]) -> str:
    # MusicGen takes a single text description per sample
    description = f"{genre or ''}, {mood or ''}"
    if prompt and prompt.strip():
        description += f", {prompt.strip()}"
    return description

def process_music_generation(task_id: str, melody_vectors: List[MelodyVector], description: str, duration_seconds: float, presigned_url: str, callback_url: str):
    print(f"\n[START] Starting music generation task: {task_id}")
    print(f"  Text condition: {description}")
    print(f"  Duration: {duration_seconds}s")
    print(f"  Melody vectors count: {len(melody_vectors)}")
    local_output_path = f"/tmp/{task_id}.wav"
    try:
        load_model()
        
        # 1. Convert melody vectors to audio tensor (failure goes to the FAILED callback)
        print(f"[{task_id}] Synthesizing melody vectors to audio waveform...")
        melody_wav = convert_vectors_to_wav_tensor(melody_vectors, sample_rate=16000)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        melody_wav = melody_wav.to(device)
        
        # 2. Setup generation parameters and generate music (Thread-Safe)
        with model_lock:
            print(f"[{task_id}] Running MusicGen Melody model inference...")
            model.set_generation_params(duration=duration_seconds)
            outputs = model.generate_with_chroma([description], melody_wav, 16000)

        # 4. Save output locally
        print(f"[{task_id}] Generation complete. Saving output WAV locally...")
        output_wav = outputs[0].cpu()
        torchaudio.save(local_output_path, output_wav, OUTPUT_SAMPLE_RATE)
        output_duration_seconds = output_wav.shape[-1] / OUTPUT_SAMPLE_RATE
        
        # 5. Upload via presigned URL
        print(f"[{task_id}] Uploading generated file to S3 via presigned URL...")
        upload_via_presigned_url(local_output_path, presigned_url)
        
        # 6. Callback backend (Success)
        clean_audio_url = presigned_url.split('?')[0]
        callback_payload = {
            "generated_audio_url": clean_audio_url,
            "duration_seconds": output_duration_seconds
        }
        print(f"[{task_id}] Calling backend callback: {callback_url} (duration_seconds={output_duration_seconds})")
        requests.post(callback_url, json=callback_payload, timeout=10)
        print(f"[SUCCESS] Task {task_id} completed successfully!\n")
        
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
    description = build_description(req.genre, req.mood, req.prompt)
    background_tasks.add_task(process_music_generation, req.task_id, req.melody_vectors, description, req.duration_seconds, req.presigned_url, req.callback_url)
    return {"task_id": req.task_id}

@app.post("/internal/v1/ai/generation/songs/{song_id}/modifications", status_code=status.HTTP_202_ACCEPTED)
async def modify_songs(song_id: str, req: ModificationRequest, background_tasks: BackgroundTasks):
    description = build_description(req.genre, req.mood, req.prompt)
    background_tasks.add_task(process_music_generation, req.task_id, req.melody_vectors, description, MODIFY_DURATION_SECONDS, req.presigned_url, req.callback_url)
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

@app.post("/run", status_code=status.HTTP_200_OK)
async def runpod_run(req: RunpodRequestEnvelope, background_tasks: BackgroundTasks):
    inp = req.input
    task_id = inp.task_id or f"mock_task_{uuid.uuid4().hex[:8]}"
    
    if inp.action in ["generate", "modify"]:
        if not inp.callback_url or not inp.presigned_url:
            raise HTTPException(status_code=400, detail="Both callback_url and presigned_url are required.")
        
        # duration_seconds is required for generate; modify keeps the fixed default
        if inp.action == "generate":
            if inp.duration_seconds is None or not 0 < inp.duration_seconds <= MAX_DURATION_SECONDS:
                raise HTTPException(status_code=400, detail=f"duration_seconds is required and must be in (0, {MAX_DURATION_SECONDS}].")
            duration_seconds = inp.duration_seconds
        else:
            duration_seconds = MODIFY_DURATION_SECONDS

        description = build_description(inp.genre, inp.mood, inp.prompt)

        background_tasks.add_task(
            process_music_generation,
            task_id,
            inp.melody_vectors or [],
            description,
            duration_seconds,
            inp.presigned_url,
            inp.callback_url
        )
        return {"id": task_id, "status": "IN_QUEUE"}
    else:
        raise HTTPException(status_code=400, detail=f"Invalid action for async run: {inp.action}")

@app.post("/runsync", status_code=status.HTTP_200_OK)
def runpod_runsync(req: RunpodRequestEnvelope):
    inp = req.input
    if inp.action == "melody-extract":
        if not inp.s3_url:
            raise HTTPException(status_code=400, detail="s3_url is required for melody-extract action.")
        try:
            signal = processor.preprocess_audio(inp.s3_url)
            f0_data = processor.extract_f0(signal)
            n_raw = processor.hz_to_midi(f0_data)
            result_vector = processor.quantize_and_map(n_raw)
            return {
                "id": f"mock_job_{uuid.uuid4().hex[:8]}",
                "status": "COMPLETED",
                "output": {
                    "result_vector": result_vector
                }
            }
        except Exception as e:
            return {
                "id": f"mock_job_{uuid.uuid4().hex[:8]}",
                "status": "FAILED",
                "error": f"AI 멜로디 추출 엔진 연산 실패: {str(e)}"
            }
    else:
        raise HTTPException(status_code=400, detail=f"Invalid action for runsync: {inp.action}")

@app.get("/health")
def health():
    return {"status": "ok"}
