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
import os
import requests
from audiocraft.models import MusicGen
from vector_processor import MelodyProcessor
from prompt_builder import build_description
from melody_synth import synthesize_melody
from audio_output import OUTPUT_SAMPLE_RATE, save_output_wav, write_wav
from melody_extend import extend_melody
from progress_report import EXTENDING_MELODY, GENERATING, LOADING_MODEL, UPLOADING, ProgressReporter
from section_regen import build_modify_text, fit_length, load_source_audio, regenerate_section

import threading

# MusicGen switches to extended generation beyond 30 seconds
MAX_DURATION_SECONDS = 30

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
    # Optional: stage-by-stage progress reports (progress_report.py)
    progress_url: Optional[str] = None

class ModificationRequest(BaseModel):
    # Section regeneration (#13): all times are on the source song's timeline
    task_id: str
    melody_vectors: List[MelodyVector] = Field(min_length=1)
    genre: str = ""
    mood: str = ""
    prompt: Optional[str] = None
    text_condition: Optional[str] = None
    edit_prompt: Optional[str] = None
    humming_end_seconds: Optional[float] = None
    source_audio_url: str
    section_start_seconds: float
    section_end_seconds: float
    callback_url: str
    presigned_url: str
    progress_url: Optional[str] = None

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
    progress_url: Optional[str] = None
    prompt: Optional[str] = None
    duration_seconds: Optional[float] = None
    s3_url: Optional[str] = None
    text_condition: Optional[str] = None
    edit_prompt: Optional[str] = None
    humming_end_seconds: Optional[float] = None
    source_audio_url: Optional[str] = None
    section_start_seconds: Optional[float] = None
    section_end_seconds: Optional[float] = None

class RunpodRequestEnvelope(BaseModel):
    input: RunpodInput

def convert_vectors_to_wav_tensor(melody_vectors: List[dict], sample_rate=16000):
    audio_data = synthesize_melody(melody_vectors, sample_rate=sample_rate)
    return torch.from_numpy(audio_data).unsqueeze(0).unsqueeze(0)

def upload_via_presigned_url(local_path, presigned_url):
    print(f"Uploading output to S3 via presigned URL...")
    with open(local_path, "rb") as f:
        # Upload binary file using HTTP PUT
        res = requests.put(presigned_url, data=f, headers={"Content-Type": "audio/wav"})
        res.raise_for_status()

def process_music_generation(task_id: str, melody_vectors: List[MelodyVector], description: str, duration_seconds: float, presigned_url: str, callback_url: str, progress_url: Optional[str] = None):
    with ProgressReporter(progress_url, task_id) as progress:
        _process_music_generation(task_id, melody_vectors, description, duration_seconds, presigned_url, callback_url, progress)

def _process_music_generation(task_id, melody_vectors, description, duration_seconds, presigned_url, callback_url, progress):
    print(f"\n[START] Starting music generation task: {task_id}")
    print(f"  Text condition: {description}")
    print(f"  Duration: {duration_seconds}s")
    print(f"  Melody vectors count: {len(melody_vectors)}")
    local_output_path = f"/tmp/{task_id}.wav"
    try:
        progress.start_stage(LOADING_MODEL)
        load_model()
        
        # 1. Extend a short humming to the song length with AMT (#25); on failure the humming is looped
        print(f"[{task_id}] Extending the humming melody...")
        progress.start_stage(EXTENDING_MELODY)
        extended_melody, humming_end_seconds = extend_melody(
            [v.model_dump() for v in melody_vectors], duration_seconds)

        # 2. Convert melody vectors to audio tensor (failure goes to the FAILED callback)
        print(f"[{task_id}] Synthesizing melody vectors to audio waveform...")
        melody_wav = convert_vectors_to_wav_tensor(extended_melody, sample_rate=16000)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        melody_wav = melody_wav.to(device)
        
        # 2. Setup generation parameters and generate music (Thread-Safe)
        with model_lock:
            print(f"[{task_id}] Running MusicGen Melody model inference...")
            progress.start_stage(GENERATING)
            model.set_generation_params(duration=duration_seconds)
            model.set_custom_progress_callback(progress.musicgen_callback())
            try:
                outputs = model.generate_with_chroma([description], melody_wav, 16000, progress=True)
            finally:
                model.set_custom_progress_callback(None)

        # 4. Save output locally
        print(f"[{task_id}] Generation complete. Saving output WAV locally...")
        # Save exactly the requested length (#21): audiocraft generates whole 1/50 s frames
        num_samples = round(duration_seconds * OUTPUT_SAMPLE_RATE)
        output_wav = fit_length(outputs[0].cpu(), num_samples)
        save_output_wav(local_output_path, output_wav)
        output_duration_seconds = num_samples / OUTPUT_SAMPLE_RATE
        
        # 5. Upload via presigned URL
        print(f"[{task_id}] Uploading generated file to S3 via presigned URL...")
        progress.start_stage(UPLOADING)
        upload_via_presigned_url(local_output_path, presigned_url)
        # No progress report may arrive after the completion callback
        progress.close()
        
        # 6. Callback backend (Success)
        clean_audio_url = presigned_url.split('?')[0]
        callback_payload = {
            "generated_audio_url": clean_audio_url,
            "duration_seconds": output_duration_seconds,
            # The melody this version used (backend stores it for regeneration)
            "extended_melody_vectors": extended_melody,
            "humming_end_seconds": humming_end_seconds
        }
        print(f"[{task_id}] Calling backend callback: {callback_url} (duration_seconds={output_duration_seconds})")
        requests.post(callback_url, json=callback_payload, timeout=10)
        print(f"[SUCCESS] Task {task_id} completed successfully!\n")
        
    except Exception as e:
        print(f"Error generating music for task {task_id}: {str(e)}")
        progress.close()
        # Send failure callback to prevent backend hanging
        try:
            requests.post(callback_url, json={"generated_audio_url": "FAILED"}, timeout=10)
        except Exception as cb_err:
            print(f"Failed to send failure callback: {cb_err}")
    finally:
        # Clean up local file
        if os.path.exists(local_output_path):
            os.remove(local_output_path)

def process_section_regeneration(task_id: str, req: ModificationRequest):
    """Regenerate [section_start_seconds, section_end_seconds) of a version and upload the merged song (#13)."""
    with ProgressReporter(req.progress_url, task_id) as progress:
        _process_section_regeneration(task_id, req, progress)

def _process_section_regeneration(task_id: str, req: ModificationRequest, progress):
    description = build_modify_text(req.text_condition, req.edit_prompt,
                                    build_description(req.genre, req.mood, req.prompt))
    print(f"\n[START] Starting section regeneration task: {task_id}")
    print(f"  Text condition: {description}")
    print(f"  Section: {req.section_start_seconds}~{req.section_end_seconds}s")
    local_output_path = f"/tmp/{task_id}.wav"
    try:
        progress.start_stage(LOADING_MODEL)
        load_model()
        melody_vectors = [v.model_dump() for v in req.melody_vectors]
        source = load_source_audio(req.source_audio_url)
        # The heartbeat keeps reporting while the section is generated
        progress.start_stage(GENERATING)
        with model_lock:
            merged = regenerate_section(model, description, source, melody_vectors,
                                        req.section_start_seconds, req.section_end_seconds, synthesize_melody,
                                        humming_end_seconds=req.humming_end_seconds)
        # Only the generated part was limited in the merge; the source part keeps its saved samples
        write_wav(local_output_path, merged)
        progress.start_stage(UPLOADING)
        upload_via_presigned_url(local_output_path, req.presigned_url)
        progress.close()

        callback_payload = {
            "generated_audio_url": req.presigned_url.split('?')[0],
            "duration_seconds": merged.shape[-1] / OUTPUT_SAMPLE_RATE,
            # The melody is not changed yet (AMT infilling comes after #25), so the received one is sent back
            "extended_melody_vectors": melody_vectors,
            "humming_end_seconds": req.humming_end_seconds,
            "text_condition": description,
        }
        requests.post(req.callback_url, json=callback_payload, timeout=10)
        print(f"[SUCCESS] Task {task_id} completed successfully!\n")
    except Exception as e:
        print(f"Error regenerating section for task {task_id}: {str(e)}")
        progress.close()
        try:
            requests.post(req.callback_url, json={"generated_audio_url": "FAILED"}, timeout=10)
        except Exception as cb_err:
            print(f"Failed to send failure callback: {cb_err}")
    finally:
        if os.path.exists(local_output_path):
            os.remove(local_output_path)

@app.post("/internal/v1/ai/generation/songs", status_code=status.HTTP_202_ACCEPTED)
async def generate_songs(req: GenerationRequest, background_tasks: BackgroundTasks):
    description = build_description(req.genre, req.mood, req.prompt)
    background_tasks.add_task(process_music_generation, req.task_id, req.melody_vectors, description, req.duration_seconds, req.presigned_url, req.callback_url, req.progress_url)
    return {"task_id": req.task_id}

@app.post("/internal/v1/ai/generation/songs/{song_id}/modifications", status_code=status.HTTP_202_ACCEPTED)
async def modify_songs(song_id: str, req: ModificationRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_section_regeneration, req.task_id, req)
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
        
        if inp.action == "modify":
            try:
                req = ModificationRequest(**{**inp.model_dump(exclude_none=True), "task_id": task_id})
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid modify request: {e}")
            background_tasks.add_task(process_section_regeneration, task_id, req)
            return {"id": task_id, "status": "IN_QUEUE"}

        if inp.duration_seconds is None or not 0 < inp.duration_seconds <= MAX_DURATION_SECONDS:
            raise HTTPException(status_code=400, detail=f"duration_seconds is required and must be in (0, {MAX_DURATION_SECONDS}].")
        duration_seconds = inp.duration_seconds
        description = build_description(inp.genre, inp.mood, inp.prompt)

        background_tasks.add_task(
            process_music_generation,
            task_id,
            inp.melody_vectors or [],
            description,
            duration_seconds,
            inp.presigned_url,
            inp.callback_url,
            inp.progress_url
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
