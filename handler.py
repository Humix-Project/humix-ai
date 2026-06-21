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

import runpod
import torch
import torchaudio
import pretty_midi
import numpy as np
import os
import requests
from audiocraft.models import MusicGen
from vector_processor import MelodyProcessor

# Initialize models globally
model = None
processor = MelodyProcessor()

def load_model():
    global model
    if model is None:
        print("Loading MusicGen Melody model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Targeting device for load: {device}")
        try:
            model = MusicGen.get_pretrained('facebook/musicgen-melody', device=device)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model on {device}: {e}. Retrying on CPU...")
            try:
                model = MusicGen.get_pretrained('facebook/musicgen-melody', device="cpu")
                print("Model loaded successfully on CPU.")
            except Exception as cpu_err:
                print(f"Failed to load model on CPU as well: {cpu_err}")
                raise cpu_err

# Removed pre-load model at container boot to support lazy loading and avoid CUDA initialization crashes during melody extraction.

def convert_vectors_to_wav_tensor(melody_vectors, sample_rate=16000):
    pm = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    
    for vector in melody_vectors:
        pitch = vector.get('pitch')
        onset = vector.get('onset_seconds') if vector.get('onset_seconds') is not None else vector.get('start_time_seconds')
        if onset is None:
            onset = 0.0
        duration = vector.get('duration_seconds', 0.5)
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

def synthesize_sine_fallback(melody_vectors, sample_rate=16000):
    try:
        print("Using sine wave fallback synthesizer...")
        duration = sum(v.get('duration_seconds', 0.5) for v in melody_vectors)
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
        res = requests.put(presigned_url, data=f, headers={"Content-Type": "audio/wav"})
        res.raise_for_status()

def handle_music_generation(job_input, job_id):
    load_model()
    task_id = job_input.get("task_id", job_id)
    melody_vectors = job_input.get("melody_vectors", [])
    genre = job_input.get("genre", "")
    mood = job_input.get("mood", "")
    prompt = job_input.get("prompt", "")  # for modification
    reference_track = job_input.get("reference_track", None)
    callback_url = job_input.get("callback_url")
    presigned_url = job_input.get("presigned_url")
    
    # Validation
    if not callback_url or not presigned_url:
        raise ValueError("Both callback_url and presigned_url are required.")
        
    local_output_path = f"/tmp/{task_id}.wav"
    try:
        # Resolve prompt
        if not prompt:
            prompt = f"{genre}, {mood}"
            if reference_track:
                prompt += f", in the style of {reference_track}"
                
        # 1. Convert melody vectors to audio tensor (with fallback)
        try:
            melody_wav = convert_vectors_to_wav_tensor(melody_vectors, sample_rate=16000)
        except Exception as e:
            print(f"convert_vectors_to_wav_tensor failed: {e}. Falling back.")
            melody_wav = synthesize_sine_fallback(melody_vectors, sample_rate=16000)
            
        device = model.device if hasattr(model, 'device') else ("cuda" if torch.cuda.is_available() else "cpu")
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
        clean_audio_url = presigned_url.split('?')[0]
        callback_payload = {
            "generated_audio_url": clean_audio_url
        }
        print(f"Calling backend callback: {callback_url}")
        requests.post(callback_url, json=callback_payload, timeout=10)
        
        return {
            "status": "COMPLETED",
            "generated_audio_url": clean_audio_url
        }
        
    except Exception as e:
        error_msg = f"Generation failed: {str(e)}"
        print(error_msg)
        # Send failure callback to prevent backend hanging
        try:
            requests.post(callback_url, json={"generated_audio_url": "FAILED"}, timeout=10)
        except Exception as cb_err:
            print(f"Failed to send failure callback: {cb_err}")
        return {
            "status": "FAILED",
            "error": error_msg
        }
    finally:
        if os.path.exists(local_output_path):
            os.remove(local_output_path)

def handle_melody_extraction(job_input):
    s3_url = job_input.get("s3_url")
    if not s3_url:
        raise ValueError("s3_url is required for melody-extract action.")
    
    signal = processor.preprocess_audio(s3_url)
    f0_data = processor.extract_f0(signal)
    n_raw = processor.hz_to_midi(f0_data)
    result_vector = processor.quantize_and_map(n_raw)
    return {
        "status": "COMPLETED",
        "result_vector": result_vector
    }

def handler(job):
    job_input = job['input']
    action = job_input.get("action", "generate")  # default action
    
    if action == "melody-extract":
        return handle_melody_extraction(job_input)
    elif action in ["generate", "modify"]:
        return handle_music_generation(job_input, job.get("id"))
    else:
        return {
            "status": "FAILED",
            "error": f"Invalid action: {action}"
        }

runpod.serverless.start({"handler": handler})
