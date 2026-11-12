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
import math
import os
import requests
from audiocraft.models import MusicGen
from vector_processor import MelodyProcessor
from prompt_builder import build_description
from melody_synth import synthesize_melody
from audio_output import OUTPUT_SAMPLE_RATE, save_output_wav, write_wav
from melody_extend import extend_melody
from progress_report import EXTENDING_MELODY, GENERATING, LOADING_MODEL, UPLOADING, ProgressReporter
from section_regen import build_modify_text, fit_length, load_source_audio, parse_section_seconds, regenerate_section

# MusicGen switches to extended generation beyond 30 seconds
MAX_DURATION_SECONDS = 30

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
    audio_data = synthesize_melody(melody_vectors, sample_rate=sample_rate)
    return torch.from_numpy(audio_data).unsqueeze(0).unsqueeze(0)

def upload_via_presigned_url(local_path, presigned_url):
    print(f"Uploading output to S3 via presigned URL...")
    with open(local_path, "rb") as f:
        res = requests.put(presigned_url, data=f, headers={"Content-Type": "audio/wav"})
        res.raise_for_status()

def parse_duration_seconds(value):
    # bool is a subclass of int, so reject it explicitly
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"duration_seconds must be a number, got: {value!r}")
    if not 0 < value <= MAX_DURATION_SECONDS:
        raise ValueError(f"duration_seconds must be in (0, {MAX_DURATION_SECONDS}], got: {value}")
    return float(value)

def send_failure_callback(callback_url):
    try:
        requests.post(callback_url, json={"generated_audio_url": "FAILED"}, timeout=10)
    except Exception as cb_err:
        print(f"Failed to send failure callback: {cb_err}")

def finish_with_upload(local_output_path, output_wav, presigned_url, callback_url, progress, extra_fields=None):
    """Upload the saved wav, send the success callback and return the job output."""
    progress.start_stage(UPLOADING)
    upload_via_presigned_url(local_output_path, presigned_url)
    # No progress report may arrive after the completion callback
    progress.close()
    clean_audio_url = presigned_url.split('?')[0]
    output_duration_seconds = output_wav.shape[-1] / OUTPUT_SAMPLE_RATE
    callback_payload = {
        "generated_audio_url": clean_audio_url,
        "duration_seconds": output_duration_seconds,
        **(extra_fields or {})
    }
    print(f"Calling backend callback: {callback_url} (duration_seconds={output_duration_seconds})")
    requests.post(callback_url, json=callback_payload, timeout=10)
    return {"status": "COMPLETED", **callback_payload}

def fail(callback_url, error_msg, progress=None):
    print(error_msg)
    if progress is not None:
        progress.close()
    # Send failure callback to prevent backend hanging
    send_failure_callback(callback_url)
    return {
        "status": "FAILED",
        "error": error_msg
    }

def handle_music_generation(job_input, job_id):
    task_id = job_input.get("task_id", job_id)
    with ProgressReporter(job_input.get("progress_url"), task_id) as progress:
        return _generate_music(job_input, task_id, progress)

def _generate_music(job_input, task_id, progress):
    melody_vectors = job_input.get("melody_vectors", [])
    genre = job_input.get("genre", "")
    mood = job_input.get("mood", "")
    prompt = job_input.get("prompt")
    callback_url = job_input.get("callback_url")
    presigned_url = job_input.get("presigned_url")

    # Validation
    if not callback_url or not presigned_url:
        raise ValueError("Both callback_url and presigned_url are required.")

    try:
        duration_seconds = parse_duration_seconds(job_input.get("duration_seconds"))
    except ValueError as e:
        return fail(callback_url, f"Invalid request: {e}", progress)

    progress.start_stage(LOADING_MODEL)
    load_model()
    local_output_path = f"/tmp/{task_id}.wav"
    try:
        description = build_description(genre, mood, prompt)
        print(f"Text condition: {description}")

        # 1. Extend a short humming to the song length with AMT (#25); on failure the humming is looped
        progress.start_stage(EXTENDING_MELODY)
        extended_melody, humming_end_seconds = extend_melody(melody_vectors, duration_seconds)

        # 2. Convert melody vectors to audio tensor (failure goes to the FAILED callback)
        melody_wav = convert_vectors_to_wav_tensor(extended_melody, sample_rate=16000)

        device = model.device if hasattr(model, 'device') else ("cuda" if torch.cuda.is_available() else "cpu")
        melody_wav = melody_wav.to(device)

        # 3. Generate music, reporting MusicGen's token progress
        progress.start_stage(GENERATING)
        model.set_generation_params(duration=duration_seconds)
        model.set_custom_progress_callback(progress.musicgen_callback())
        try:
            outputs = model.generate_with_chroma([description], melody_wav, 16000, progress=True)
        finally:
            model.set_custom_progress_callback(None)

        # 4. Save exactly the requested length (#21): audiocraft generates whole 1/50 s frames
        output_wav = fit_length(outputs[0].cpu(), round(duration_seconds * OUTPUT_SAMPLE_RATE))
        save_output_wav(local_output_path, output_wav)

        # 5. Upload and callback with the melody this version used (backend stores it for regeneration)
        return finish_with_upload(local_output_path, output_wav, presigned_url, callback_url, progress, {
            "extended_melody_vectors": extended_melody,
            "humming_end_seconds": humming_end_seconds,
        })
    except Exception as e:
        return fail(callback_url, f"Generation failed: {str(e)}", progress)
    finally:
        if os.path.exists(local_output_path):
            os.remove(local_output_path)

def handle_section_regeneration(job_input, job_id):
    """Regenerate [section_start_seconds, section_end_seconds) of a version and upload the merged song (#13)."""
    task_id = job_input.get("task_id", job_id)
    with ProgressReporter(job_input.get("progress_url"), task_id) as progress:
        return _regenerate_section(job_input, task_id, progress)

def _regenerate_section(job_input, task_id, progress):
    callback_url = job_input.get("callback_url")
    presigned_url = job_input.get("presigned_url")
    if not callback_url or not presigned_url:
        raise ValueError("Both callback_url and presigned_url are required.")

    try:
        melody_vectors = job_input.get("melody_vectors")
        source_audio_url = job_input.get("source_audio_url")
        if not melody_vectors:
            raise ValueError("melody_vectors is required.")
        if not source_audio_url:
            raise ValueError("source_audio_url is required.")
        start = parse_section_seconds(job_input.get("section_start_seconds"), "section_start_seconds")
        end = parse_section_seconds(job_input.get("section_end_seconds"), "section_end_seconds")
    except ValueError as e:
        return fail(callback_url, f"Invalid request: {e}", progress)

    humming_end_seconds = job_input.get("humming_end_seconds")
    description = build_modify_text(
        job_input.get("text_condition"), job_input.get("edit_prompt"),
        build_description(job_input.get("genre", ""), job_input.get("mood", ""), job_input.get("prompt")))

    progress.start_stage(LOADING_MODEL)
    load_model()
    local_output_path = f"/tmp/{task_id}.wav"
    try:
        print(f"Text condition: {description}")
        print(f"Section: {start}~{end}s (humming_end_seconds={humming_end_seconds})")
        source = load_source_audio(source_audio_url)
        # The heartbeat keeps reporting while the section is generated
        progress.start_stage(GENERATING)
        merged = regenerate_section(model, description, source, melody_vectors, start, end, synthesize_melody,
                                    humming_end_seconds=humming_end_seconds)
        # Only the generated part was limited in the merge; the source part keeps its saved samples
        write_wav(local_output_path, merged)
        return finish_with_upload(local_output_path, merged, presigned_url, callback_url, progress, {
            # The melody is not changed yet (AMT infilling comes after #25), so the received one is sent back
            "extended_melody_vectors": melody_vectors,
            "humming_end_seconds": humming_end_seconds,
            "text_condition": description,
        })
    except Exception as e:
        return fail(callback_url, f"Section regeneration failed: {str(e)}", progress)
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
    elif action == "generate":
        return handle_music_generation(job_input, job.get("id"))
    elif action == "modify":
        return handle_section_regeneration(job_input, job.get("id"))
    else:
        return {
            "status": "FAILED",
            "error": f"Invalid action: {action}"
        }

runpod.serverless.start({"handler": handler})
