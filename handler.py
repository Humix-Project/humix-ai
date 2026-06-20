import runpod
import torch
import torchaudio
import pretty_midi
import boto3
import os
import requests
from audiocraft.models import MusicGen

# Initialize model globally (loaded on first request or startup)
model = None

def load_model():
    global model
    if model is None:
        print("Loading MusicGen Melody model...")
        model = MusicGen.get_pretrained('facebook/musicgen-melody')
        print("Model loaded successfully.")

def convert_vectors_to_wav_tensor(melody_vectors, sample_rate=16000):
    """
    Converts MIDI-like vectors (notes) into a synthesized mono audio waveform tensor at 16kHz
    as expected by MusicGen's melody conditioning.
    """
    pm = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    
    for vector in melody_vectors:
        pitch = vector.get('pitch')
        # Support both 'onset_seconds' and 'start_time_seconds'
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

def handler(job):
    job_input = job['input']
    task_id = job_input.get("task_id", job.get("id"))
    melody_vectors = job_input.get("melody_vectors", [])
    genre = job_input.get("genre", "")
    mood = job_input.get("mood", "")
    reference_track = job_input.get("reference_track", None)
    callback_url = job_input.get("callback_url", None)
    
    try:
        load_model()
        
        # 1. Convert melody vectors to audio tensor
        melody_wav = convert_vectors_to_wav_tensor(melody_vectors, sample_rate=16000)
        
        # Move tensor to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        melody_wav = melody_wav.to(device)
        
        # 2. Setup generation parameters
        model.set_generation_params(duration=30)  # Default max 30s as per MusicGen spec
        
        # Combine genre, mood as prompt builder
        prompt = f"{genre}, {mood}"
        if reference_track:
            prompt += f", in the style of {reference_track}"
            
        # 3. Generate music
        outputs = model.generate_with_chroma([prompt], melody_wav, 16000)
        
        # 4. Save output locally (MusicGen outputs at 32kHz sampling rate)
        local_output_path = f"/tmp/{task_id}.wav"
        output_wav = outputs[0].cpu()
        torchaudio.save(local_output_path, output_wav, 32000)
        
        # 5. Upload to S3
        bucket = os.environ.get('AWS_S3_BUCKET')
        s3_key = f"generated/songs/{task_id}.wav"
        generated_audio_url = upload_to_s3(local_output_path, bucket, s3_key)
        
        # Clean up local file
        if os.path.exists(local_output_path):
            os.remove(local_output_path)
            
        # 6. Optional: Callback backend directly (allows compatibility without changing backend endpoints)
        if callback_url:
            try:
                callback_payload = {
                    "generated_audio_url": generated_audio_url
                }
                print(f"Triggering callback to: {callback_url}")
                requests.post(callback_url, json=callback_payload, timeout=10)
            except Exception as cb_err:
                print(f"Direct callback failed: {cb_err}")
                
        return {
            "status": "COMPLETED",
            "generated_audio_url": generated_audio_url
        }
        
    except Exception as e:
        error_msg = f"Generation failed for task {task_id}: {str(e)}"
        print(error_msg)
        return {
            "status": "FAILED",
            "error": error_msg
        }

# Start the RunPod serverless loop
runpod.serverless.start({"handler": handler})
