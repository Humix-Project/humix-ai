"""Issue #14 PoC: compare how a song develops past a short humming.

Methods (same text condition, seed and humming for each):
  loop       current server behavior: chroma of the H-second humming is repeated to fill the song
  two_stage  candidate 1: generate_with_chroma for H seconds, then generate_continuation
             (no melody condition) until the full duration
  extended   candidate 2: generate_with_chroma once with the melody extended to the full duration
             by extend_melody.py (run that first in venv-amt)

Humming of length H is simulated by cropping the sample melody vectors to the first H seconds.

Example:
  python compare.py --samples samples/vector1.json samples/vector2.json --hum 5 10 20 --duration 30
"""
import argparse
import json
import time
from pathlib import Path

import xformers_stub  # noqa: F401  (must come before audiocraft)
import numpy as np
import pretty_midi
import torch
import torchaudio
from audiocraft.models import MusicGen

from melody_utils import crop_vectors, extended_name

MELODY_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 32000
MAX_DURATION_SECONDS = 30
METHODS = ('loop', 'two_stage', 'extended')


def build_description(genre, mood, prompt):
    # Same rule as handler.build_description
    description = f"{genre}, {mood}"
    if prompt and prompt.strip():
        description += f", {prompt.strip()}"
    return description


def synthesize_melody(melody_vectors, length_seconds, sample_rate=MELODY_SAMPLE_RATE):
    # Same synthesis as handler.convert_vectors_to_wav_tensor, but the length is the humming
    # length H (silence after the last note is part of the humming)
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'))
    for vector in melody_vectors:
        if vector['pitch'] == 0:
            continue
        onset = vector['onset_seconds']
        piano.notes.append(pretty_midi.Note(velocity=100, pitch=vector['pitch'], start=onset,
                                            end=onset + vector['duration_seconds']))
    if not piano.notes:
        raise ValueError("melody_vectors has no pitched notes.")
    pm.instruments.append(piano)

    num_samples = round(length_seconds * sample_rate)
    audio = pm.synthesize(fs=sample_rate)[:num_samples]
    if len(audio) < num_samples:
        audio = np.pad(audio, (0, num_samples - len(audio)))
    return torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def generate_loop(model, description, melody_wav, hum_seconds, duration_seconds):
    model.set_generation_params(duration=duration_seconds)
    return model.generate_with_chroma([description], melody_wav, MELODY_SAMPLE_RATE)


def generate_two_stage(model, description, melody_wav, hum_seconds, duration_seconds):
    model.set_generation_params(duration=hum_seconds)
    first = model.generate_with_chroma([description], melody_wav, MELODY_SAMPLE_RATE)
    if hum_seconds >= duration_seconds:
        return first
    # generate_continuation counts the prompt toward the duration
    model.set_generation_params(duration=duration_seconds)
    return model.generate_continuation(first, OUTPUT_SAMPLE_RATE, [description])


def generate_extended(model, description, melody_wav, hum_seconds, duration_seconds):
    # melody_wav is the full-length melody extended by extend_melody.py, so nothing is repeated
    model.set_generation_params(duration=duration_seconds)
    return model.generate_with_chroma([description], melody_wav, MELODY_SAMPLE_RATE)


GENERATORS = {'loop': generate_loop, 'two_stage': generate_two_stage, 'extended': generate_extended}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--samples', nargs='+', required=True, help='melody_vectors JSON files')
    parser.add_argument('--hum', nargs='+', type=float, default=[5, 10, 20], help='humming lengths H (seconds)')
    parser.add_argument('--duration', type=float, default=30)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--genre', default='pop')
    parser.add_argument('--mood', default='happy')
    parser.add_argument('--prompt', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--out', default=str(Path(__file__).parent / 'outputs'))
    parser.add_argument('--extended-dir', default=str(Path(__file__).parent / 'outputs' / 'extended'),
                        help='extend_melody.py output (for --methods extended)')
    args = parser.parse_args()

    if not 0 < args.duration <= MAX_DURATION_SECONDS:
        raise SystemExit(f"--duration must be in (0, {MAX_DURATION_SECONDS}]")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    description = build_description(args.genre, args.mood, args.prompt)
    print(f"Text condition: {description} / device: {args.device}", flush=True)

    model = MusicGen.get_pretrained('facebook/musicgen-melody', device=args.device)
    results_path = out_dir / 'results.jsonl'

    for sample_path in args.samples:
        sample = Path(sample_path).stem
        melody_vectors = json.loads(Path(sample_path).read_text())
        for hum_seconds in args.hum:
            hum_seconds = min(hum_seconds, args.duration)
            cropped = crop_vectors(melody_vectors, hum_seconds)
            melody_wav = synthesize_melody(cropped, hum_seconds)
            name = f"{sample}_h{hum_seconds:g}"
            torchaudio.save(str(out_dir / f"{name}_melody.wav"), melody_wav[0], MELODY_SAMPLE_RATE)

            for method in args.methods:
                method_melody = melody_wav
                if method == 'extended':
                    extended = json.loads((Path(args.extended_dir) / extended_name(sample, hum_seconds)).read_text())
                    method_melody = synthesize_melody(extended, args.duration)
                    torchaudio.save(str(out_dir / f"{name}_extended_melody.wav"), method_melody[0],
                                    MELODY_SAMPLE_RATE)

                torch.manual_seed(args.seed)
                start = time.time()
                output = GENERATORS[method](model, description, method_melody.to(args.device),
                                            hum_seconds, args.duration)
                elapsed = time.time() - start
                output_wav = output[0].cpu()
                output_path = out_dir / f"{name}_{method}.wav"
                torchaudio.save(str(output_path), output_wav, OUTPUT_SAMPLE_RATE)

                result = {
                    'sample': sample, 'hum_seconds': hum_seconds, 'method': method,
                    'notes': len(cropped), 'description': description, 'seed': args.seed,
                    'device': args.device, 'elapsed_seconds': round(elapsed, 1),
                    'output_seconds': output_wav.shape[-1] / OUTPUT_SAMPLE_RATE,
                    'peak': round(output_wav.abs().max().item(), 3), 'file': output_path.name,
                }
                print(json.dumps(result, ensure_ascii=False), flush=True)
                with results_path.open('a') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
