"""Does MusicGen-melody follow mood tempo/mode better from text, or from an edited melody?

Conditions per (humming sample, mood):
  A_text       original melody, text "kpop, {mood}" (current server)
  B_text_meta  original melody, text + "bpm: X. key: T mode" (MusicGen's training metadata format)
  C_melody     melody edited by apply_mood (tempo toward the mood's BPM, parallel mode switch), text as A

Each output's tempo and key are measured with librosa. Resumable: finished rows in results.jsonl are skipped.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path[:0] = [str(HERE), str(ROOT)]

import xformers_stub  # noqa: E402,F401  (must come before audiocraft; no-op when xformers is installed)
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from melody_synth import synthesize_melody  # noqa: E402
from mood_mapping import apply_mood, audio_tempo_key, mood_targets, melody_key, melody_tempo  # noqa: E402

MELODY_SR = 16000
OUTPUT_SR = 32000
CONDITIONS = ('A_text', 'B_text_meta', 'C_melody')


def melody_audio(vectors, seconds):
    audio = synthesize_melody(vectors, MELODY_SR)[:round(seconds * MELODY_SR)]
    return np.pad(audio, (0, max(round(seconds * MELODY_SR) - len(audio), 0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', nargs='+', default=['humix_test_2', 'humix_test_4', 'humix_test_5'])
    parser.add_argument('--moods', nargs='+', default=['exciting', 'peaceful', 'sad', 'epic'])
    parser.add_argument('--genre', default='kpop')
    parser.add_argument('--duration', type=float, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=str(HERE / 'outputs'))
    # MPS on a 16GB Mac swaps at 10s; CPU takes about 3 min per 10s song and is reproducible
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    out = Path(args.out)
    (out / 'audio').mkdir(parents=True, exist_ok=True)
    results_path = out / 'results.jsonl'
    done = set()
    if results_path.exists():
        done = {(r['sample'], r['mood'], r['condition']) for r in map(json.loads, results_path.open())}

    from audiocraft.models import MusicGen
    device = args.device
    model = MusicGen.get_pretrained('facebook/musicgen-melody', device=device)
    model.set_generation_params(duration=args.duration)

    for sample in args.samples:
        vectors = json.loads((HERE / 'samples' / f'{sample}.json').read_text())
        for mood in args.moods:
            edited, info = apply_mood(vectors, mood)
            base_text = f"{args.genre}, {mood}"
            texts = {'A_text': base_text,
                     'B_text_meta': f"{base_text}. bpm: {info['bpm']}. key: {info['target_key']}",
                     'C_melody': base_text}
            melodies = {'A_text': vectors, 'B_text_meta': vectors, 'C_melody': edited}
            for condition in CONDITIONS:
                if (sample, mood, condition) in done:
                    continue
                name = f"{sample}_{mood}_{condition}"
                melody = melody_audio(melodies[condition], args.duration)
                sf.write(out / 'audio' / f"{name}_melody.wav", melody, MELODY_SR)
                torch.manual_seed(args.seed)
                start = time.time()
                wav = model.generate_with_chroma([texts[condition]],
                                                 torch.from_numpy(melody)[None, None].to(device), MELODY_SR)
                elapsed = time.time() - start
                audio = wav[0, 0].cpu().numpy()
                del wav
                gc.collect()
                if device == 'mps':
                    torch.mps.empty_cache()
                sf.write(out / 'audio' / f"{name}.wav", audio, OUTPUT_SR)
                cropped = [v for v in melodies[condition] if v['onset_seconds'] < args.duration]
                row = {'sample': sample, 'mood': mood, 'condition': condition, 'text': texts[condition],
                       'targets': info, 'input_bpm': round(melody_tempo(cropped), 1),
                       'input_key': melody_key(cropped)['name'],
                       'output': audio_tempo_key(audio, OUTPUT_SR),
                       'elapsed_seconds': round(elapsed, 1), 'file': f"audio/{name}.wav"}
                with results_path.open('a') as f:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
                print(f"[{name}] {elapsed:.0f}s target {info['bpm']}bpm {info['target_key']} -> "
                      f"out {row['output']['bpm']}bpm {row['output']['key']['name']}", flush=True)


if __name__ == '__main__':
    main()
