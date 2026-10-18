"""Issue #14 PoC candidate 2: extend an H-second humming melody to the full duration with the
Anticipatory Music Transformer (symbolic MIDI model), then feed it to compare.py --methods extended.

Runs in its own venv (anticipation pins transformers==4.29.2, which conflicts with audiocraft):
  python3.12 -m venv venv-amt
  venv-amt/bin/pip install torch transformers numpy tqdm mido pretty_midi
  venv-amt/bin/pip install --no-deps git+https://github.com/jthickstun/anticipation.git

Example:
  ../../venv-amt/bin/python extend_melody.py --samples samples/vector1.json --hum 5 10 20 --duration 30
"""
import argparse
import json
import time
from pathlib import Path

import pretty_midi
import torch
from transformers import AutoModelForCausalLM

from anticipation import sample as amt_sample
from anticipation.config import MAX_DUR, MAX_NOTE, MAX_PITCH, TIME_RESOLUTION
from anticipation.vocab import DUR_OFFSET, NOTE_OFFSET, TIME_OFFSET

from melody_utils import crop_vectors, extended_name

MELODY_INSTRUMENT = 0  # Acoustic Grand Piano, same as the synthesized MusicGen melody
_original_instr_logits = amt_sample.instr_logits


def melody_only_instr_logits(logits, full_history):
    # Only allow notes on the melody instrument so the model continues the melody, not an arrangement
    logits = _original_instr_logits(logits, full_history)
    logits[NOTE_OFFSET:NOTE_OFFSET + MELODY_INSTRUMENT * MAX_PITCH] = -float('inf')
    logits[NOTE_OFFSET + (MELODY_INSTRUMENT + 1) * MAX_PITCH:NOTE_OFFSET + MAX_NOTE] = -float('inf')
    return logits


amt_sample.instr_logits = melody_only_instr_logits


def vectors_to_events(melody_vectors):
    events = []
    for vector in sorted(melody_vectors, key=lambda v: v['onset_seconds']):
        if vector['pitch'] == 0:
            continue
        onset = round(vector['onset_seconds'] * TIME_RESOLUTION)
        duration = min(max(round(vector['duration_seconds'] * TIME_RESOLUTION), 1), MAX_DUR - 1)
        events.extend([TIME_OFFSET + onset, DUR_OFFSET + duration,
                       NOTE_OFFSET + MELODY_INSTRUMENT * MAX_PITCH + vector['pitch']])
    return events


def events_to_vectors(events, start_seconds, end_seconds):
    """Decode generated events after start_seconds into a monophonic melody (highest note wins)."""
    notes = []
    for time_token, dur_token, note_token in zip(events[0::3], events[1::3], events[2::3]):
        note = note_token - NOTE_OFFSET
        if note // MAX_PITCH != MELODY_INSTRUMENT:
            continue
        onset = (time_token - TIME_OFFSET) / TIME_RESOLUTION
        if not start_seconds <= onset < end_seconds:
            continue
        duration = (dur_token - DUR_OFFSET) / TIME_RESOLUTION
        notes.append([onset, min(onset + duration, end_seconds), note % MAX_PITCH])

    # Skyline: among notes with the same onset keep the highest, and cut each note at the next onset
    notes.sort(key=lambda n: (n[0], -n[2]))
    melody = []
    for onset, end, pitch in notes:
        if melody and melody[-1][0] == onset:
            continue
        if melody:
            melody[-1][1] = min(melody[-1][1], onset)
        melody.append([onset, end, pitch])
    return [{'pitch': p, 'onset_seconds': round(s, 2), 'duration_seconds': round(e - s, 2)}
            for s, e, p in melody if e - s > 0]


def save_midi(melody_vectors, path):
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=MELODY_INSTRUMENT)
    for v in melody_vectors:
        piano.notes.append(pretty_midi.Note(velocity=100, pitch=v['pitch'], start=v['onset_seconds'],
                                            end=v['onset_seconds'] + v['duration_seconds']))
    pm.instruments.append(piano)
    pm.write(str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--samples', nargs='+', required=True, help='melody_vectors JSON files')
    parser.add_argument('--hum', nargs='+', type=float, default=[5, 10, 20])
    parser.add_argument('--duration', type=float, default=30)
    parser.add_argument('--model', default='stanford-crfm/music-medium-800k')
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=str(Path(__file__).parent / 'outputs' / 'extended'))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model).eval()
    print(f"Loaded {args.model} in {time.time() - start:.1f}s", flush=True)

    for sample_path in args.samples:
        sample = Path(sample_path).stem
        melody_vectors = json.loads(Path(sample_path).read_text())
        for hum_seconds in args.hum:
            hum_seconds = min(hum_seconds, args.duration)
            humming = crop_vectors(melody_vectors, hum_seconds)
            torch.manual_seed(args.seed)
            start = time.time()
            events = amt_sample.generate(model, start_time=hum_seconds, end_time=args.duration,
                                         inputs=vectors_to_events(humming), top_p=args.top_p)
            elapsed = time.time() - start
            extension = events_to_vectors(events, hum_seconds, args.duration)
            extended = humming + extension

            name = extended_name(sample, hum_seconds)
            (out_dir / name).write_text(json.dumps(extended, indent=2))
            save_midi(extended, out_dir / name.replace('.json', '.mid'))
            print(json.dumps({'sample': sample, 'hum_seconds': hum_seconds, 'model': args.model,
                              'humming_notes': len(humming), 'generated_notes': len(extension),
                              'elapsed_seconds': round(elapsed, 1), 'file': name}), flush=True)


if __name__ == '__main__':
    main()
