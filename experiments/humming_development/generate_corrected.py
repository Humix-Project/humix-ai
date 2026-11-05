"""Issue #14 candidate 2: generate songs from corrected extended melodies.

Each extend_melody.py output is corrected by correct_melody.py (extension part only), synthesized with the
#23 continuous synthesis and generated once with generate_with_chroma for the full duration (same as
compare.py's extended method).

Variants:
  none              no correction
  range             humming range +-2 semitones (octave folding)
  range_legato      range + fill gaps under 0.3 s
  range_key         range + snap to the major scale estimated from the humming
  range_key_legato  all three

Example:
  python generate_corrected.py --extended results/extended/humix_test_*_extended.json \\
      --variants range range_legato --export
"""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import compare  # noqa: F401  (loads xformers_stub before audiocraft)
import torch
import torchaudio
from audiocraft.models import MusicGen

from correct_melody import correct

BASE = Path(__file__).parent
VARIANTS = {
    'none': None,
    'range': dict(use_key=False, use_legato=False),
    'range_legato': dict(use_key=False, use_legato=True),
    'range_key': dict(use_key=True, use_legato=False),
    'range_key_legato': dict(use_key=True, use_legato=True),
}


def parse_name(path):
    # {sample}_h{H}_extended.json from extend_melody.py
    match = re.fullmatch(r'(.+)_h([\d.]+)_extended', Path(path).stem)
    if not match:
        raise SystemExit(f"unexpected extended melody name: {path}")
    return match.group(1), float(match.group(2))


def mp3(wav, out):
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(wav), '-codec:a', 'libmp3lame',
                    '-b:a', '128k', str(out)], check=True)


def update_stats(stats_path, result):
    # corrected.html reads peaks from this file, one row per sample and method
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else []
    stats = [d for d in stats if (d['sample'], d['method']) != (result['sample'], result['method'])]
    stats.append({key: result[key] for key in ('sample', 'method', 'elapsed_seconds', 'peak')})
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--extended', nargs='+', required=True, help='extend_melody.py output json files')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=['range', 'range_legato'])
    parser.add_argument('--duration', type=float, default=30)
    parser.add_argument('--genre', default='pop')
    parser.add_argument('--mood', default='happy')
    parser.add_argument('--prompt', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--out', default=str(BASE / 'outputs' / 'corrected'))
    parser.add_argument('--export', action='store_true',
                        help='also write mp3 and stats to results/audio/corrected')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    export_dir = BASE / 'results' / 'audio' / 'corrected'
    if args.export:
        export_dir.mkdir(parents=True, exist_ok=True)
    description = compare.build_description(args.genre, args.mood, args.prompt)
    model = MusicGen.get_pretrained('facebook/musicgen-melody', device=args.device)

    for path in args.extended:
        sample, hum_seconds = parse_name(path)
        extended = json.loads(Path(path).read_text())
        for variant in args.variants:
            options = VARIANTS[variant]
            vectors = extended if options is None else correct(extended, hum_seconds, **options)
            name = f"{sample}_{variant}"
            (out_dir / f"{name}.json").write_text(json.dumps(vectors))
            melody_wav = compare.synthesize_melody_continuous(vectors, args.duration)
            input_path = out_dir / f"{name}_input.wav"
            torchaudio.save(str(input_path), melody_wav[0], compare.MELODY_SAMPLE_RATE)

            torch.manual_seed(args.seed)
            start = time.time()
            output = compare.generate_extended(model, description, melody_wav.to(args.device),
                                               hum_seconds, args.duration)
            output_wav = output[0].cpu()
            output_path = out_dir / f"{name}.wav"
            torchaudio.save(str(output_path), output_wav, compare.OUTPUT_SAMPLE_RATE)

            result = {
                'sample': sample, 'hum_seconds': hum_seconds, 'method': f'extended_{variant}',
                'description': description, 'seed': args.seed, 'device': args.device,
                'elapsed_seconds': round(time.time() - start, 1),
                'peak': round(output_wav.abs().max().item(), 3), 'file': output_path.name,
            }
            print(json.dumps(result, ensure_ascii=False), flush=True)
            with (out_dir / 'results.jsonl').open('a') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

            if args.export:
                # corrected.html names the uncorrected input "original"
                mp3(input_path, export_dir / f"{sample}_{'original' if variant == 'none' else variant}_input.mp3")
                mp3(output_path, export_dir / f"{sample}_extended_{variant}.mp3")
                update_stats(export_dir / 'stats.json', result)


if __name__ == '__main__':
    main()
