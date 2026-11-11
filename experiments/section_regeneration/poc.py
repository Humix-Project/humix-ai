"""Issue #13 PoC: regenerate a section of a song with section_regen.py and save the merged song.

Each source is a generated song (32 kHz wav) and the 30-second melody it was generated with.
For every section and take, the merged song and the raw generated part are written to --out.
MusicGen on local MPS is not reproducible even with a fixed seed, so use --takes > 1 to judge.

Example:
  python poc.py --source song.wav melody.json 12.12 --sections 10-20 3-13 20-30 --takes 2
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import xformers_stub  # noqa: E402,F401  (local Mac only, must come before audiocraft)
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from audiocraft.models import MusicGen  # noqa: E402

from melody_synth import synthesize_melody  # noqa: E402
from section_regen import (MELODY_SAMPLE_RATE, OUTPUT_SAMPLE_RATE, fit_length, generate_section,  # noqa: E402
                           has_pitched_notes, merge_section, plan_section, select_melody)


def melody_wav(vectors, seconds):
    audio = torch.from_numpy(synthesize_melody(vectors, MELODY_SAMPLE_RATE)) if has_pitched_notes(vectors) \
        else torch.zeros(0)
    return fit_length(audio[None], round(seconds * MELODY_SAMPLE_RATE))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', nargs=3, action='append', required=True,
                        metavar=('WAV', 'MELODY_JSON', 'HUMMING_END'), help='repeatable')
    parser.add_argument('--sections', nargs='+', default=['10-20', '3-13', '2-29', '20-30'])
    parser.add_argument('--takes', type=int, default=2)
    parser.add_argument('--description', default='pop, happy')
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--out', default=str(Path(__file__).parent / 'outputs'))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = MusicGen.get_pretrained('facebook/musicgen-melody', device=args.device)

    for wav_path, melody_path, humming_end in args.source:
        name = Path(wav_path).stem
        audio, sample_rate = sf.read(wav_path, dtype='float32')
        assert sample_rate == OUTPUT_SAMPLE_RATE, sample_rate
        source = torch.from_numpy(audio)[None]
        source_seconds = source.shape[-1] / OUTPUT_SAMPLE_RATE
        melody = json.loads(Path(melody_path).read_text())

        for section in args.sections:
            start, end = (float(x) for x in section.split('-'))
            context, tail = plan_section(source_seconds, start, end)
            window = select_melody(melody, start - context, end + tail, humming_end_seconds=float(humming_end))
            window_wav = melody_wav(window, context + (end - start) + tail)
            for take in range(1, args.takes + 1):
                begin = time.time()
                generated = generate_section(model, args.description, source, window_wav, start, end, context, tail)
                merged = merge_section(source, generated.cpu(), start, end, tail)
                tag = f"{name}_s{start:g}-{end:g}_t{take}"
                sf.write(out_dir / f"{tag}.wav", merged[0].numpy(), OUTPUT_SAMPLE_RATE, subtype='PCM_16')
                result = {'source': name, 'start': start, 'end': end, 'context': context, 'tail': tail,
                          'take': take, 'elapsed_seconds': round(time.time() - begin, 1),
                          'generated_seconds': round(generated.shape[-1] / OUTPUT_SAMPLE_RATE, 3),
                          'generated_peak': round(generated.abs().max().item(), 3),
                          'merged_seconds': merged.shape[-1] / OUTPUT_SAMPLE_RATE, 'file': f"{tag}.wav"}
                print(json.dumps(result), flush=True)
                with (out_dir / 'results.jsonl').open('a') as f:
                    f.write(json.dumps(result) + '\n')


if __name__ == '__main__':
    main()
