"""Copy comparison outputs into results/ for committing: wav -> mp3 (128 kbps) plus metadata.

outputs/ stays git-ignored (full wav files); results/ is small enough to commit.
Requires ffmpeg with libmp3lame.
"""
import json
import shutil
import subprocess
from pathlib import Path

BASE = Path(__file__).parent
OUTPUTS = BASE / 'outputs'
RESULTS = BASE / 'results'


def json_lines(path):
    if not path.exists():
        return []
    return [line for line in path.read_text().splitlines() if line.startswith('{')]


def main():
    audio_dir = RESULTS / 'audio'
    extended_dir = RESULTS / 'extended'
    audio_dir.mkdir(parents=True, exist_ok=True)
    extended_dir.mkdir(parents=True, exist_ok=True)

    for wav in sorted(OUTPUTS.glob('*.wav')):
        mp3 = audio_dir / f"{wav.stem}.mp3"
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(wav), '-codec:a', 'libmp3lame',
                        '-b:a', '128k', str(mp3)], check=True)

    for path in sorted((OUTPUTS / 'extended').glob('*')):
        if path.suffix in ('.json', '.mid'):
            shutil.copy2(path, extended_dir / path.name)

    # Keep the latest result per output file (re-runs append to results.jsonl)
    generation = {}
    for line in json_lines(OUTPUTS / 'results.jsonl'):
        result = json.loads(line)
        generation[result['file']] = result
    (RESULTS / 'generation.jsonl').write_text(
        ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in generation.values()))
    (RESULTS / 'extension.jsonl').write_text(
        ''.join(line + '\n' for line in json_lines(OUTPUTS / 'extend.log')))

    size = sum(p.stat().st_size for p in RESULTS.rglob('*') if p.is_file())
    print(f"Exported {len(list(audio_dir.glob('*.mp3')))} mp3 files, {len(generation)} generation results "
          f"({size / 1e6:.1f} MB) to {RESULTS}")


if __name__ == '__main__':
    main()
