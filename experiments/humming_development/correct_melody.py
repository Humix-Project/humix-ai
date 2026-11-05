"""Pull an AMT-extended melody back toward the humming it continues.

Only notes that start after the humming ends are changed; the humming part is kept as is.
- range:  notes outside the humming range (plus a margin) are moved by octaves into it
- key:    out-of-scale notes are snapped one semitone to the major scale estimated from the humming
- legato: short gaps between extension notes are filled by holding the previous note
          (AMT writes detached piano notes, which become choppy as a sine melody)
"""
import argparse
import json
from collections import Counter

MAJOR = (0, 2, 4, 5, 7, 9, 11)
RANGE_MARGIN = 2
LEGATO_MAX_GAP_SECONDS = 0.3


def estimate_scale(notes):
    # Duration-weighted pitch-class histogram, best-fitting major scale (covers its relative minor too)
    weight = Counter()
    for note in notes:
        weight[note['pitch'] % 12] += note['duration_seconds']
    tonic = max(range(12), key=lambda t: sum(weight[(t + i) % 12] for i in MAJOR))
    return {(tonic + i) % 12 for i in MAJOR}


def fold_into_range(pitch, low, high):
    while pitch < low:
        pitch += 12
    while pitch > high:
        pitch -= 12
    # A range narrower than an octave can leave no fit; take the nearer edge's octave
    return pitch if pitch >= low else min(pitch + 12, high)


def snap_to_scale(pitch, scale, previous):
    if pitch % 12 in scale:
        return pitch
    # Prefer the neighbour toward the previous note so stepwise motion stays stepwise
    order = (pitch - 1, pitch + 1) if previous is not None and previous < pitch else (pitch + 1, pitch - 1)
    return next(p for p in order if p % 12 in scale)


def fill_gaps(vectors, hum_seconds):
    pitched = sorted((v for v in vectors if v['pitch']), key=lambda v: v['onset_seconds'])
    for current, following in zip(pitched, pitched[1:]):
        if current['onset_seconds'] < hum_seconds:
            continue
        gap = following['onset_seconds'] - (current['onset_seconds'] + current['duration_seconds'])
        if 0 < gap < LEGATO_MAX_GAP_SECONDS:
            current['duration_seconds'] = round(following['onset_seconds'] - current['onset_seconds'], 4)
    return vectors


def correct(vectors, hum_seconds, use_key, use_legato=False):
    humming = [v for v in vectors if v['pitch'] and v['onset_seconds'] < hum_seconds]
    low = min(v['pitch'] for v in humming) - RANGE_MARGIN
    high = max(v['pitch'] for v in humming) + RANGE_MARGIN
    scale = estimate_scale(humming)
    corrected, previous = [], None
    for vector in vectors:
        vector = dict(vector)
        if vector['pitch'] and vector['onset_seconds'] >= hum_seconds:
            pitch = fold_into_range(vector['pitch'], low, high)
            if use_key:
                pitch = snap_to_scale(pitch, scale, previous)
                # Snapping can step one semitone past the edge
                pitch = fold_into_range(pitch, low, high) if not low <= pitch <= high else pitch
            vector['pitch'] = pitch
        if vector['pitch']:
            previous = vector['pitch']
        corrected.append(vector)
    return fill_gaps(corrected, hum_seconds) if use_legato else corrected


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('extended', help='extended melody json from extend_melody.py')
    parser.add_argument('--hum', type=float, required=True, help='humming length in seconds')
    parser.add_argument('--key', action='store_true', help='also snap to the humming scale')
    parser.add_argument('--legato', action='store_true', help='also fill short gaps between extension notes')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    with open(args.extended) as f:
        vectors = json.load(f)
    with open(args.out, 'w') as f:
        json.dump(correct(vectors, args.hum, args.key, args.legato), f)


if __name__ == '__main__':
    main()
