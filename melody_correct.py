"""Pull an AMT-extended melody back toward the humming it continues (#25, from the #14 PoC).

Only notes that start after the humming ends are changed; the humming part is kept as is.
- range:  notes outside the humming range (plus a margin) are moved by octaves into it
- legato: short gaps between extension notes are filled by holding the previous note
          (AMT writes detached piano notes, which become choppy as a sine melody)
No key correction: the humming itself is too far from any scale to estimate the key reliably (#14).
"""

RANGE_MARGIN = 2
LEGATO_MAX_GAP_SECONDS = 0.3


def fold_into_range(pitch, low, high):
    while pitch < low:
        pitch += 12
    while pitch > high:
        pitch -= 12
    # A range narrower than an octave can leave no fit; take the nearer edge's octave
    return pitch if pitch >= low else min(pitch + 12, high)


def fill_gaps(vectors, humming_end_seconds):
    pitched = sorted((v for v in vectors if v['pitch']), key=lambda v: v['onset_seconds'])
    for current, following in zip(pitched, pitched[1:]):
        if current['onset_seconds'] < humming_end_seconds:
            continue
        gap = following['onset_seconds'] - (current['onset_seconds'] + current['duration_seconds'])
        if 0 < gap < LEGATO_MAX_GAP_SECONDS:
            current['duration_seconds'] = round(following['onset_seconds'] - current['onset_seconds'], 4)
    return vectors


def correct_extension(vectors, humming_end_seconds):
    """Range and legato correction of the notes after humming_end_seconds. Returns new vectors."""
    humming = [v for v in vectors if v['pitch'] and v['onset_seconds'] < humming_end_seconds]
    if not humming:
        return [dict(v) for v in vectors]
    low = min(v['pitch'] for v in humming) - RANGE_MARGIN
    high = max(v['pitch'] for v in humming) + RANGE_MARGIN
    corrected = []
    for vector in vectors:
        vector = dict(vector)
        if vector['pitch'] and vector['onset_seconds'] >= humming_end_seconds:
            vector['pitch'] = fold_into_range(vector['pitch'], low, high)
        corrected.append(vector)
    return fill_gaps(corrected, humming_end_seconds)
