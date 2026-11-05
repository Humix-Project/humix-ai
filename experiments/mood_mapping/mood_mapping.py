"""Mood (V-A) -> target tempo/mode, and the melody edits the backend would apply to note lists.

Pure functions on melody vectors ({pitch, onset_seconds, duration_seconds}, pitch 0 = rest),
so the logic can be ported to the backend as is.
"""
import numpy as np

# Valence-Arousal coordinates from the planning doc (V-A 공간 그래프)
MOODS = {
    'intense': (0.3, 0.9), 'dynamic': (0.4, 0.9), 'anxious': (0.2, 0.8), 'epic': (0.4, 0.8),
    'tense': (0.2, 0.7), 'grand': (0.3, 0.7), 'dark': (0.1, 0.5), 'sentimental': (0.2, 0.4),
    'sad': (0.1, 0.3), 'melancholic': (0.1, 0.2), 'nostalgic': (0.3, 0.3), 'dreamy': (0.4, 0.3),
    'exciting': (0.9, 0.9), 'energetic': (0.8, 0.8), 'cheerful': (0.9, 0.7), 'bright': (0.8, 0.5),
    'warm': (0.7, 0.3), 'peaceful': (0.7, 0.2), 'cozy': (0.8, 0.2), 'calm': (0.6, 0.1),
}

# Provisional linear mapping (to be calibrated from the literature / data)
BPM_AT_AROUSAL_0 = 60
BPM_AT_AROUSAL_1 = 150
MINOR_BELOW_VALENCE = 0.5
# Keep the humming recognizable
MAX_TEMPO_RATIO = 1.33
MIN_TEMPO_RATIO = 0.75

PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
# Krumhansl & Kessler (1982) key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
# Scale degrees (semitones above the tonic) that differ between parallel major and minor
MAJOR_TO_MINOR = {4: -1, 9: -1, 11: -1}
MINOR_TO_MAJOR = {3: +1, 8: +1, 10: +1}


def mood_targets(mood):
    valence, arousal = MOODS[mood]
    bpm = BPM_AT_AROUSAL_0 + (BPM_AT_AROUSAL_1 - BPM_AT_AROUSAL_0) * arousal
    mode = 'minor' if valence < MINOR_BELOW_VALENCE else 'major'
    return {'valence': valence, 'arousal': arousal, 'bpm': round(bpm), 'mode': mode}


def estimate_key(pitch_class_weights):
    """Krumhansl-Schmuckler: best (tonic, mode) by profile correlation, plus margin over the runner-up."""
    scores = []
    for tonic in range(12):
        rotated = np.roll(pitch_class_weights, -tonic)
        scores.append((np.corrcoef(rotated, MAJOR_PROFILE)[0, 1], tonic, 'major'))
        scores.append((np.corrcoef(rotated, MINOR_PROFILE)[0, 1], tonic, 'minor'))
    scores.sort(reverse=True)
    best, runner_up = scores[0], scores[1]
    return {'tonic': best[1], 'mode': best[2], 'name': f"{PITCH_CLASSES[best[1]]} {best[2]}",
            'corr': round(float(best[0]), 3), 'margin': round(float(best[0] - runner_up[0]), 3)}


def melody_key(vectors):
    weights = np.zeros(12)
    for v in vectors:
        if v['pitch']:
            weights[v['pitch'] % 12] += v['duration_seconds']
    return estimate_key(weights)


def melody_tempo(vectors, hop_seconds=0.01):
    """Tempo of the note onsets (rests excluded), via librosa's tempo estimator on an impulse train."""
    import librosa
    end = max(v['onset_seconds'] + v['duration_seconds'] for v in vectors)
    envelope = np.zeros(int(end / hop_seconds) + 1)
    for v in vectors:
        if v['pitch']:
            envelope[int(v['onset_seconds'] / hop_seconds)] += 1
    sr = 1 / hop_seconds
    return float(librosa.feature.tempo(onset_envelope=envelope, sr=sr, hop_length=1)[0])


def change_tempo(vectors, ratio):
    """ratio > 1 is faster."""
    return [{**v, 'onset_seconds': round(v['onset_seconds'] / ratio, 4),
             'duration_seconds': round(v['duration_seconds'] / ratio, 4)} for v in vectors]


def change_mode(vectors, tonic, to_mode):
    shifts = MAJOR_TO_MINOR if to_mode == 'minor' else MINOR_TO_MAJOR
    out = []
    for v in vectors:
        degree = (v['pitch'] - tonic) % 12 if v['pitch'] else None
        out.append({**v, 'pitch': v['pitch'] + shifts.get(degree, 0) if v['pitch'] else 0})
    return out


def apply_mood(vectors, mood):
    """What the backend would do: move tempo toward the mood's BPM (clamped), switch to its mode."""
    targets = mood_targets(mood)
    tempo = melody_tempo(vectors)
    ratio = float(np.clip(targets['bpm'] / tempo, MIN_TEMPO_RATIO, MAX_TEMPO_RATIO))
    key = melody_key(vectors)
    edited = change_tempo(vectors, ratio)
    if key['mode'] != targets['mode']:
        edited = change_mode(edited, key['tonic'], targets['mode'])
    info = {**targets, 'hum_bpm': round(tempo, 1), 'tempo_ratio': round(ratio, 3), 'hum_key': key,
            'target_key': f"{PITCH_CLASSES[key['tonic']]} {targets['mode']}"}
    return edited, info


def audio_tempo_key(audio, sr):
    """Measured tempo (beat tracker) and key (chroma + Krumhansl-Schmuckler) of generated audio."""
    import librosa
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    onsets = librosa.onset.onset_detect(y=audio, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr).mean(axis=1)
    return {'bpm': round(float(np.atleast_1d(tempo)[0]), 1),
            'onsets_per_sec': round(len(onsets) / (len(audio) / sr), 2),
            'key': estimate_key(chroma)}
