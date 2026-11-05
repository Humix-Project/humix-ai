"""Dependency-free melody vector helpers shared by the MusicGen (venv) and AMT (venv-amt) scripts."""


def crop_vectors(melody_vectors, hum_seconds):
    """Simulate an H-second humming: keep notes starting before H and clip them at H."""
    cropped = []
    for vector in melody_vectors:
        onset = vector.get('onset_seconds', vector.get('start_time_seconds', 0.0))
        if onset >= hum_seconds:
            continue
        duration = min(vector.get('duration_seconds', 0.5), hum_seconds - onset)
        cropped.append({'pitch': vector['pitch'], 'onset_seconds': onset, 'duration_seconds': duration})
    return cropped


def melody_end(melody_vectors):
    """Humming length: end of the last note, rests (pitch 0) included."""
    return max(v.get('onset_seconds', v.get('start_time_seconds', 0.0)) + v.get('duration_seconds', 0.5)
               for v in melody_vectors)


def resolve_hum_lengths(hum_args, melody_vectors, duration_seconds):
    """--hum values are seconds or 'full' (the recording's own length), capped at the song duration."""
    lengths = []
    for value in hum_args:
        seconds = melody_end(melody_vectors) if value == 'full' else float(value)
        lengths.append(round(min(seconds, duration_seconds), 2))
    return lengths


def extended_name(sample, hum_seconds):
    return f"{sample}_h{hum_seconds:g}_extended.json"
