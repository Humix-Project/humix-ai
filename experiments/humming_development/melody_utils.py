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


def extended_name(sample, hum_seconds):
    return f"{sample}_h{hum_seconds:g}_extended.json"
