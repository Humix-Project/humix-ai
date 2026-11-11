import numpy as np

# Notes closer than this are one phrase: synthesized as one continuous tone without a break
PHRASE_GAP_SECONDS = 0.01
# Short ramp at phrase edges to avoid clicks
EDGE_FADE_SECONDS = 0.005
AMPLITUDE = 0.5


def _note_events(melody_vectors):
    """Return (pitched notes as (start, end, pitch), melody end) in seconds. Rests count toward the end."""
    notes = []
    melody_end = 0.0
    for vector in melody_vectors:
        onset = vector.get('onset_seconds')
        if onset is None:
            onset = vector.get('start_time_seconds')
        if onset is None:
            onset = 0.0
        duration = vector.get('duration_seconds', 0.5)
        end = onset + duration
        melody_end = max(melody_end, end)
        # pitch 0 is a rest
        if vector.get('pitch') == 0 or duration <= 0:
            continue
        notes.append((onset, end, vector['pitch']))
    notes.sort(key=lambda note: note[0])
    return notes, melody_end


def synthesize_melody(melody_vectors, sample_rate=16000):
    """Synthesize melody vectors as a sine melody for MusicGen's chroma condition.

    Consecutive notes are joined into phrases and synthesized with continuous phase and constant
    amplitude, so a held (or slightly wavering) note does not break at every note boundary.
    Only rests and real gaps are silent. Output length is the end of the last vector (rests included).
    """
    notes, melody_end = _note_events(melody_vectors)
    if not notes:
        raise ValueError("melody_vectors has no pitched notes.")

    num_samples = round(melody_end * sample_rate)
    audio = np.zeros(num_samples, dtype=np.float32)
    gap = round(PHRASE_GAP_SECONDS * sample_rate)

    # Per-sample frequency for each phrase; a later note cuts an overlapping earlier one
    phrases = []
    for start_s, end_s, pitch in notes:
        start = min(round(start_s * sample_rate), num_samples)
        end = min(round(end_s * sample_rate), num_samples)
        if end <= start:
            continue
        frequency = 440.0 * 2 ** ((pitch - 69) / 12)
        if phrases and start <= phrases[-1]['end'] + gap:
            phrase = phrases[-1]
            # Hold the previous pitch through a tiny gap, or cut it at an overlap
            hold = start - phrase['end']
            if hold > 0:
                phrase['freqs'].append(np.full(hold, phrase['freqs'][-1][-1]))
            elif hold < 0:
                phrase['freqs'][-1] = phrase['freqs'][-1][:max(len(phrase['freqs'][-1]) + hold, 0)]
            phrase['freqs'].append(np.full(end - start, frequency))
            phrase['end'] = end
        else:
            phrases.append({'start': start, 'end': end, 'freqs': [np.full(end - start, frequency)]})

    fade = round(EDGE_FADE_SECONDS * sample_rate)
    for phrase in phrases:
        freqs = np.concatenate(phrase['freqs'])
        # Accumulate phase so frequency changes inside a phrase do not click
        phase = 2 * np.pi * np.cumsum(freqs) / sample_rate
        tone = AMPLITUDE * np.sin(phase)
        edge = min(fade, len(tone) // 2)
        if edge > 0:
            ramp = np.linspace(0.0, 1.0, edge)
            tone[:edge] *= ramp
            tone[-edge:] *= ramp[::-1]
        audio[phrase['start']:phrase['start'] + len(tone)] = tone
    return audio
