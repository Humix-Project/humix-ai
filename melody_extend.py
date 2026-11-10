"""Extend a humming shorter than the song with the Anticipatory Music Transformer (#25).

The humming is the prompt and AMT writes the rest up to the song length as a piano part, reduced to
one line (skyline). The extension is then corrected toward the humming (melody_correct.py) and padded
with a rest to the song length, so MusicGen gets a full-length melody instead of a looped humming.
If AMT fails or takes too long, the humming is used as is (MusicGen loops it, the old behavior).
"""
import time

import torch

AMT_MODEL = 'stanford-crfm/music-medium-800k'
TOP_P = 0.95
# A humming this close to the song length is used as is
MIN_EXTENSION_SECONDS = 1.0
# Sampling recomputes the whole context per token (no KV cache), so it slows down as it goes
EXTENSION_TIMEOUT_SECONDS = 180
MELODY_INSTRUMENT = 0  # Acoustic Grand Piano

amt_model = None


def humming_end(melody_vectors):
    """End of the humming: the last entry's end, rests included (same as the synthesized length)."""
    return max(_onset(v) + v['duration_seconds'] for v in melody_vectors)


def _onset(vector):
    onset = vector.get('onset_seconds')
    return vector.get('start_time_seconds', 0.0) if onset is None else onset


def load_amt_model():
    global amt_model
    if amt_model is None:
        from transformers import AutoModelForCausalLM
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading AMT model {AMT_MODEL} on {device}...")
        amt_model = AutoModelForCausalLM.from_pretrained(AMT_MODEL).to(device).eval()
    return amt_model


def vectors_to_events(melody_vectors):
    from anticipation.config import MAX_DUR, MAX_PITCH, TIME_RESOLUTION
    from anticipation.vocab import DUR_OFFSET, NOTE_OFFSET, TIME_OFFSET

    events = []
    for vector in sorted(melody_vectors, key=_onset):
        if vector['pitch'] == 0:
            continue
        onset = round(_onset(vector) * TIME_RESOLUTION)
        duration = min(max(round(vector['duration_seconds'] * TIME_RESOLUTION), 1), MAX_DUR - 1)
        events.extend([TIME_OFFSET + onset, DUR_OFFSET + duration,
                       NOTE_OFFSET + MELODY_INSTRUMENT * MAX_PITCH + vector['pitch']])
    return events


def events_to_vectors(events, start_seconds, end_seconds):
    """Decode generated events after start_seconds into a monophonic melody (highest note wins)."""
    from anticipation.config import MAX_PITCH, TIME_RESOLUTION
    from anticipation.vocab import DUR_OFFSET, NOTE_OFFSET, TIME_OFFSET

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


def _amt_generate(model, inputs, start_seconds, end_seconds, timeout_seconds):
    """AMT sampling restricted to the melody instrument, aborted once the timeout passes."""
    from anticipation import sample as amt_sample
    from anticipation.config import MAX_NOTE, MAX_PITCH
    from anticipation.vocab import NOTE_OFFSET

    original_instr_logits = amt_sample.instr_logits
    deadline = time.monotonic() + timeout_seconds

    def melody_only_instr_logits(logits, full_history):
        # Called once per sampled token, so it is also where the timeout is checked
        if time.monotonic() > deadline:
            raise TimeoutError(f"AMT extension took longer than {timeout_seconds} seconds")
        logits = original_instr_logits(logits, full_history)
        logits[NOTE_OFFSET:NOTE_OFFSET + MELODY_INSTRUMENT * MAX_PITCH] = -float('inf')
        logits[NOTE_OFFSET + (MELODY_INSTRUMENT + 1) * MAX_PITCH:NOTE_OFFSET + MAX_NOTE] = -float('inf')
        return logits

    amt_sample.instr_logits = melody_only_instr_logits
    try:
        with torch.no_grad():
            return amt_sample.generate(model, start_time=start_seconds, end_time=end_seconds,
                                       inputs=inputs, top_p=TOP_P)
    finally:
        amt_sample.instr_logits = original_instr_logits


def pad_to(melody_vectors, duration_seconds):
    """End the melody exactly at duration_seconds with a trailing rest, so it is not looped."""
    end = humming_end(melody_vectors)
    if end < duration_seconds:
        melody_vectors = melody_vectors + [{'pitch': 0, 'onset_seconds': round(end, 4),
                                            'duration_seconds': round(duration_seconds - end, 4)}]
    return melody_vectors


def extend_melody(melody_vectors, duration_seconds, timeout_seconds=EXTENSION_TIMEOUT_SECONDS):
    """Return (melody for generation, humming_end_seconds).

    The melody is the humming plus the corrected AMT extension, padded to duration_seconds; or the
    humming as is when it already (almost) fills the song or the extension fails.
    """
    from melody_correct import correct_extension

    humming = [dict(v, onset_seconds=_onset(v)) for v in melody_vectors]
    end = humming_end(humming)
    if duration_seconds - end < MIN_EXTENSION_SECONDS:
        return humming, end
    try:
        start = time.monotonic()
        model = load_amt_model()
        events = _amt_generate(model, vectors_to_events(humming), end, duration_seconds, timeout_seconds)
        extension = events_to_vectors(events, end, duration_seconds)
        print(f"AMT extension: {end:.2f}s -> {duration_seconds}s, {len(extension)} notes "
              f"in {time.monotonic() - start:.1f}s")
    except Exception as e:
        print(f"AMT extension failed, looping the humming instead: {e}")
        return humming, end
    extended = correct_extension(humming + extension, end)
    return pad_to(extended, duration_seconds), end
