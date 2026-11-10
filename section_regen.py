"""Section regeneration (#13): regenerate [start, end) of a song and merge it back into the original.

MusicGen cannot inpaint, so the section is generated as a continuation of the audio just before it
(context), with a short extra tail that is crossfaded into the original after the section.
All times are seconds on the original song's timeline unless noted.
"""
import io
import math
from contextlib import contextmanager

import requests
import torch

from audio_output import OUTPUT_SAMPLE_RATE, soft_clip

MAX_GENERATION_SECONDS = 30.0
MIN_SECTION_SECONDS = 5.0
# The backend checks lengths within ±0.1 s, so a section end that far past the source is its end
SECTION_END_TOLERANCE_SECONDS = 0.1
MAX_TEXT_CONDITION_CHARS = 1000
SOURCE_DOWNLOAD_TIMEOUT_SECONDS = 60
MAX_CONTEXT_SECONDS = 5.0
MAX_TAIL_SECONDS = 1.0
START_CROSSFADE_SECONDS = 0.03
MELODY_SAMPLE_RATE = 16000


def plan_section(source_seconds, start, end, max_total=MAX_GENERATION_SECONDS,
                 max_context=MAX_CONTEXT_SECONDS, max_tail=MAX_TAIL_SECONDS):
    """Return (context, tail) seconds so that context + section + tail fits in max_total.

    The section always gets its full length; what is left goes to the tail first, then to the context.
    """
    if not 0 <= start < end <= source_seconds:
        raise ValueError(f"section must satisfy 0 <= start < end <= {source_seconds}, got {start}~{end}")
    budget = max_total - (end - start)
    if budget < 0:
        raise ValueError(f"section is longer than {max_total} seconds")
    tail = min(max_tail, source_seconds - end, budget)
    context = min(max_context, start, budget - tail)
    return context, tail


def _onset(vector):
    onset = vector.get('onset_seconds')
    return vector.get('start_time_seconds', 0.0) if onset is None else onset


def repeat_melody(melody_vectors, until_seconds):
    """Old songs (no extended melody): the humming was looped with period H = end of the last entry."""
    period = max(_onset(v) + v['duration_seconds'] for v in melody_vectors)
    repeated, offset = [], 0.0
    while offset < until_seconds:
        for v in melody_vectors:
            repeated.append({'pitch': v['pitch'], 'onset_seconds': round(offset + _onset(v), 4),
                             'duration_seconds': v['duration_seconds']})
        offset += period
    return repeated


def select_melody(melody_vectors, window_start, window_end, humming_end_seconds=None):
    """Melody for the generation window, shifted so that window_start becomes 0.

    New songs send the full-length melody the version was generated with (humming_end_seconds is set),
    so the window is taken as is. Old songs send only the humming, which was looped during generation.
    Notes crossing the window edges are cut at the edges.
    """
    if humming_end_seconds is None:
        melody_vectors = repeat_melody(melody_vectors, window_end)
    selected = []
    for v in melody_vectors:
        onset = _onset(v)
        start, end = max(onset, window_start), min(onset + v['duration_seconds'], window_end)
        if end <= start:
            continue
        selected.append({'pitch': v['pitch'], 'onset_seconds': round(start - window_start, 4),
                         'duration_seconds': round(end - start, 4)})
    return selected


def has_pitched_notes(melody_vectors):
    return any(v['pitch'] != 0 for v in melody_vectors)


def fit_length(wav, num_samples):
    """Cut or zero-pad the last axis to exactly num_samples (#21)."""
    length = wav.shape[-1]
    if length >= num_samples:
        return wav[..., :num_samples]
    return torch.nn.functional.pad(wav, (0, num_samples - length))


def merge_section(source, generated, start, end, tail, sample_rate=OUTPUT_SAMPLE_RATE,
                  start_crossfade=START_CROSSFADE_SECONDS):
    """Put the generated section (starting at `start`, covering section + tail) back into the source.

    - Only the generated audio is limited: the source was already limited when it was saved, and limiting
      it again would change the untouched parts of the song.
    - Start: a short linear crossfade from the source to the generated audio against clicks.
    - End: the tail is crossfaded from the generated audio back into the source.
    The result has exactly the source's length.
    """
    start_i, end_i = round(start * sample_rate), round(end * sample_rate)
    tail_i = min(round(tail * sample_rate), source.shape[-1] - end_i)
    generated = soft_clip(fit_length(generated, end_i - start_i + tail_i))

    merged = source.clone()
    merged[..., start_i:end_i] = generated[..., :end_i - start_i]

    fade_i = min(round(start_crossfade * sample_rate), end_i - start_i)
    if start_i > 0 and fade_i > 0:
        ramp = torch.linspace(0, 1, fade_i)
        merged[..., start_i:start_i + fade_i] = (source[..., start_i:start_i + fade_i] * (1 - ramp)
                                                 + generated[..., :fade_i] * ramp)
    if tail_i > 0:
        ramp = torch.linspace(1, 0, tail_i)
        merged[..., end_i:end_i + tail_i] = (generated[..., end_i - start_i:] * ramp
                                             + source[..., end_i:end_i + tail_i] * (1 - ramp))
    return merged


@contextmanager
def melody_masking(model, enabled):
    """Mask the melody embedding so a `None` melody means "no melody" as in training.

    get_pretrained() turns masking off, which makes a `None` melody a chroma of all C.
    Only flipped for the call and always restored, since the model is shared across requests.
    """
    conditioner = model.lm.condition_provider.conditioners['self_wav']
    previous = conditioner._use_masking
    conditioner._use_masking = enabled or previous
    try:
        yield
    finally:
        conditioner._use_masking = previous


def generate_section(model, description, source, melody_wav, start, end, context, tail,
                     sample_rate=OUTPUT_SAMPLE_RATE):
    """Generate [start - context, end + tail] with the melody condition and return it from `start` on.

    generate_continuation() drops the melody condition, so the internal API is called directly
    (audiocraft==1.3.0). melody_wav is [1, T] at MELODY_SAMPLE_RATE covering the whole window, or None
    to generate without a melody (a section with rests only).
    """
    from audiocraft.data.audio_utils import convert_audio

    model.set_generation_params(duration=context + (end - start) + tail)
    melody = None
    if melody_wav is not None:
        melody = convert_audio(melody_wav, MELODY_SAMPLE_RATE, model.sample_rate, model.audio_channels)

    with melody_masking(model, melody is None):
        if context <= 0:
            if melody is None:
                return model.generate([description])[0]
            return model.generate_with_chroma([description], melody[None], model.sample_rate)[0]

        context_wav = source[..., round((start - context) * sample_rate):round(start * sample_rate)]
        attributes, prompt_tokens = model._prepare_tokens_and_attributes(
            descriptions=[description], prompt=context_wav[None],
            melody_wavs=None if melody is None else [melody])
        tokens = model._generate_tokens(attributes, prompt_tokens)
        audio = model.generate_audio(tokens)[0]
    # The output starts with the decoded context; cut at the actual prompt token count (640 samples per frame)
    context_samples = prompt_tokens.shape[-1] * model.sample_rate // model.frame_rate
    return audio[..., context_samples:]


def build_modify_text(text_condition, edit_prompt, fallback):
    """Text condition for a regeneration (before the #15 rewrite).

    The edited version's text condition with this edit's prompt appended; songs without a stored
    text condition use the "{genre}, {mood}, {prompt}" fallback as is. At most 1000 characters.
    """
    if text_condition and text_condition.strip():
        text = text_condition.strip()
        if edit_prompt and edit_prompt.strip():
            text += f", {edit_prompt.strip()}"
    else:
        text = fallback
    return text[:MAX_TEXT_CONDITION_CHARS]


def parse_section_seconds(value, name):
    # bool is a subclass of int, so reject it explicitly
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a number, got: {value!r}")
    return float(value)


def load_source_audio(url, sample_rate=OUTPUT_SAMPLE_RATE):
    """Download the version to regenerate and return it as [1, T] mono at sample_rate."""
    import julius
    import soundfile as sf

    res = requests.get(url, timeout=SOURCE_DOWNLOAD_TIMEOUT_SECONDS)
    res.raise_for_status()
    data, source_rate = sf.read(io.BytesIO(res.content), dtype='float32', always_2d=True)
    wav = torch.from_numpy(data.T).mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        wav = julius.resample_frac(wav, source_rate, sample_rate)
    return wav


def melody_audio(melody_vectors, seconds, synthesize):
    """Synthesize the window melody as [1, T] at MELODY_SAMPLE_RATE, padded or cut to the window."""
    audio = torch.from_numpy(synthesize(melody_vectors, sample_rate=MELODY_SAMPLE_RATE))
    return fit_length(audio[None], round(seconds * MELODY_SAMPLE_RATE))


def regenerate_section(model, description, source, melody_vectors, start, end, synthesize,
                       humming_end_seconds=None, sample_rate=OUTPUT_SAMPLE_RATE):
    """Regenerate [start, end) of `source` and return the merged song with the source's exact length.

    Regenerating the whole song (0 to the end) has no context and no tail, so it is a plain
    generate_with_chroma whose output replaces the source.
    """
    source_seconds = source.shape[-1] / sample_rate
    if source_seconds < end <= source_seconds + SECTION_END_TOLERANCE_SECONDS:
        end = source_seconds
    if end - start < MIN_SECTION_SECONDS:
        raise ValueError(f"section must be at least {MIN_SECTION_SECONDS} seconds, got {start}~{end}")
    context, tail = plan_section(source_seconds, start, end)
    window_start, window_end = start - context, end + tail

    # A section with rests only is generated without a melody condition (a rest in the chroma is a C)
    if has_pitched_notes(select_melody(melody_vectors, start, end, humming_end_seconds)):
        window_melody = select_melody(melody_vectors, window_start, window_end, humming_end_seconds)
        melody_wav = melody_audio(window_melody, window_end - window_start, synthesize)
    else:
        melody_wav = None

    generated = generate_section(model, description, source, melody_wav, start, end, context, tail,
                                 sample_rate=sample_rate)
    return merge_section(source, generated.cpu(), start, end, tail, sample_rate=sample_rate)
