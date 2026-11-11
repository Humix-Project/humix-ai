from typing import Callable, Dict, Optional
import os

DEFAULT_STRATEGY = os.getenv("PROMPT_STRATEGY", "rewrite")

# Template parts for the genre/mood values the backend documents; unknown values pass through as-is.
# "instruments" is only used when the user prompt is empty, so it never fights instruments the user asked for.
GENRE_TEMPLATES = {
    "pop": {"style": "modern pop track, polished production", "instruments": "punchy drums, electric bass, bright synths, piano"},
    "jazz": {"style": "jazz piece, live band recording", "instruments": "upright bass, brushed drums, piano comping, saxophone"},
    "classical": {"style": "classical piece, concert hall reverb", "instruments": "orchestral strings, piano, woodwinds"},
    "hiphop": {"style": "hip hop instrumental beat", "instruments": "boom bap drums, deep sub bass, sampled keys"},
}
MOOD_TEMPLATES = {
    "upbeat": "upbeat, energetic and cheerful",
    "calm": "calm, relaxing and gentle",
    "romantic": "romantic, warm and tender",
    "sad": "sad, melancholic and emotional",
}

TRANSLATE_INSTRUCTION = (
    "Translate the user's Korean music request into natural English. "
    "Output only the translation in one line, without quotes or explanations."
)

REWRITE_INSTRUCTION = """You convert a user's music request (often Korean, casual) into a short English description for the MusicGen text-to-music model.
MusicGen understands short caption-style descriptions: instruments, playing style, texture, tempo, energy and production.

Rules:
- Output ONE line of 3 to 8 comma-separated English descriptors. No sentences, quotes or explanations.
- Describe only what the request asks for. Genre and mood are given for context; do not repeat them.
- Turn vague feelings into concrete musical terms (instruments, tempo, dynamics, texture).
- Instrumental only: never mention vocals, singing or lyrics.
- Structure or timing words (e.g. "later", "at the end") can't be controlled; express them as texture (e.g. "strings swelling in").

Examples:
genre: pop / mood: upbeat / request: 우쿨렐레랑 박수 소리 넣어서
ukulele strumming, hand claps, light percussion, sunny and bouncy
genre: jazz / mood: calm / request: 트럼펫 소리가 은은하게
muted trumpet, soft and mellow, slow tempo, smoky lounge
genre: hiphop / mood: sad / request: 드럼을 좀 더 세게
hard-hitting drums, punchy kick and snare, heavy groove
genre: pop / mood: calm / request: 좀 더 신나게
lively and energetic, faster groove, bright synths
genre: jazz / mood: upbeat / request: 템포 느리게 해줘
slow tempo, relaxed swing, laid-back
genre: classical / mood: sad / request: 영화 OST처럼
cinematic film score, sweeping strings, emotional crescendo
genre: hiphop / mood: romantic / request: 새벽 감성으로
late night lo-fi mood, mellow electric piano, vinyl crackle, laid-back beat
genre: pop / mood: sad / request: 마음 아프게 해줘
heartbreaking piano, soft strings, slow tempo, emotional
genre: pop / mood: upbeat / request: 레트로한 80년대 느낌
80s retro synthwave, analog synths, gated reverb drums, funky bass
genre: classical / mood: calm / request: 비 오는 날 카페에서 듣는 느낌
soft piano, gentle cello, rainy cafe ambience, slow tempo, mellow
genre: pop / mood: upbeat / request: 신나게 춤출 수 있게
danceable four-on-the-floor beat, groovy bass, bright synths, 124 bpm
genre: jazz / mood: romantic / request: 색소폰 솔로 넣어줘
expressive saxophone solo, smooth, warm tone
genre: hiphop / mood: upbeat / request: 기타 리프 들어간 느낌으로
catchy electric guitar riff, punchy drums, bouncy groove
genre: classical / mood: romantic / request: 하프로 시작해서 점점 풍성하게
delicate harp arpeggios, strings gradually building, lush and tender
genre: pop / mood: calm / request: 몽환적으로
dreamy atmosphere, airy synth pads, reverb-drenched guitar, slow tempo
genre: pop / mood: sad / request: with piano only
solo piano, soft and intimate"""

REWRITE_NOTEMPLATE_INSTRUCTION = """You convert a user's music request (genre, mood, prompt) into a short, dense English music caption for MusicGen.
MusicGen understands short caption-style descriptions: instruments, playing style, texture, tempo, energy, and production.

Rules:
- Output ONE line of 3 to 8 comma-separated English descriptors. No sentences, quotes, or explanations.
- Do NOT use rigid preset template phrases or prefixes (like "modern pop track" or "concert hall reverb").
- Blend the given genre, mood, and user prompt into a cohesive, accurate musical description focusing heavily on the user's specific request.
- Instrumental only: never mention vocals, singing, or lyrics.

Examples:
genre: pop / mood: upbeat / request: 어쿠스틱 기타랑 부드러운 드럼으로
acoustic pop, acoustic guitar strumming, soft brush drums, warm and bright tone, upbeat groove
genre: jazz / mood: calm / request: 피아노랑 콘트라베이스로 잔잔하게
intimate jazz, soft acoustic piano, mellow upright bass, slow tempo, gentle and calm vibe
genre: classical / mood: romantic / request: 잔잔한 피아노 위주로, 후반부에 스트링이 들어오는 느낌
romantic classical, soft solo piano, gentle texture, strings gradually building in, warm and tender
genre: hiphop / mood: sad / request: 비트를 더 묵직하게, 808 베이스 강조
melancholic hip hop, heavy 808 bass, deep sub-bass, hard-hitting dark drums, emotional vibe"""


def build_simple_description(genre: Optional[str], mood: Optional[str], prompt: Optional[str]) -> str:
    # Plain comma-joined condition from #12; empty fields are skipped
    parts = [part.strip() for part in (genre, mood, prompt) if part and part.strip()]
    return ", ".join(parts)


def _needs_translation(text: str) -> bool:
    # T5 maps Hangul (and other non-Latin scripts) to <unk>, so only non-ASCII text needs the LLM
    return any(ord(ch) > 127 for ch in text)


def build_translated_description(genre: Optional[str], mood: Optional[str], prompt: Optional[str]) -> str:
    # Method A: literal translation of the prompt only, joined like simple
    from gemini_client import generate_text
    if prompt and prompt.strip() and _needs_translation(prompt):
        prompt = generate_text(TRANSLATE_INSTRUCTION, prompt.strip())
    return build_simple_description(genre, mood, prompt)


def _template_parts(genre: Optional[str], mood: Optional[str], has_prompt: bool):
    genre_key = (genre or "").strip().lower()
    mood_key = (mood or "").strip().lower()
    genre_template = GENRE_TEMPLATES.get(genre_key)
    if genre_template:
        genre_part = genre_template["style"] if has_prompt else f"{genre_template['style']}, {genre_template['instruments']}"
    else:
        genre_part = (genre or "").strip()
    mood_part = MOOD_TEMPLATES.get(mood_key, (mood or "").strip())
    return genre_part, mood_part


def build_rewritten_description(genre: Optional[str], mood: Optional[str], prompt: Optional[str]) -> str:
    # Method D: genre/mood from templates, the free-form prompt rewritten by the LLM
    from gemini_client import generate_text
    prompt = (prompt or "").strip()
    genre_part, mood_part = _template_parts(genre, mood, has_prompt=bool(prompt))
    prompt_part = ""
    if prompt:
        request = f"genre: {genre or '-'} / mood: {mood or '-'} / request: {prompt}"
        prompt_part = generate_text(REWRITE_INSTRUCTION, request).splitlines()[0].strip()
    # User's request goes right after the genre so it is not drowned out by the mood words
    return ", ".join(part for part in (genre_part, prompt_part, mood_part) if part)


def build_rewritten_notemplate_description(genre: Optional[str], mood: Optional[str], prompt: Optional[str]) -> str:
    # Method D-slim: No rigid templates, pure LLM-driven tag synthesis based on genre, mood, and prompt
    from gemini_client import generate_text
    
    g = (genre or "").strip()
    m = (mood or "").strip()
    p = (prompt or "").strip()
    
    request_payload = f"genre: {g or '-'} / mood: {m or '-'} / request: {p if p else 'none'}"
    rewritten_tags = generate_text(REWRITE_NOTEMPLATE_INSTRUCTION, request_payload).splitlines()[0].strip()
    
    return rewritten_tags if rewritten_tags else ""


# Strategy name -> builder; new prompt strategies register here
STRATEGIES: Dict[str, Callable[[Optional[str], Optional[str], Optional[str]], str]] = {
    "simple": build_simple_description,
    "translate": build_translated_description,
    "rewrite": build_rewritten_description,
    "rewrite_notemplate": build_rewritten_notemplate_description,
}


def build_description(genre: Optional[str], mood: Optional[str], prompt: Optional[str], strategy: str = DEFAULT_STRATEGY) -> str:
    # MusicGen takes a single text description per sample
    builder = STRATEGIES.get(strategy)
    if builder is None:
        raise ValueError(f"Unknown prompt strategy: {strategy}")
    if builder is build_simple_description:
        return builder(genre, mood, prompt)

    try:
        description = builder(genre, mood, prompt)
        if description and description.strip():
            return description.strip()
        print(f"Prompt strategy '{strategy}' returned an empty description. Falling back to simple.")
    except Exception as e:
        print(f"Prompt strategy '{strategy}' failed: {e}. Falling back to simple.")
    return build_simple_description(genre, mood, prompt)
