import hashlib
import json
import os
import threading
import time

# Flash-Lite: fastest and the most generous free-tier daily quota among the Flash models
DEFAULT_MODEL = "gemini-3.5-flash-lite"
REQUEST_TIMEOUT_MS = 10000
MAX_ATTEMPTS = 3
RETRYABLE_CODES = (429, 500, 503)

_client = None
_cache_lock = threading.Lock()


def _get_client():
    global _client
    if _client is None:
        # Imported lazily so the server runs without google-genai until an LLM strategy is used
        from google import genai
        from google.genai import types
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        _client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))
    return _client


def _cache_key(model, system_instruction, user_text):
    raw = json.dumps([model, system_instruction, user_text], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_cache(path, key):
    if not path or not os.path.exists(path):
        return None
    with _cache_lock, open(path, encoding="utf-8") as f:
        return json.load(f).get(key)


def _write_cache(path, key, text):
    if not path:
        return
    with _cache_lock:
        cache = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                cache = json.load(f)
        cache[key] = text
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)


def _generate(model, system_instruction, user_text, thinking_config):
    from google.genai import types
    response = _get_client().models.generate_content(
        model=model,
        contents=user_text,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            thinking_config=thinking_config,
            temperature=0.2,
        ),
    )
    return (response.text or "").strip()


def generate_text(system_instruction: str, user_text: str) -> str:
    """Short single-turn Gemini call with thinking minimized, retries and an optional disk cache.

    GEMINI_MODEL overrides the model; GEMINI_CACHE_PATH (experiments only) reuses earlier answers
    so re-running a comparison does not spend free-tier requests again.
    """
    model = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
    cache_path = os.environ.get("GEMINI_CACHE_PATH")
    key = _cache_key(model, system_instruction, user_text)
    cached = _read_cache(cache_path, key)
    if cached is not None:
        return cached

    # Imported after the cache check so cached runs work without google-genai installed
    from google.genai import errors, types

    # Thinking is wasted on one-line rewrites; Gemini 3 Lite models take a level, others a budget
    thinking_config = types.ThinkingConfig(thinking_level="minimal")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            text = _generate(model, system_instruction, user_text, thinking_config)
            break
        except errors.APIError as e:
            if e.code == 400 and "thinking" in str(e).lower() and thinking_config.thinking_level:
                thinking_config = types.ThinkingConfig(thinking_budget=0)
                continue
            if e.code in RETRYABLE_CODES and attempt < MAX_ATTEMPTS:
                wait = 2 ** attempt
                print(f"Gemini {e.code}, retrying in {wait}s ({attempt}/{MAX_ATTEMPTS})")
                time.sleep(wait)
                continue
            raise
    else:
        raise RuntimeError("Gemini call did not succeed.")

    if text:
        _write_cache(cache_path, key, text)
    return text
