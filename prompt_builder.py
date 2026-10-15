from typing import Callable, Dict, Optional

DEFAULT_STRATEGY = "simple"


def build_simple_description(genre: Optional[str], mood: Optional[str], prompt: Optional[str]) -> str:
    # Plain comma-joined condition from #12; empty fields are skipped
    parts = [part.strip() for part in (genre, mood, prompt) if part and part.strip()]
    return ", ".join(parts)


# Strategy name -> builder; new prompt strategies (template, LLM rewrite, ...) register here
STRATEGIES: Dict[str, Callable[[Optional[str], Optional[str], Optional[str]], str]] = {
    "simple": build_simple_description,
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
