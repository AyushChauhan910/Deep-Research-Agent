import json
import re
from typing import Optional


def approx_tokens(text: str) -> int:
    # ~4 chars/token for English — good enough for budgeting decisions
    return len(text) // 4


def extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of an LLM response, tolerating fences and prose."""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "")
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(cleaned)):
        c = cleaned[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."