from __future__ import annotations

import json


def extract_json_object(text: str) -> dict:
    """
    Attempts to parse a JSON object from model output.
    If the model includes extra text, this tries to recover the outermost {...}.
    """
    text = (text or "").strip()

    # First try: direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Recovery: find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM output.")

    candidate = text[start : end + 1]
    obj = json.loads(candidate)
    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object.")
    return obj