from __future__ import annotations

from datetime import datetime, timezone
from typing import Tuple
from uuid import uuid4

from pydantic import ValidationError

from .json_utils import extract_json_object
from .llm_ollama import OllamaClient
from .prompts import SYSTEM_PROMPT
from .schemas import ExtractionOutput, TicketDraft, Transcript, IntakeArtifacts, UNKNOWN


REQUIRED_PATHS = [
    "content.title",
    "content.summary",
    "content.observed_behavior",
    "content.expected_behavior",
    "content.steps_to_reproduce",
    "environment.os.name",
    "environment.os.version",
    "environment.app.name",
    "environment.app.version",
    "environment.device.model",
]


def _is_unknown(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "unknown", "n/a", "na"}:
        return True
    if isinstance(value, list) and len(value) == 0:
        return True
    return False


def _get_path(obj, path: str):
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def compute_missing_fields(extraction: ExtractionOutput) -> list[str]:
    data = extraction.model_dump()
    missing = []
    for p in REQUIRED_PATHS:
        v = _get_path(data, p)
        if _is_unknown(v):
            missing.append(p)
    return missing


def compute_completeness(extraction: ExtractionOutput) -> float:
    data = extraction.model_dump()
    filled = 0
    for p in REQUIRED_PATHS:
        v = _get_path(data, p)
        if not _is_unknown(v):
            filled += 1
    return filled / max(1, len(REQUIRED_PATHS))


def llm_extract_structured(
    llm: OllamaClient,
    transcript: str,
    max_retries: int = 2,
) -> Tuple[ExtractionOutput, str]:
    """
    Calls LLM to generate ExtractionOutput JSON and validates it.
    If invalid, retries with repair instructions.
    """
    user_prompt = f'TRANSCRIPT:\n"""\n{transcript}\n"""\n\nReturn JSON only.'

    last_text = ""
    last_err = ""

    for attempt in range(max_retries + 1):
        out_text = llm.chat(system=SYSTEM_PROMPT, user=user_prompt, temperature=0.2)
        last_text = out_text

        try:
            data = extract_json_object(out_text)
            extraction = ExtractionOutput.model_validate(data)
            return extraction, out_text
        except (ValueError, ValidationError) as e:
            last_err = str(e)
            user_prompt = (
                "Your previous output was INVALID.\n"
                f"Validation/parsing error:\n{last_err}\n\n"
                f'TRANSCRIPT:\n"""\n{transcript}\n"""\n\n'
                "Fix the JSON to match the required output format. Output JSON ONLY."
            )

    raise RuntimeError(f"LLM failed to produce valid JSON after retries. Last error: {last_err}\nLast output:\n{last_text}")


def build_ticket_draft(transcript_obj: Transcript, extraction: ExtractionOutput) -> TicketDraft:
    # Merge missing fields computed deterministically
    missing = set(extraction.quality.missing_fields or [])
    missing.update(compute_missing_fields(extraction))

    extraction.quality.missing_fields = sorted(missing)
    extraction.quality.completeness_score = compute_completeness(extraction)

    ticket = TicketDraft(
        ticket_id=uuid4(),
        created_at=datetime.now(timezone.utc),
        source="voice",
        status="new",
        content=extraction.content,
        environment=extraction.environment,
        evidence=extraction.evidence,
        impact=extraction.impact,
        classification=extraction.classification,
        routing=extraction.routing,
        intake_artifacts=IntakeArtifacts(
            transcript=transcript_obj.transcript,
            transcript_segments=transcript_obj.segments,
            whisper=transcript_obj.meta,
        ),
        quality=extraction.quality,
    )
    return ticket