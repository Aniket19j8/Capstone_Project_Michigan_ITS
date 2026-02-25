from __future__ import annotations

from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel

from .schemas import Transcript, TranscriptSegment, WhisperMeta


def transcribe_audio(
    audio_path: str,
    whisper_model: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    language: Optional[str] = None,
) -> Transcript:
    """
    Local Whisper transcription using faster-whisper.
    Requires ffmpeg available in PATH.
    """
    audio_path = str(Path(audio_path))

    model = WhisperModel(whisper_model, device=device, compute_type=compute_type)
    segments_iter, info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=True,
        beam_size=5,
    )

    segments = []
    transcript_lines = []

    for s in segments_iter:
        text = (s.text or "").strip()
        if not text:
            continue
        segments.append(TranscriptSegment(start=float(s.start), end=float(s.end), text=text))
        transcript_lines.append(text)

    transcript_text = " ".join(transcript_lines).strip()

    meta = WhisperMeta(
        model=whisper_model,
        device=device,
        compute_type=compute_type,
        language=getattr(info, "language", None),
        duration=getattr(info, "duration", None),
    )

    return Transcript(transcript=transcript_text, segments=segments, meta=meta)