from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint

from .llm_ollama import OllamaClient
from .pipeline import build_ticket_draft, llm_extract_structured

app = typer.Typer(add_completion=False, invoke_without_command=True)


def _run_impl(
    audio: str,
    out: str = "outputs/ticket_draft.json",
    whisper_model: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    language: Optional[str] = None,
    ollama_model: str = "llama3.1:8b-instruct",
    ollama_url: str = "http://localhost:11434",
    print_transcript: bool = True,
) -> None:
    audio_path = Path(audio)
    if not audio_path.exists():
        raise typer.BadParameter(f"Audio file not found: {audio_path}")

    # Delay ASR import so CLI parsing/help/tests don't require whisper runtime.
    from .whisper_asr import transcribe_audio

    rprint(
        f"[bold]1) Transcribing with Whisper[/bold] "
        f"model={whisper_model} device={device} compute_type={compute_type}"
    )
    transcript_obj = transcribe_audio(
        audio_path=str(audio_path),
        whisper_model=whisper_model,
        device=device,
        compute_type=compute_type,
        language=language,
    )

    if print_transcript:
        rprint("\n[bold]Transcript:[/bold]")
        rprint(transcript_obj.transcript)

    rprint(f"\n[bold]2) Extracting Ticket JSON with local LLM[/bold] model={ollama_model}")
    llm = OllamaClient(model=ollama_model, base_url=ollama_url)
    extraction, _raw_llm = llm_extract_structured(llm=llm, transcript=transcript_obj.transcript)

    rprint("[bold]3) Building TicketDraft + validating[/bold]")
    ticket = build_ticket_draft(transcript_obj=transcript_obj, extraction=extraction)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(ticket.model_dump_json(indent=2), encoding="utf-8")

    rprint(f"\nSaved TicketDraft JSON -> [green]{out_path}[/green]")
    rprint(f"Completeness score: {ticket.quality.completeness_score:.2f}")
    if ticket.quality.missing_fields:
        rprint("[yellow]Missing fields:[/yellow]", ticket.quality.missing_fields)
    if ticket.quality.followup_questions:
        rprint("[yellow]Follow-up questions:[/yellow]")
        for q in ticket.quality.followup_questions:
            rprint(f" - {q}")


@app.callback()
def main(
    ctx: typer.Context,
    audio: Optional[str] = typer.Option(None, help="Path to audio file (wav/mp3/m4a/etc)."),
    out: str = typer.Option("outputs/ticket_draft.json", help="Output JSON path."),
    whisper_model: str = typer.Option(
        "large-v3",
        help="Whisper model size (e.g., small, medium, large-v3).",
    ),
    device: str = typer.Option("cuda", help="Whisper device: cuda or cpu."),
    compute_type: str = typer.Option(
        "float16",
        help="Whisper compute type: float16, int8, int8_float16, etc.",
    ),
    language: Optional[str] = typer.Option(
        None,
        help="Force language (e.g., en). Leave empty to auto-detect.",
    ),
    ollama_model: str = typer.Option("llama3.1:8b-instruct", help="Ollama model name."),
    ollama_url: str = typer.Option("http://localhost:11434", help="Ollama base URL."),
    print_transcript: bool = typer.Option(True, help="Print transcript to console."),
) -> None:
    # Legacy behavior: allow running without subcommand as long as --audio is passed.
    if ctx.invoked_subcommand is not None:
        return

    if audio is None:
        rprint(ctx.get_help())
        raise typer.Exit()

    _run_impl(
        audio=audio,
        out=out,
        whisper_model=whisper_model,
        device=device,
        compute_type=compute_type,
        language=language,
        ollama_model=ollama_model,
        ollama_url=ollama_url,
        print_transcript=print_transcript,
    )


@app.command("run")
def run_command(
    audio: str = typer.Option(..., help="Path to audio file (wav/mp3/m4a/etc)."),
    out: str = typer.Option("outputs/ticket_draft.json", help="Output JSON path."),
    whisper_model: str = typer.Option(
        "large-v3",
        help="Whisper model size (e.g., small, medium, large-v3).",
    ),
    device: str = typer.Option("cuda", help="Whisper device: cuda or cpu."),
    compute_type: str = typer.Option(
        "float16",
        help="Whisper compute type: float16, int8, int8_float16, etc.",
    ),
    language: Optional[str] = typer.Option(
        None,
        help="Force language (e.g., en). Leave empty to auto-detect.",
    ),
    ollama_model: str = typer.Option("llama3.1:8b-instruct", help="Ollama model name."),
    ollama_url: str = typer.Option("http://localhost:11434", help="Ollama base URL."),
    print_transcript: bool = typer.Option(True, help="Print transcript to console."),
) -> None:
    _run_impl(
        audio=audio,
        out=out,
        whisper_model=whisper_model,
        device=device,
        compute_type=compute_type,
        language=language,
        ollama_model=ollama_model,
        ollama_url=ollama_url,
        print_transcript=print_transcript,
    )


if __name__ == "__main__":
    app()
