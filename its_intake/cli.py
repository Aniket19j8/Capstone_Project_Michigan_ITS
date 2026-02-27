from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint

from .llm_ollama import OllamaClient
from .pipeline import build_ticket_draft, llm_extract_structured
from .schemas import TicketDraft

app = typer.Typer(add_completion=False, invoke_without_command=True)


def _build_ticket_from_audio(
    audio: str,
    whisper_model: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    language: Optional[str] = None,
    ollama_model: str = "qwen3:8b",
    ollama_url: str = "http://localhost:11434",
    print_transcript: bool = True,
) -> TicketDraft:
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
    return build_ticket_draft(transcript_obj=transcript_obj, extraction=extraction)


def _save_ticket(ticket: TicketDraft, out: str) -> Path:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(ticket.model_dump_json(indent=2), encoding="utf-8")
    return out_path


def _print_ticket_quality(ticket: TicketDraft) -> None:
    rprint(f"Completeness score: {ticket.quality.completeness_score:.2f}")
    if ticket.quality.missing_fields:
        rprint("[yellow]Missing fields:[/yellow]", ticket.quality.missing_fields)
    if ticket.quality.followup_questions:
        rprint("[yellow]Follow-up questions:[/yellow]")
        for q in ticket.quality.followup_questions:
            rprint(f" - {q}")


def _run_impl(
    audio: str,
    out: str = "outputs/ticket.json",
    whisper_model: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    language: Optional[str] = None,
    ollama_model: str = "qwen3:8b",
    ollama_url: str = "http://localhost:11434",
    print_transcript: bool = True,
) -> Path:
    ticket = _build_ticket_from_audio(
        audio=audio,
        whisper_model=whisper_model,
        device=device,
        compute_type=compute_type,
        language=language,
        ollama_model=ollama_model,
        ollama_url=ollama_url,
        print_transcript=print_transcript,
    )
    out_path = _save_ticket(ticket=ticket, out=out)

    rprint(f"\nSaved TicketDraft JSON -> [green]{out_path}[/green]")
    _print_ticket_quality(ticket)
    return out_path


def _resolve_ticket_impl(
    ticket_json: str,
    out: str = "outputs/resolution.json",
    rag_model: str = "qwen3:8b",
    rag_ollama_url: str = "http://localhost:11434",
    ticket_top_k: int = 3,
    kb_top_k: int = 3,
    rag_verbose: bool = False,
) -> Path:
    ticket_path = Path(ticket_json)
    if not ticket_path.exists():
        raise typer.BadParameter(f"Ticket JSON not found: {ticket_path}")

    from .end_to_end import resolve_ticket_json_with_rag, save_resolution_json

    rprint(
        f"\n[bold]4) Running RAG resolution[/bold] "
        f"model={rag_model} ticket_top_k={ticket_top_k} kb_top_k={kb_top_k}"
    )
    result = resolve_ticket_json_with_rag(
        ticket_json_path=ticket_path,
        rag_model=rag_model,
        ollama_url=rag_ollama_url,
        ticket_top_k=ticket_top_k,
        kb_top_k=kb_top_k,
        verbose=rag_verbose,
    )
    out_path = save_resolution_json(result=result, out_path=out)
    rprint(f"Saved RAG resolution JSON -> [green]{out_path}[/green]")
    return out_path


@app.callback()
def main(
    ctx: typer.Context,
    audio: Optional[str] = typer.Option(None, help="Path to audio file (wav/mp3/m4a/etc)."),
    out: str = typer.Option("outputs/ticket.json", help="Output JSON path."),
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
    ollama_model: str = typer.Option("qwen3:8b", help="Ollama model name."),
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
    out: str = typer.Option("outputs/ticket.json", help="Output JSON path."),
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
    ollama_model: str = typer.Option("qwen3:8b", help="Ollama model name."),
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


@app.command("resolve-ticket")
def resolve_ticket_command(
    ticket_json: str = typer.Option("outputs/ticket.json", help="Path to TicketDraft JSON."),
    out: str = typer.Option("outputs/resolution.json", help="Output resolution JSON path."),
    rag_model: str = typer.Option("qwen3:8b", help="Ollama model for RAG generation."),
    rag_ollama_url: str = typer.Option("http://localhost:11434", help="Ollama base URL for RAG."),
    ticket_top_k: int = typer.Option(3, min=1, max=20, help="Number of similar tickets to retrieve."),
    kb_top_k: int = typer.Option(3, min=1, max=20, help="Number of KB chunks to retrieve."),
    rag_verbose: bool = typer.Option(False, help="Enable verbose retriever logging."),
) -> None:
    _resolve_ticket_impl(
        ticket_json=ticket_json,
        out=out,
        rag_model=rag_model,
        rag_ollama_url=rag_ollama_url,
        ticket_top_k=ticket_top_k,
        kb_top_k=kb_top_k,
        rag_verbose=rag_verbose,
    )


@app.command("run-e2e")
def run_e2e_command(
    audio: str = typer.Option(..., help="Path to audio file (wav/mp3/m4a/etc)."),
    out: str = typer.Option("outputs/ticket.json", help="Output TicketDraft JSON path."),
    resolution_out: str = typer.Option("outputs/resolution.json", help="Output RAG resolution JSON path."),
    whisper_model: str = typer.Option("large-v3", help="Whisper model size."),
    device: str = typer.Option("cuda", help="Whisper device: cuda or cpu."),
    compute_type: str = typer.Option("float16", help="Whisper compute type."),
    language: Optional[str] = typer.Option(None, help="Force language (e.g., en)."),
    ollama_model: str = typer.Option("qwen3:8b", help="Ollama model for ticket extraction."),
    ollama_url: str = typer.Option("http://localhost:11434", help="Ollama base URL for ticket extraction."),
    print_transcript: bool = typer.Option(True, help="Print transcript to console."),
    rag_model: str = typer.Option("qwen3:8b", help="Ollama model for RAG generation."),
    rag_ollama_url: str = typer.Option("http://localhost:11434", help="Ollama base URL for RAG."),
    ticket_top_k: int = typer.Option(3, min=1, max=20, help="Number of similar tickets to retrieve."),
    kb_top_k: int = typer.Option(3, min=1, max=20, help="Number of KB chunks to retrieve."),
    rag_verbose: bool = typer.Option(False, help="Enable verbose retriever logging."),
) -> None:
    ticket_out = _run_impl(
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
    _resolve_ticket_impl(
        ticket_json=str(ticket_out),
        out=resolution_out,
        rag_model=rag_model,
        rag_ollama_url=rag_ollama_url,
        ticket_top_k=ticket_top_k,
        kb_top_k=kb_top_k,
        rag_verbose=rag_verbose,
    )


if __name__ == "__main__":
    app()
