from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


UNKNOWN_VALUES = {"", "unknown", "n/a", "na", "none", "null"}


def _is_meaningful(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in UNKNOWN_VALUES
    if isinstance(value, list):
        return any(_is_meaningful(item) for item in value)
    return True


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value)


def build_ticket_query_text(ticket_data: Mapping[str, Any]) -> str:
    content = ticket_data.get("content", {}) if isinstance(ticket_data, Mapping) else {}
    environment = ticket_data.get("environment", {}) if isinstance(ticket_data, Mapping) else {}

    lines: list[str] = []
    title = _clean_text(content.get("title"))
    summary = _clean_text(content.get("summary"))
    description = _clean_text(content.get("description"))
    observed = _clean_text(content.get("observed_behavior"))
    expected = _clean_text(content.get("expected_behavior"))

    if _is_meaningful(title):
        lines.append(f"Title: {title}")
    if _is_meaningful(summary):
        lines.append(f"Summary: {summary}")
    if _is_meaningful(description):
        lines.append(f"Description: {description}")
    if _is_meaningful(observed):
        lines.append(f"Observed behavior: {observed}")
    if _is_meaningful(expected):
        lines.append(f"Expected behavior: {expected}")

    steps = content.get("steps_to_reproduce", [])
    if isinstance(steps, list):
        step_lines = [f"- {_clean_text(step)}" for step in steps if _is_meaningful(step)]
        if step_lines:
            lines.append("Steps to reproduce:")
            lines.extend(step_lines)

    os_info = environment.get("os", {}) if isinstance(environment, Mapping) else {}
    app_info = environment.get("app", {}) if isinstance(environment, Mapping) else {}
    device_info = environment.get("device", {}) if isinstance(environment, Mapping) else {}

    os_name = _clean_text(os_info.get("name"))
    os_version = _clean_text(os_info.get("version"))
    app_name = _clean_text(app_info.get("name"))
    app_version = _clean_text(app_info.get("version"))
    device_model = _clean_text(device_info.get("model"))

    env_parts = []
    if _is_meaningful(os_name):
        env_parts.append(os_name if not _is_meaningful(os_version) else f"{os_name} {os_version}".strip())
    if _is_meaningful(app_name):
        env_parts.append(app_name if not _is_meaningful(app_version) else f"{app_name} {app_version}".strip())
    if _is_meaningful(device_model):
        env_parts.append(device_model)

    if env_parts:
        lines.append(f"Environment: {', '.join(env_parts)}")

    if not lines:
        return json.dumps(ticket_data, ensure_ascii=False)
    return "\n".join(lines)


def _load_rag_module() -> Any:
    repo_root = Path(__file__).resolve().parent.parent
    rag_path = repo_root / "05_rag_pipeline.py"
    if not rag_path.exists():
        raise FileNotFoundError(f"RAG pipeline file not found: {rag_path}")

    spec = importlib.util.spec_from_file_location("its_rag_pipeline_runtime", rag_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load RAG pipeline module from {rag_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_utf8_console() -> None:
    # RAG scripts print Unicode symbols that can fail on default Windows code pages.
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _patch_rag_ollama_client(rag_module: Any, rag_model: str, ollama_url: str) -> None:
    from .llm_ollama import OllamaClient

    class PatchedOllamaLLM:
        # Chat-only compatibility wrapper for RAG pipeline.
        def __init__(self, model: str = rag_model, base_url: str = ollama_url):
            self.model = model
            self.client = OllamaClient(model=model, base_url=base_url)

        def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 2048, system: str = "") -> str:
            _ = max_tokens  # Ollama chat endpoint ignores this in our client wrapper.
            return self.client.chat(system=system or "You are a helpful IT support assistant.", user=prompt, temperature=temperature)

        def chat(self, messages: list[dict[str, str]], temperature: float = 0.1) -> str:
            if not messages:
                return ""
            system_chunks = [m.get("content", "") for m in messages if m.get("role") == "system"]
            user_chunks = [m.get("content", "") for m in messages if m.get("role") in {"user", "assistant"}]
            system = "\n\n".join(system_chunks) if system_chunks else "You are a helpful IT support assistant."
            user = "\n\n".join(user_chunks) if user_chunks else ""
            return self.client.chat(system=system, user=user, temperature=temperature)

    rag_module.MODEL_NAME = rag_model
    rag_module.OllamaLLM = PatchedOllamaLLM


def resolve_ticket_json_with_rag(
    ticket_json_path: str | Path,
    rag_model: str = "qwen3:8b",
    ollama_url: str = "http://localhost:11434",
    ticket_top_k: int = 3,
    kb_top_k: int = 3,
    verbose: bool = False,
) -> dict[str, Any]:
    ticket_path = Path(ticket_json_path)
    if not ticket_path.exists():
        raise FileNotFoundError(f"Ticket JSON not found: {ticket_path}")

    ticket_data = json.loads(ticket_path.read_text(encoding="utf-8"))
    ticket_text = build_ticket_query_text(ticket_data)

    _ensure_utf8_console()
    rag_module = _load_rag_module()
    _patch_rag_ollama_client(rag_module=rag_module, rag_model=rag_model, ollama_url=ollama_url)
    rag = rag_module.RAGPipeline()
    rag_result = rag.generate_resolution(
        ticket_text=ticket_text,
        ticket_top_k=ticket_top_k,
        kb_top_k=kb_top_k,
        verbose=verbose,
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ticket_json_path": str(ticket_path),
        "ticket_query_text": ticket_text,
        "resolution": rag_result.get("resolution", ""),
        "similar_tickets": rag_result.get("similar_tickets", []),
        "kb_articles": rag_result.get("kb_articles", []),
    }


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def save_resolution_json(result: Mapping[str, Any], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return path
