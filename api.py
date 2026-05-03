"""
FastAPI layer for the React app — same brain as streamlit_app, but JSON + CORS.

Run: uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import threading
import importlib.util
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── config ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:4b")
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE}/api/generate"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE}/api/chat"

# Ollama session — keeps TCP warm so the UI feels less chunky
_ollama_session = requests.Session()
MIN_RERANK_SCORE = -2.0
MIN_KB_RERANK_SCORE = -5.0
MAX_CONTEXT_TICKETS = 3
MAX_CONTEXT_KB = 3
MAX_TICKET_TEXT = 420
MAX_KB_TEXT = 700

sys.path.insert(0, str(BASE_DIR))

app = FastAPI(title="ITS API", version="1.3")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── lazy-load retriever and ticket lookup ────────────────────────────────────
_ticket_retriever = None
_kb_retriever = None
_retriever_error = None
_ticket_lookup: Dict[str, Dict[str, Any]] | None = None


def _load_retriever() -> bool:
    global _ticket_retriever, _kb_retriever, _retriever_error
    if _ticket_retriever is not None:
        return True
    try:
        spec = importlib.util.spec_from_file_location(
            "hybrid_retrieval", str(BASE_DIR / "04_hybrid_retrieval.py")
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        _ticket_retriever = mod.HybridRetriever("its_tickets", load_reranker=True)
        _kb_retriever = mod.HybridRetriever("its_knowledge_base", load_reranker=True)
        _retriever_error = None
        return True
    except Exception as e:
        _retriever_error = str(e)
        return False


def _load_ticket_lookup() -> Dict[str, Dict[str, Any]]:
    global _ticket_lookup
    if _ticket_lookup is not None:
        return _ticket_lookup

    lookup: Dict[str, Dict[str, Any]] = {}
    csv_path = BASE_DIR / "data" / "processed" / "all_tickets.csv"
    if not csv_path.exists():
        _ticket_lookup = lookup
        return lookup

    try:
        df = pd.read_csv(csv_path).fillna("")
        for _, row in df.iterrows():
            record = row.to_dict()
            unified_id = str(record.get("unified_id", "")).strip()
            ticket_id = str(record.get("ticket_id", "")).strip()
            if unified_id:
                lookup[unified_id] = record
            if ticket_id and ticket_id not in lookup:
                lookup[ticket_id] = record
    except Exception:
        lookup = {}

    _ticket_lookup = lookup
    return lookup


def _strip_thinking_tokens(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(
        r"^\s*(We are given|Let's break down|But note:|Wait,|Steps:|Step \d+:).*",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return text.strip()


def _safe_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:limit]


def _score(doc: Dict[str, Any]) -> float:
    return float(doc.get("rerank_score", doc.get("rrf_score", doc.get("score", 0))) or 0)


def _is_relevant(doc: Dict[str, Any], min_score: float = MIN_RERANK_SCORE) -> bool:
    score = doc.get("rerank_score", None)
    return score is None or float(score) >= min_score


def _dedupe_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for doc in docs:
        doc_id = str(doc.get("id", ""))
        if doc_id and doc_id not in seen:
            unique.append(doc)
            seen.add(doc_id)
    return unique


def _filter_ticket_results(results: List[Dict[str, Any]], limit: int = MAX_CONTEXT_TICKETS) -> List[Dict[str, Any]]:
    filtered = [doc for doc in _dedupe_docs(results) if _is_relevant(doc, MIN_RERANK_SCORE)]
    return filtered[:limit]


def _filter_kb_results(results: List[Dict[str, Any]], limit: int = MAX_CONTEXT_KB) -> List[Dict[str, Any]]:
    unique_docs = _dedupe_docs(results)
    filtered = [doc for doc in unique_docs if _is_relevant(doc, MIN_KB_RERANK_SCORE)]
    if filtered:
        return filtered[:limit]
    ranked = sorted(unique_docs, key=_score, reverse=True)
    return ranked[:limit]


def _ticket_record_for_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    lookup = _load_ticket_lookup()
    doc_id = str(doc.get("id", "")).strip()
    ticket_id = str(doc.get("metadata", {}).get("ticket_id", "")).strip()
    return lookup.get(doc_id) or lookup.get(ticket_id) or {}


def _clean_kb_snippet(text: str, limit: int = 320) -> str:
    text = str(text or "")
    text = re.sub(r"#{1,6}\s*", "", text)
    text = text.replace("```", " ")
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _format_ticket_context(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "No sufficiently relevant historical tickets found."

    parts: List[str] = []
    for idx, doc in enumerate(results, start=1):
        meta = doc.get("metadata", {})
        rec = _ticket_record_for_doc(doc)
        issue = _safe_text(rec.get("title_clean") or meta.get("title") or rec.get("title") or "", 180)
        desc = _safe_text(rec.get("description_clean") or rec.get("description") or doc.get("text", ""), MAX_TICKET_TEXT)
        resolution = _safe_text(rec.get("resolution_clean") or rec.get("resolution") or "", 220)
        status = _safe_text(meta.get("status") or rec.get("status") or "Unknown", 40)
        component = _safe_text(meta.get("component") or rec.get("component") or "General", 60)
        ticket_id = _safe_text(meta.get("ticket_id") or rec.get("ticket_id") or doc.get("id") or f"Ticket-{idx}", 50)

        parts.append(
            f"Ticket {idx}: [{ticket_id}]\n"
            f"Component: {component}\n"
            f"Status: {status}\n"
            f"Issue: {issue}\n"
            f"Evidence: {desc}\n"
            f"Resolution Used: {resolution if resolution else 'No recorded resolution available in ticket history.'}\n"
        )
    return "\n".join(parts)


def _format_kb_context(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "No sufficiently relevant KB articles found."

    parts: List[str] = []
    for idx, doc in enumerate(results, start=1):
        meta = doc.get("metadata", {})
        source = _safe_text(meta.get("source_file") or meta.get("doc_title") or "KB", 100)
        section = _safe_text(meta.get("section") or "General", 80)
        text = _clean_kb_snippet(doc.get("text", ""), MAX_KB_TEXT)
        parts.append(
            f"KB {idx}: [{source}]\n"
            f"Section: {section}\n"
            f"Content: {text}\n"
        )
    return "\n".join(parts)


def _extract_json_object(text: str) -> Dict[str, Any] | None:
    cleaned = _strip_thinking_tokens(text)
    if not cleaned:
        return None

    decoder = json.JSONDecoder()
    for start, ch in enumerate(cleaned):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[start:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _kb_source_label(doc: Dict[str, Any]) -> str:
    meta = doc.get("metadata", {})
    return _safe_text(meta.get("source_file") or meta.get("doc_title") or "KB", 100)


def _default_email(problem: str, steps: List[Dict[str, str]], escalate: List[Dict[str, str]]) -> Dict[str, str]:
    subject = _safe_text(f"Update on your support ticket: {problem}", 120)
    lines = [
        "Hi,",
        "",
        f"Thank you for reaching out regarding {problem.lower()}. Based on our initial review, we recommend following the steps below carefully.",
        "",
    ]
    for idx, item in enumerate(steps[:5], start=1):
        lines.append(f"{idx}. {item.get('step', f'Step {idx}')}: {item.get('details', '')}")
        lines.append("")
    if escalate:
        owners = ", ".join(sorted({item.get("owner", "the appropriate support team") for item in escalate if item.get("owner")}))
        lines.append(
            "If the issue continues after completing these steps, please reply with any error messages, screenshots, your device and operating system, and the exact point where the process fails so we can escalate it to "
            f"{owners or 'the appropriate support team'}."
        )
        lines.append("")
    lines.extend(["Best regards,", "IT Support"])
    return {"subject": subject, "body": "\n".join(lines).strip()}


def _normalize_blueprint(data: Dict[str, Any], similar_tickets: List[Dict[str, Any]], kb_results: List[Dict[str, Any]], desc: str) -> Dict[str, Any]:
    problem = _safe_text(data.get("problem") or desc, 220)

    resolution = data.get("resolution", [])
    if isinstance(resolution, str):
        resolution = [resolution]
    resolution = [re.sub(r"\s+", " ", str(x)).strip() for x in resolution if str(x).strip()][:2]

    similar = data.get("similar_tickets", [])
    if isinstance(similar, str):
        similar = [similar]

    normalized_similar = []
    for item in similar[:3]:
        if isinstance(item, dict):
            normalized_similar.append({
                "ticket_id": _safe_text(item.get("ticket_id", ""), 50),
                "component": _safe_text(item.get("component", "General"), 60),
                "status": _safe_text(item.get("status", "Unknown"), 40),
                "summary": _safe_text(item.get("summary", ""), 220),
            })
        elif str(item).strip():
            normalized_similar.append({
                "ticket_id": "",
                "component": "General",
                "status": "Unknown",
                "summary": _safe_text(item, 220),
            })

    similar_resolutions = data.get("similar_ticket_resolutions", [])
    if isinstance(similar_resolutions, str):
        similar_resolutions = [similar_resolutions]

    normalized_resolutions = []
    for item in similar_resolutions[:3]:
        if isinstance(item, dict):
            normalized_resolutions.append({
                "ticket_id": _safe_text(item.get("ticket_id", ""), 50),
                "resolution": _safe_text(item.get("resolution", ""), 260),
            })
        elif str(item).strip():
            normalized_resolutions.append({
                "ticket_id": "",
                "resolution": _safe_text(item, 260),
            })

    kb_items = data.get("knowledge_base", [])
    if isinstance(kb_items, str):
        kb_items = [kb_items]

    normalized_kb = []
    for item in kb_items[:3]:
        if isinstance(item, dict):
            normalized_kb.append({
                "source": _safe_text(item.get("source", "KB"), 100),
                "guidance": _safe_text(item.get("guidance", ""), 320),
            })
        elif str(item).strip():
            normalized_kb.append({
                "source": "KB",
                "guidance": _safe_text(item, 320),
            })

    if not normalized_similar:
        for doc in similar_tickets[:3]:
            meta = doc.get("metadata", {})
            rec = _ticket_record_for_doc(doc)
            issue = _safe_text(rec.get("title_clean") or meta.get("title") or rec.get("title") or doc.get("text", ""), 160)
            normalized_similar.append({
                "ticket_id": _safe_text(meta.get("ticket_id") or rec.get("ticket_id") or doc.get("id"), 50),
                "component": _safe_text(meta.get("component") or rec.get("component") or "General", 60),
                "status": _safe_text(meta.get("status") or rec.get("status") or "Unknown", 40),
                "summary": issue or "Related historical ticket retrieved.",
            })

    if not normalized_resolutions:
        for doc in similar_tickets[:3]:
            meta = doc.get("metadata", {})
            rec = _ticket_record_for_doc(doc)
            resolution_text = _safe_text(rec.get("resolution_clean") or rec.get("resolution") or "No recorded resolution available.", 260)
            normalized_resolutions.append({
                "ticket_id": _safe_text(meta.get("ticket_id") or rec.get("ticket_id") or doc.get("id"), 50),
                "resolution": resolution_text,
            })

    if not normalized_kb:
        for doc in kb_results[:3]:
            normalized_kb.append({
                "source": _kb_source_label(doc),
                "guidance": _clean_kb_snippet(doc.get("text", ""), 320),
            })

    troubleshooting_steps = data.get("troubleshooting_steps", [])
    if isinstance(troubleshooting_steps, str):
        troubleshooting_steps = [troubleshooting_steps]
    normalized_steps = []
    for item in troubleshooting_steps[:5]:
        if isinstance(item, dict):
            normalized_steps.append({
                "step": _safe_text(item.get("step", ""), 80),
                "details": _safe_text(item.get("details", ""), 520),
            })
        elif str(item).strip():
            normalized_steps.append({
                "step": "Action",
                "details": _safe_text(item, 520),
            })

    if not normalized_steps:
        if kb_results:
            for idx, doc in enumerate(kb_results[:3], start=1):
                source = _kb_source_label(doc)
                snippet = _clean_kb_snippet(doc.get("text", ""), 520)
                normalized_steps.append({
                    "step": f"KB-guided step {idx}",
                    "details": (
                        f"Use guidance from {source}. Walk through the documented process carefully, including the menu path, "
                        f"configuration checks, and validation actions described in the KB. Focus on this relevant guidance: {snippet}"
                    ),
                })
        if not normalized_steps:
            for idx, doc in enumerate(similar_tickets[:3], start=1):
                meta = doc.get("metadata", {})
                rec = _ticket_record_for_doc(doc)
                issue = _safe_text(rec.get("title_clean") or meta.get("title") or doc.get("text", ""), 140)
                res = _safe_text(rec.get("resolution_clean") or rec.get("resolution") or "No recorded resolution available.", 260)
                normalized_steps.append({
                    "step": f"Historical ticket check {idx}",
                    "details": (
                        f"Compare the user's issue against ticket "
                        f"[{_safe_text(meta.get('ticket_id') or rec.get('ticket_id') or doc.get('id'), 50)}], which described "
                        f"'{issue}'. Use the historical handling as a guide for the next troubleshooting action. "
                        f"Recorded handling: {res}"
                    ),
                })

    escalate_if = data.get("escalate_if", [])
    if isinstance(escalate_if, str):
        escalate_if = [escalate_if]

    normalized_escalate = []
    for item in escalate_if[:3]:
        if isinstance(item, dict):
            normalized_escalate.append({
                "condition": _safe_text(item.get("condition", ""), 260),
                "owner": _safe_text(item.get("owner", "Tier 2 / appropriate support team"), 120),
            })
        elif str(item).strip():
            normalized_escalate.append({
                "condition": _safe_text(item, 260),
                "owner": "Tier 2 / appropriate support team",
            })

    if len(resolution) < 2:
        if kb_results:
            resolution = [
                "Use the retrieved knowledge base guidance as the primary troubleshooting path because it is the strongest documented evidence available for the reported symptom. Start by matching the user's environment and failure behavior to the KB scenario before making changes.",
                "Apply the KB-guided corrective steps in a controlled order, use similar historical tickets as supporting context where relevant, and verify the issue is fully resolved before closing the ticket. If the observed behavior differs from the documented evidence, capture the mismatch and escalate with those details."
            ]
        else:
            resolution = [
                "The ticket is partially supported by similar historical incidents, but there is limited direct knowledge base guidance for the exact wording provided by the user. Treat the initial response as a guided first-pass troubleshooting workflow rather than a confirmed root-cause resolution.",
                "Use the closest historical examples to guide the first troubleshooting pass, capture any missing environment details from the user, and validate each action before moving on. If the issue does not align with the retrieved evidence, document the gap clearly and escalate to the relevant team."
            ]

    if not normalized_escalate:
        normalized_escalate = [{
            "condition": "The issue remains unresolved after the documented steps, the user encounters a different error pattern than the retrieved evidence, or the required remediation falls outside standard Tier 1 handling.",
            "owner": "Escalate to the responsible Tier 2 team for the affected system, such as Network/VPN, Endpoint/Desktop Support, Printer Support, IT Infrastructure, or the application owner.",
        }]

    user_email = data.get("user_email", {})
    if isinstance(user_email, str):
        user_email = {"subject": "Update on your support ticket", "body": user_email}
    if not isinstance(user_email, dict):
        user_email = {}

    email_subject = _safe_text(
        user_email.get("subject") or f"Update on your support ticket: {problem[:60]}",
        120
    )

    email_body = _safe_text(user_email.get("body") or "", 2500)
    if not email_body:
        email_defaults = _default_email(problem, normalized_steps, normalized_escalate)
        email_subject = email_subject or email_defaults["subject"]
        email_body = email_defaults["body"]

    return {
        "problem": problem,
        "resolution": resolution[:2],
        "troubleshooting_steps": normalized_steps[:5],
        "similar_tickets": normalized_similar[:3],
        "similar_ticket_resolutions": normalized_resolutions[:3],
        "knowledge_base": normalized_kb[:3],
        "escalate_if": normalized_escalate[:3],
        "user_email": {
            "subject": email_subject,
            "body": email_body,
        },
    }


def _render_blueprint(data: Dict[str, Any]) -> str:
    lines = ["## Resolution Blueprint", "", "**Problem**", data["problem"], "", "**Resolution**"]
    for para in data["resolution"]:
        lines.append(para)
        lines.append("")

    if lines and lines[-1] == "":
        lines.pop()

    lines.extend(["", "**Troubleshooting Steps**"])
    for idx, item in enumerate(data.get("troubleshooting_steps", []), start=1):
        step = item.get("step") or f"Step {idx}"
        details = item.get("details") or "No details available."
        lines.append(f"{idx}. **{step}** — {details}")

    lines.extend(["", "**Similar Tickets**"])
    for item in data["similar_tickets"]:
        ticket_id = item.get("ticket_id") or "N/A"
        component = item.get("component") or "General"
        status = item.get("status") or "Unknown"
        summary = item.get("summary") or "No summary available."
        lines.append(f"- [{ticket_id}] ({component} · {status}): {summary}")

    lines.extend(["", "**Similar Tickets Resolution**"])
    for item in data["similar_ticket_resolutions"]:
        ticket_id = item.get("ticket_id") or "N/A"
        resolution = item.get("resolution") or "No recorded resolution available."
        lines.append(f"- [{ticket_id}]: {resolution}")

    lines.extend(["", "**Knowledge Base**"])
    for item in data["knowledge_base"]:
        source = item.get("source") or "KB"
        guidance = item.get("guidance") or "No KB guidance available."
        lines.append(f"- [{source}]: {guidance}")

    lines.extend(["", "**Escalate If**"])
    for rule in data["escalate_if"]:
        condition = rule.get("condition") if isinstance(rule, dict) else str(rule)
        owner = rule.get("owner") if isinstance(rule, dict) else "Tier 2 / appropriate support team"
        lines.append(f"- {condition} **Escalate to:** {owner}")

    email = data.get("user_email", {}) or {}
    lines.extend(["", "**User Email Draft**"])
    lines.append(f"Subject: {email.get('subject', 'Update on your support ticket')}")
    lines.append("")
    lines.append(email.get("body", ""))

    return "\n".join(lines).strip()


def _fallback_blueprint(desc: str, similar_tickets: List[Dict[str, Any]], kb_results: List[Dict[str, Any]]) -> str:
    normalized = _normalize_blueprint({}, similar_tickets, kb_results, desc)
    return _render_blueprint(normalized)


def _llm_generate_blueprint(prompt: str, system: str = "", temperature: float = 0.1, max_tokens: int = 900) -> str:
    try:
        resp = _ollama_session.post(
            OLLAMA_CHAT_URL,
            json={
                "model": LLM_MODEL,
                "stream": False,
                "think": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        if "message" in data and isinstance(data["message"], dict):
            return _strip_thinking_tokens(data["message"].get("content", ""))
        return _strip_thinking_tokens(data.get("response", ""))
    except Exception as e:
        return f"⚠️ LLM Error: {e}. Make sure Ollama is running."


def get_chat_rag_context(description: str) -> Dict[str, Any]:
    """
    RAG context for the ticketing chat (Model 1 + confidence heuristics).
    Importable from its_brain after the app module is loaded.
    """
    d = (description or "").strip()
    if not d:
        return {
            "ok": False,
            "similar_tickets": [],
            "kb_articles": [],
            "best_rerank": None,
            "retriever_error": "empty",
        }
    if not _load_retriever():
        return {
            "ok": False,
            "similar_tickets": [],
            "kb_articles": [],
            "best_rerank": None,
            "retriever_error": _retriever_error,
        }
    try:
        raw_sim = _ticket_retriever.search(  # type: ignore[union-attr]
            d[:2000], top_k=5, use_reranking=True
        )
        similar = _filter_ticket_results(raw_sim, 3)
        raw_kb = _kb_retriever.search(  # type: ignore[union-attr]
            d[:2000], top_k=5, use_reranking=True
        )
        kb = _filter_kb_results(raw_kb, 3)
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "similar_tickets": [],
            "kb_articles": [],
            "best_rerank": None,
            "retriever_error": str(e),
        }
    best = 0.0
    for doc in similar + kb:
        s = _score(doc)
        if s > best:
            best = s
    return {
        "ok": True,
        "similar_tickets": similar,
        "kb_articles": kb,
        "best_rerank": best,
    }


# ─── models ───────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    description: str
    top_k_tickets: int = 3
    top_k_kb: int = 3
    use_reranking: bool = True


# ─── routes ───────────────────────────────────────────────────────────────────
@app.get("/api/status")
def status():
    """Return system status — mirrors streamlit sidebar checks."""
    retriever_ready = _load_retriever()

    ticket_count = kb_count = None
    if retriever_ready:
        try:
            ticket_count = _ticket_retriever.collection.count()
            kb_count = _kb_retriever.collection.count()
        except Exception:
            pass

    llm_ready = False
    try:
        resp = _ollama_session.post(
            OLLAMA_GENERATE_URL,
            json={
                "model": LLM_MODEL,
                "prompt": "Say ready",
                "stream": False,
                "options": {"num_predict": 5},
                "think": False,
            },
            timeout=10,
        )
        llm_ready = resp.status_code == 200
    except Exception:
        pass

    return {
        "retriever_ready": retriever_ready,
        "retriever_error": _retriever_error,
        "ticket_count": ticket_count,
        "kb_count": kb_count,
        "llm_ready": llm_ready,
        "llm_model": LLM_MODEL,
    }


@app.post("/api/analyze")
def analyze(body: AnalyzeRequest):
    """
    Core endpoint — mirrors the 'Submit & Analyze' button logic in streamlit_app.py.
    Returns similar tickets, KB articles, resolution blueprint, and timing info.
    """
    desc = body.description.strip()
    if not desc:
        raise HTTPException(400, "Description required")

    if not _load_retriever():
        raise HTTPException(503, f"Retriever not loaded: {_retriever_error}")

    timings: Dict[str, float] = {}

    t0 = time.time()
    raw_similar = _ticket_retriever.search(
        desc, top_k=max(body.top_k_tickets, MAX_CONTEXT_TICKETS), use_reranking=body.use_reranking
    )
    similar_tickets = _filter_ticket_results(raw_similar, limit=max(1, body.top_k_tickets))
    t1 = time.time()
    timings["tickets_s"] = round(t1 - t0, 2)

    raw_kb = _kb_retriever.search(
        desc, top_k=max(body.top_k_kb, MAX_CONTEXT_KB), use_reranking=body.use_reranking
    )
    kb_results = _filter_kb_results(raw_kb, limit=max(1, body.top_k_kb))
    t2 = time.time()
    timings["kb_s"] = round(t2 - t1, 2)

    system = (
        "You are an IT helpdesk assistant writing a detailed, structured, and easy-to-follow resolution blueprint for ticket handlers. "
        "Never reveal chain-of-thought, hidden reasoning, analysis steps, or internal deliberation. "
        "Do not say phrases like 'we are given', 'let's think', 'but note', 'based on the context above', or similar. "
        "Use ONLY the retrieved evidence. If evidence is weak or missing, say that plainly. Return JSON only."
    )
    prompt = f"""Ticket:
{desc}

Relevant historical tickets:
{_format_ticket_context(similar_tickets)}

Relevant knowledge base:
{_format_kb_context(kb_results)}

Return ONLY valid JSON with this exact schema:
{{
  "problem": "1-2 sentence restatement of the user's ticket in clear, plain language",

  "resolution": [
    "Paragraph 1: Explain the issue and most likely cause using retrieved ticket and KB evidence in a clear, human-readable way.",
    "Paragraph 2: Explain the recommended resolution approach, what should be done, and how it helps resolve the issue."
  ],

  "troubleshooting_steps": [
    {{
      "step": "Short step title",
      "details": "Detailed, structured, easy-to-follow instructions for the ticket handler. Include where to click, what to check, what outcome to expect, and any manual fallback if needed."
    }},
    {{
      "step": "Short step title",
      "details": "Detailed, structured, easy-to-follow instructions for the ticket handler."
    }},
    {{
      "step": "Short step title",
      "details": "Detailed, structured, easy-to-follow instructions for the ticket handler."
    }},
    {{
      "step": "Verification",
      "details": "Explain how to confirm the issue is resolved and what the ticket handler should verify before closing."
    }}
  ],

  "similar_tickets": [
    {{
      "ticket_id": "ticket id",
      "component": "component",
      "status": "status",
      "summary": "One clear sentence describing what the similar issue was."
    }}
  ],

  "similar_ticket_resolutions": [
    {{
      "ticket_id": "ticket id",
      "resolution": "One clear sentence describing how the issue was resolved; if unavailable, explicitly say no recorded resolution available."
    }}
  ],

  "knowledge_base": [
    {{
      "source": "kb file name",
      "guidance": "Convert KB content into clear, human-readable instructions. Do NOT copy raw markdown, headers, or long chunks directly."
    }}
  ],

  "escalate_if": [
    {{
      "condition": "Clear condition when escalation is required.",
      "owner": "Who should handle it, such as Network Team, Desktop Support, IT Infrastructure, Printer Support, or Vendor Support."
    }}
  ],

  "user_email": {{
    "subject": "Short subject summarizing the issue",
    "body": "A complete, professional, user-facing email. It must include a greeting, a brief explanation of the issue, and the FULL troubleshooting steps written clearly inside the email itself. Do NOT say 'as listed above' or 'refer to the steps above'. The email must be fully self-contained and easy for the user to follow."
  }}
}}

Rules:
- Keep the final blueprint detailed, structured, and easy for a ticket handler to act on.
- Resolution MUST contain exactly 2 paragraphs.
- Troubleshooting steps MUST be detailed, structured, and user-friendly, not one-line bullets.
- Each troubleshooting step should explain what to do, where to do it, and why it matters.
- If a KB article is available, use it as the primary source of troubleshooting steps.
- Use similar tickets to strengthen the recommendation and add practical context.
- Maximum 5 troubleshooting_steps, 3 similar_tickets, 3 similar_ticket_resolutions, and 3 knowledge_base items.
- Always prefer KB-backed guidance when available.
- Do NOT copy raw KB text or markdown directly. Rewrite it into clean instructions.
- If KB is weak or only partially relevant, explicitly say it is a partial match.
- If a similar ticket has no resolution, explicitly state "No recorded resolution available."
- Do NOT invent unsupported fixes, policies, versions, teams, or commands not grounded in the retrieved evidence.
- Escalation must include BOTH condition and owner.
- The email MUST be fully self-contained and include the actual troubleshooting steps in full detail.
- Do NOT output markdown, explanations, commentary, or reasoning.
- Output JSON only.
"""

    raw_resolution = _llm_generate_blueprint(prompt, system=system)
    parsed = _extract_json_object(raw_resolution)
    if parsed is not None:
        resolution = _render_blueprint(_normalize_blueprint(parsed, similar_tickets, kb_results, desc))
    else:
        resolution = _fallback_blueprint(desc, similar_tickets, kb_results)

    t3 = time.time()
    timings["llm_s"] = round(t3 - t2, 2)
    timings["total_s"] = round(t3 - t0, 1)

    def clean_docs(docs: List[Dict[str, Any]], max_text: int = 300) -> List[Dict[str, Any]]:
        out = []
        for d in docs:
            meta = d.get("metadata", {})
            rec = _ticket_record_for_doc(d)
            score = _score(d)
            out.append({
                "id": d.get("id", ""),
                "text": _safe_text(d.get("text", ""), max_text),
                "score": round(float(score), 3),
                "title": meta.get("title") or rec.get("title_clean") or rec.get("title") or "N/A",
                "category": meta.get("category") or rec.get("category") or "N/A",
                "severity": meta.get("severity") or rec.get("severity") or "N/A",
                "status": meta.get("status") or rec.get("status") or "N/A",
                "component": meta.get("component") or rec.get("component") or "N/A",
                "ticket_id": meta.get("ticket_id") or rec.get("ticket_id") or d.get("id", ""),
                "source_file": meta.get("source_file") or "N/A",
                "resolution": _safe_text(rec.get("resolution_clean") or rec.get("resolution") or "", 220),
            })
        return out

    return {
        "resolution": resolution,
        "similar_tickets": clean_docs(similar_tickets),
        "kb_articles": clean_docs(kb_results, max_text=400),
        "timings": timings,
        "counts": {
            "tickets": len(similar_tickets),
            "kb": len(kb_results),
        }
    }


def _tokenize_for_bm25_api(text: str) -> List[str]:
    import re as _re
    if not isinstance(text, str):
        return []
    tokens = _re.findall(r'[a-z0-9]+', text.lower())
    stopwords = {
        "the","a","an","is","are","was","were","be","been","being","have","has","had",
        "do","does","did","will","would","could","should","may","might","shall","can",
        "to","of","in","for","on","with","at","by","from","as","into","through",
        "during","before","after","and","but","or","nor","not","so","yet","both",
        "either","neither","each","every","all","any","few","more","most","other",
        "some","such","no","only","own","same","than","too","very","just","because",
        "if","when","that","this","these","those","it","its","i","me","my","we","our",
        "you","your","he","him","his","she","her","they","them","their","what",
        "which","who","whom",
    }
    return [t for t in tokens if t not in stopwords and len(t) > 1]


def _add_ticket_to_retriever(
    unified_id: str,
    embedding_text: str,
    chroma_metadata: Dict[str, Any],
    csv_record: Dict[str, Any],
) -> None:
    """Embed a new ticket and insert it into ChromaDB + BM25 + lookup + CSV."""
    global _ticket_lookup

    # Ensure retriever is loaded (may not be if caller skipped analyze())
    if _ticket_retriever is None:
        _load_retriever()
    if _ticket_retriever is None:
        return

    # 1. Embed and upsert into ChromaDB (upsert avoids duplicate-ID errors on retry)
    embedding = _ticket_retriever.embedder.encode(embedding_text).tolist()
    try:
        _ticket_retriever.collection.upsert(
            ids=[unified_id],
            embeddings=[embedding],
            documents=[embedding_text],
            metadatas=[chroma_metadata],
        )
    except Exception:
        pass

    # 2. Append to in-memory BM25 state and rebuild index
    tokens = _tokenize_for_bm25_api(embedding_text)
    _ticket_retriever.bm25_corpus.append(tokens)
    _ticket_retriever.bm25_ids.append(unified_id)
    _ticket_retriever.bm25_documents.append(embedding_text)

    from rank_bm25 import BM25Okapi
    _ticket_retriever.bm25 = BM25Okapi(_ticket_retriever.bm25_corpus)

    # 3. Persist updated BM25 corpus to disk in a background thread so the
    #    API response is not blocked by a potentially large JSON write.
    _corpus_snapshot = {
        "corpus": list(_ticket_retriever.bm25_corpus),
        "ids": list(_ticket_retriever.bm25_ids),
        "documents": list(_ticket_retriever.bm25_documents),
    }
    bm25_path = BASE_DIR / "data" / "processed" / "bm25_corpus_tickets.json"

    def _dump() -> None:
        try:
            with open(bm25_path, "w") as _f:
                json.dump(_corpus_snapshot, _f)
        except Exception:
            pass

    threading.Thread(target=_dump, daemon=True).start()

    # 4. Update in-memory ticket lookup
    if _ticket_lookup is not None:
        _ticket_lookup[unified_id] = csv_record
        tid = str(csv_record.get("ticket_id", "")).strip()
        if tid and tid != unified_id:
            _ticket_lookup[tid] = csv_record

    # 5. Append row to all_tickets.csv so future restarts pick it up
    csv_path = BASE_DIR / "data" / "processed" / "all_tickets.csv"
    if csv_path.exists():
        try:
            existing_cols = list(pd.read_csv(csv_path, nrows=0).columns)
            row_df = pd.DataFrame([csv_record])
            for col in existing_cols:
                if col not in row_df.columns:
                    row_df[col] = ""
            row_df[existing_cols].to_csv(csv_path, mode="a", header=False, index=False)
        except Exception:
            pd.DataFrame([csv_record]).to_csv(csv_path, mode="a", header=False, index=False)


class AdminCreateTicketRequest(BaseModel):
    issue_description: str
    department: str = ""
    resolution: str = ""


@app.post("/api/admin/create-ticket")
def admin_create_ticket_api(
    body: AdminCreateTicketRequest,
    authorization: str | None = Header(None, alias="Authorization"),
):
    from its_routes import _decode_token
    from its_db import get_db as _get_db
    from its_models import SupportSession, Ticket, Message
    from its_brain import run_model2_details, public_ticket_id
    import uuid as _uuid

    desc = (body.issue_description or "").strip()
    if len(desc) < 10:
        raise HTTPException(400, "issue_description must be at least 10 characters")

    # Auth: require admin token
    admin_user_id = 2
    if authorization:
        p = _decode_token(authorization)
        if p and p.get("role") == "admin":
            admin_user_id = int(p.get("user_id", 2))
        else:
            raise HTTPException(403, "Admin only")
    else:
        raise HTTPException(401, "Authorization header required")

    if (body.department or "").strip():
        dept = body.department.strip()
        triage = {
            "department": dept,
            "confidence": 1.0,
            "reason": "Department was selected by the admin during ticket creation.",
            "scores": {},
            "source": "admin",
        }
    else:
        triage = run_model2_details(desc)
        dept = str(triage.get("department") or "IT Infrastructure & Platform")
    dept_conf = float(triage.get("confidence") or 0.0)
    routing_reason = str(triage.get("reason") or "")
    triage_scores = triage.get("scores") or {}

    # Always run analyze() — we need similar_tickets + KB for insights caching.
    # If admin supplied a custom resolution we use that text but still run RAG
    # so the insights panel is fully populated on first open.
    rag = analyze(AnalyzeRequest(description=desc, top_k_tickets=4, top_k_kb=3, use_reranking=True))
    custom_resolution = (body.resolution or "").strip()
    resolution_text = custom_resolution if custom_resolution else rag.get("resolution", "")

    with next(_get_db()) as db:
        sid = str(_uuid.uuid4())
        session_row = SupportSession(id=sid, user_id=admin_user_id, state="escalated", ai_rounds=0)
        db.add(session_row)
        db.flush()

        pub = public_ticket_id()
        ticket = Ticket(
            public_id=pub,
            user_id=admin_user_id,
            session_id=sid,
            issue_summary=desc[:2000],
            department=dept,
            department_confidence=dept_conf,
            routing_reason=routing_reason,
            triage_scores=json.dumps(triage_scores),
            status="resolved",
        )
        db.add(ticket)
        db.add(Message(session_id=sid, sender="system",
                       content=f"Admin-created: {pub}\n\nResolution:\n{resolution_text[:1000]}"))
        db.commit()
        db.refresh(ticket)
        ticket_id_db = ticket.id

        # Build and cache insights immediately so the queue open is instant.
        insights_payload = {
            "ticket_id": ticket_id_db,
            "public_id": pub,
            "user_id": admin_user_id,
            "resolution": resolution_text,
            "similar_tickets": rag.get("similar_tickets", []),
            "kb_articles": rag.get("kb_articles", []),
            "timings": rag.get("timings", {}),
        }
        ticket.insights_cache = json.dumps(insights_payload)
        db.commit()

    unified_id = f"ADMIN-{ticket_id_db}"
    title_text = desc[:200]
    embedding_text = (
        f"Title: {title_text}\n"
        f"Description: {desc}\n"
        f"Category: IT Support\n"
        f"Component: {dept}\n"
        f"Severity: Medium\n"
        f"Resolution: {resolution_text[:500]}"
    )
    chroma_metadata: Dict[str, Any] = {
        "ticket_id": unified_id,
        "title": title_text,
        "category": "IT Support",
        "component": dept,
        "department": dept,
        "severity": "Medium",
        "status": "resolved",
        "source": "admin",
        "quality_score": 1.0,
    }
    csv_record: Dict[str, Any] = {
        "ticket_id": unified_id,
        "title": title_text,
        "title_clean": title_text,
        "description": desc,
        "description_clean": desc,
        "category": "IT Support",
        "component": dept,
        "assignment_group": "",
        "raw_assignment_group": "",
        "raw_department": dept,
        "department": dept,
        "mapped_department": dept,
        "issue_type": "Admin",
        "severity": "Medium",
        "status": "resolved",
        "resolution": resolution_text,
        "resolution_clean": resolution_text[:2000],
        "error_codes": "[]",
        "quality_score": 1.0,
        "embedding_text": embedding_text,
        "source": "admin",
        "unified_id": unified_id,
    }
    _add_ticket_to_retriever(unified_id, embedding_text, chroma_metadata, csv_record)

    return {
        "ticket": {
            "id": ticket_id_db,
            "public_id": pub,
            "department": dept,
            "department_confidence": dept_conf,
            "routing_reason": routing_reason,
            "triage_scores": triage_scores,
            "status": "resolved",
            "resolution": resolution_text,
        },
        "insights": insights_payload,
    }


@app.on_event("startup")
def _its_orm_startup() -> None:
    from its_db import init_db  # local import: keeps `api` importable before DB exists

    init_db()


# Intelligent ticketing: chat, sessions, admin (SQLite, user↔ticket mapping)
from its_routes import router as its_ticketing_router  # noqa: E402

app.include_router(its_ticketing_router)
