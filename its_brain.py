"""
Model 1 (resolution + confidence) and Model 2 (department) plus escalation heuristics.
"""
from __future__ import annotations

import json
import re
import secrets
from typing import Any, Dict, List, Optional

from department_mapping import DEPARTMENTS, classify_department

CONFIDENCE_THRESHOLD = 0.5
MAX_AI_ATTEMPTS = 2

# Keyword / phrase list for "still broken" and "connect" intent
ESCALATION_PHRASES: List[str] = [
    "not working",
    "still not fixed",
    "doesn't work",
    "doesnt work",
    "not fixed",
    "still broken",
    "connect to specialist",
    "connect to support",
    "customer support",
    "talk to a human",
    "speak to someone",
    "escalate",
]

CONNECT_PHRASES: List[str] = [
    "connect to specialist",
    "connect to support",
    "connect specialist",
    "customer support",
    "human support",
    "open a ticket",
    "create a ticket",
    "i want a ticket",
]


def _ollama_json(system: str, user: str, timeout: int = 120) -> str:
    import requests
    from api import LLM_MODEL, OLLAMA_CHAT_URL

    try:
        r = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": LLM_MODEL,
                "stream": False,
                "think": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "options": {"temperature": 0.1, "num_predict": 500},
            },
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        if "message" in data and isinstance(data["message"], dict):
            return str(data["message"].get("content", ""))
        return str(data.get("response", ""))
    except Exception as e:  # noqa: BLE001
        return f'{{"error": "{e!s}"}}'


def _strip_code_fences(text: str) -> str:
    t = re.sub(r"^```(?:json)?\s*", "", (text or "").strip(), flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    return t


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    t = _strip_code_fences(text)
    t = re.sub(r"<[^>]+>", "", t)
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", t, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _rag_summary_for_prompt(ctx: Dict[str, Any]) -> str:
    if not ctx.get("ok"):
        return "No RAG retriever: answer from general IT knowledge and keep confidence under 0.5 if uncertain."
    parts: List[str] = []
    for d in (ctx.get("similar_tickets") or [])[:3]:
        parts.append(
            "- similar ticket "
            f"{str((d or {}).get('ticket_id', 'unknown'))}: "
            f"score={str((d or {}).get('score', ''))}; "
            f"{str((d or {}).get('text', ''))[:360]}"
        )
    for d in (ctx.get("kb_articles") or [])[:2]:
        parts.append(
            f"- kb {str((d or {}).get('title', 'article'))}: "
            f"{str((d or {}).get('text', ''))[:280]}"
        )
    return "\n".join(parts) if parts else "Weak retrieval: say so and lower confidence."


def _strip_thinking(text: str) -> str:
    if not text:
        return ""
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return t.strip()


def _clean_steps(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    steps: List[str] = []
    for item in value[:5]:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        if text:
            steps.append(text[:350])
    return steps


def _format_model1_reply(reply: str, steps: List[str], escalation_recommended: bool) -> str:
    reply = re.sub(r"\s+", " ", (reply or "").strip())
    if not reply:
        reply = "I found a few relevant support records. Let's try the safest steps first."

    out = [reply]
    if steps:
        out.append("\nRecommended actions:")
        out.extend(f"{idx}. {step}" for idx, step in enumerate(steps, start=1))
    if escalation_recommended:
        out.append(
            "\nIf these steps do not work, choose **Connect to specialist** and I will route this with the chat summary."
        )
    return "\n".join(out).strip()


def run_model1(user_message: str, attempt_number: int = 1) -> Dict[str, Any]:
    """
    Returns a structured RAG-grounded conversational answer:
      {
        "response": formatted chat text,
        "reply": conversational intro,
        "recommended_steps": [str],
        "confidence": 0..1,
        "escalation_recommended": bool,
        "ok": bool
      }
    """
    from api import get_chat_rag_context, _llm_generate_blueprint

    ctx = get_chat_rag_context((user_message or "").strip()[:2000])
    block = _rag_summary_for_prompt(ctx)
    br = ctx.get("best_rerank")
    if br is not None and isinstance(br, (int, float)):
        heuristic = max(0.0, min(1.0, 0.35 + 0.2 * max(float(br), 0.0)))
    else:
        heuristic = 0.48

    final_attempt = attempt_number >= MAX_AI_ATTEMPTS
    system = (
        "You are a Tier-1 IT helpdesk assistant inside a RAG system. "
        "Use ONLY the retrieved evidence when it is relevant; do not pretend certainty. "
        "Talk naturally like a support agent, but return JSON ONLY, no markdown fences. "
        "Schema: {"
        "\"reply\": string, "
        "\"recommended_steps\": string[], "
        "\"confidence\": number, "
        "\"escalation_recommended\": boolean, "
        "\"escalation_reason\": string"
        "}. "
        "recommended_steps must contain 2 to 5 short, safe, user-actionable steps. "
        "confidence is 0..1 and must be below 0.5 when evidence is weak. "
        "If this is the second attempt or the issue sounds specialist-level, set escalation_recommended=true."
    )
    user = (
        f"Attempt number: {attempt_number} of {MAX_AI_ATTEMPTS}\n"
        f"Final attempt: {final_attempt}\n\n"
        f"User issue:\n{user_message}\n\n"
        f"Retrieved evidence:\n{block}\n"
    )
    raw = _ollama_json(system, user)
    raw = _strip_thinking(raw)
    parsed = _parse_json_object(raw)
    if parsed and ("reply" in parsed or "response" in parsed):
        try:
            c = float(parsed.get("confidence", heuristic))
        except (TypeError, ValueError):
            c = heuristic
        c = max(0.0, min(1.0, c))
        if ctx.get("ok") is not True:
            c = min(c, 0.6)
        steps = _clean_steps(parsed.get("recommended_steps"))
        escalation_recommended = bool(parsed.get("escalation_recommended", False)) or final_attempt
        reply = str(parsed.get("reply") or parsed.get("response") or "")
        return {
            "response": _format_model1_reply(reply, steps, escalation_recommended)[:4000],
            "reply": reply[:1500],
            "recommended_steps": steps,
            "confidence": c,
            "escalation_recommended": escalation_recommended,
            "escalation_reason": str(parsed.get("escalation_reason", ""))[:500],
            "retrieval_score": br,
            "ok": True,
        }

    # fallback: short help text via shared LLM string generator
    try:
        short = _llm_generate_blueprint(
            f"Attempt {attempt_number} of {MAX_AI_ATTEMPTS}\n"
            f"User: {user_message}\n\nContext:\n{block}\n"
            f"Write a short helpful reply followed by 2 numbered troubleshooting steps.",
            system="You are a concise helpdesk agent. Be practical and honest about uncertainty.",
            max_tokens=400,
        )
        if short and "LLM Error" not in short:
            conf = min(0.9, max(0.4, float(heuristic)))
            return {
                "response": _strip_thinking(short)[:3000],
                "reply": _strip_thinking(short)[:1500],
                "recommended_steps": [],
                "confidence": conf,
                "escalation_recommended": attempt_number >= MAX_AI_ATTEMPTS,
                "escalation_reason": "Fallback response generated without structured JSON.",
                "retrieval_score": br,
                "ok": True,
            }
    except Exception:  # noqa: BLE001
        pass
    return {
        "response": (
            "I could not run the AI service. Please choose **Connect to specialist** to open a ticket, "
            "or ensure Ollama is running and try again."
        ),
        "reply": "",
        "recommended_steps": [],
        "confidence": 0.2,
        "escalation_recommended": True,
        "escalation_reason": "AI service unavailable.",
        "retrieval_score": br,
        "ok": False,
    }


def run_model2_details(issue_summary: str) -> Dict[str, Any]:
    """Classify the ticket into one of the 4 departments with explainability."""
    rule_result = classify_department(issue_summary)
    rule_conf = float(rule_result.get("confidence", 0.0))
    if rule_conf >= 0.72:
        return rule_result

    system = (
        "You are a deterministic IT ticket triage model. Return JSON ONLY. "
        "Schema: {\"department\": exact string, \"confidence\": number 0..1, \"reason\": string}. "
        "Pick exactly one department from the provided list. Base the decision on the full conversation."
    )
    choices = "\n".join(f"- {d}" for d in DEPARTMENTS)
    user = (
        f"Conversation / issue summary:\n{issue_summary[:3000]}\n\n"
        f"Departments:\n{choices}\n\n"
        f"Rule-based pre-classification:\n{json.dumps(rule_result, indent=2)}\n"
    )
    raw = _ollama_json(system, user, timeout=40)
    raw = _strip_thinking(raw)
    parsed = _parse_json_object(raw)
    if parsed:
        d = str(parsed.get("department", "")).strip()
        for dep in DEPARTMENTS:
            if d.lower() == dep.lower() or d in dep or dep in d:
                try:
                    conf = float(parsed.get("confidence", rule_conf))
                except (TypeError, ValueError):
                    conf = rule_conf
                return {
                    "department": dep,
                    "confidence": max(0.0, min(1.0, conf)),
                    "reason": str(parsed.get("reason", ""))[:700]
                    or str(rule_result.get("reason", "")),
                    "scores": rule_result.get("scores", {}),
                    "source": "llm",
                }
    return rule_result


def run_model2(issue_summary: str) -> str:
    """Backward-compatible helper that returns only the department string."""
    return str(run_model2_details(issue_summary).get("department", DEPARTMENTS[0]))


def public_ticket_id() -> str:
    return f"ITS-{secrets.token_hex(3).upper()}"


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().strip())


def text_implies_escalation(text: str) -> bool:
    n = _normalize(text)
    for p in ESCALATION_PHRASES:
        if p in n:
            return True
    return False


def text_implies_connect_only(text: str) -> bool:
    n = _normalize(text)
    for p in CONNECT_PHRASES:
        if p in n:
            return True
    return False


def summarize_conversation(turns: List[str], max_len: int = 600) -> str:
    blob = " ".join(turns)[-3000:]
    if len(blob) <= max_len:
        return blob
    return "…" + blob[-(max_len - 1) :]
