"""
REST API for user chat, ticket escalation, and admin views.
"""
from __future__ import annotations

import base64
import json
import re
import uuid
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from its_brain import (
    CONFIDENCE_THRESHOLD,
    MAX_AI_ATTEMPTS,
    DEPARTMENTS,
    public_ticket_id,
    run_model1,
    run_model2_details,
    summarize_conversation,
    text_implies_connect_only,
    text_implies_escalation,
)
from its_db import get_db
from its_models import Message, SupportSession, Ticket, User

router = APIRouter()


# ─── auth (demo) ─────────────────────────────────────────────────────────────


class LoginIn(BaseModel):
    email: str
    password: str = "demo"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    email: str
    name: str
    role: str
    display_email: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


def _encode_token(user_id: int, role: str) -> str:
    payload = json.dumps({"user_id": user_id, "role": role}).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def _decode_token(authorization: Optional[str]) -> dict[str, Any] | None:
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    raw = authorization[7:].strip()
    pad = (4 - len(raw) % 4) % 4
    raw += "=" * pad
    try:
        return json.loads(base64.urlsafe_b64decode(raw).decode())
    except Exception:  # noqa: BLE001
        return None


def get_user(
    authorization: str | None = Header(None, alias="Authorization"),
    db: Session = Depends(get_db),
) -> User:
    p = _decode_token(authorization)
    if not p or "user_id" not in p:
        raise HTTPException(401, "Not authenticated. Login at /api/auth/login (demo password: demo).")
    u = db.get(User, int(p["user_id"]))
    if not u:
        raise HTTPException(401, "User not found")
    return u


def get_admin(
    u: User = Depends(get_user),
) -> User:
    if (u.role or "") != "admin":
        raise HTTPException(403, "Admin only")
    return u


@router.post("/api/auth/login", response_model=TokenOut)
def login(body: LoginIn, db: Session = Depends(get_db)) -> Any:
    em = (body.email or "").strip().lower()
    if body.password != "demo":
        raise HTTPException(400, "Use demo password: demo")
    u = db.execute(select(User).where(User.email == em)).scalar_one_or_none()
    if not u:
        raise HTTPException(400, f"No user {em} — use user@demo.com or admin@demo.com")
    return {
        "access_token": _encode_token(u.id, u.role or "user"),
        "user": u,
    }


# ─── sessions & chat ─────────────────────────────────────────────────────────


def _message_to_dict(m: Message) -> dict[str, Any]:
    return {
        "id": m.id,
        "sender": m.sender,
        "content": m.content,
        "created_at": m.created_at.isoformat() if m.created_at else None,
    }


@router.post("/api/sessions")
def create_session(
    u: User = Depends(get_user),
    db: Session = Depends(get_db),
) -> Any:
    sid = str(uuid.uuid4())
    row = SupportSession(
        id=sid,
        user_id=u.id,
        state="active",
        ai_rounds=0,
        needs_escalation_choice=0,
    )
    db.add(row)
    db.commit()
    return {"session_id": sid, "user_id": u.id}


@router.get("/api/sessions/{session_id}")
def get_session(
    session_id: str,
    u: User = Depends(get_user),
    db: Session = Depends(get_db),
) -> Any:
    s = _load_session_for_user(db, session_id, u.id)
    msgs = _messages_ordered(db, session_id)
    ticket = db.execute(
        select(Ticket).where(Ticket.session_id == s.id)
    ).scalar_one_or_none()
    return {
        "session_id": s.id,
        "user_id": s.user_id,
        "state": s.state,
        "ai_rounds": s.ai_rounds,
        "needs_escalation_choice": bool(s.needs_escalation_choice),
        "departments": DEPARTMENTS,
        "messages": [_message_to_dict(m) for m in msgs],
        "ticket": None
        if not ticket
        else {
            "id": ticket.id,
            "public_id": ticket.public_id,
            "user_id": ticket.user_id,
            "issue_summary": ticket.issue_summary,
            "department": ticket.department,
            "department_confidence": ticket.department_confidence,
            "routing_reason": ticket.routing_reason,
            "triage_scores": json.loads(ticket.triage_scores or "{}"),
            "status": ticket.status,
            "created_at": ticket.created_at.isoformat() if ticket.created_at else None,
        },
    }


def _load_session_for_user(db: Session, session_id: str, user_id: int) -> SupportSession:
    s = db.get(SupportSession, session_id)
    if not s or s.user_id != user_id:
        raise HTTPException(404, "Session not found")
    return s


def _messages_ordered(db: Session, session_id: str) -> List[Message]:
    q = select(Message).where(Message.session_id == session_id).order_by(Message.id.asc())
    return list(db.scalars(q).all())


class PostMessageIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)


@router.post("/api/sessions/{session_id}/messages")
def post_message(
    session_id: str,
    body: PostMessageIn,
    u: User = Depends(get_user),
    db: Session = Depends(get_db),
) -> Any:
    s = _load_session_for_user(db, session_id, u.id)
    if s.state == "escalated":
        raise HTTPException(400, "This session is closed after ticket creation.")
    text = body.text.strip()
    if s.needs_escalation_choice:
        raise HTTPException(
            400,
            "Use POST /api/sessions/{id}/action with {action: 'retry'|'escalate'}.",
        )

    # Max AI turns already used — user message just opens the two-button choice
    if int(s.ai_rounds or 0) >= MAX_AI_ATTEMPTS:
        db.add(Message(session_id=session_id, sender="user", content=text))
        s.needs_escalation_choice = 1
        db.add(
            Message(
                session_id=session_id,
                sender="system",
                content="You've reached the limit of **two** AI fix attempts. "
                "Use **Try AI again** (resets) or **Connect to specialist** to open a ticket.",
            )
        )
        db.commit()
        msgs = _messages_ordered(db, session_id)
        return {
            "session_id": session_id,
            "messages": [_message_to_dict(m) for m in msgs],
            "needs_escalation_choice": True,
            "last_model": None,
        }

    # Save the user's message immediately so the DB write lock is released
    # before we make the (potentially slow) Ollama call.
    db.add(Message(session_id=session_id, sender="user", content=text))
    if text_implies_connect_only(text) and (int(s.ai_rounds or 0) > 0 or len(text) < 100):
        s.needs_escalation_choice = 1
        db.add(
            Message(
                session_id=session_id,
                sender="system",
                content="Choose **Try AI again** or **Connect to specialist** to continue.",
            )
        )
        db.commit()
        msgs = _messages_ordered(db, session_id)
        return {
            "session_id": session_id,
            "messages": [_message_to_dict(m) for m in msgs],
            "needs_escalation_choice": True,
            "suggest_specialist": True,
            "last_model": None,
        }
    db.commit()

    current_round = int(s.ai_rounds or 0)
    if text_implies_escalation(text) and current_round >= MAX_AI_ATTEMPTS:
        s.needs_escalation_choice = 1
        db.add(
            Message(
                session_id=session_id,
                sender="system",
                content="It sounds like the previous steps did not work. Choose **Connect to specialist** and I will route this with the chat summary.",
            )
        )
        db.commit()
        msgs = _messages_ordered(db, session_id)
        return {
            "session_id": session_id,
            "messages": [_message_to_dict(m) for m in msgs],
            "needs_escalation_choice": True,
            "suggest_specialist": True,
            "last_model": None,
        }

    # No write lock is held across the Ollama call now.
    m1 = run_model1(text, attempt_number=current_round + 1)
    conf = float(m1.get("confidence", 0.0))
    response_text = str(m1.get("response", ""))

    # Re-open the session row for the AI reply + counter bump.
    s2 = _load_session_for_user(db, session_id, u.id)
    s2.ai_rounds = int(s2.ai_rounds or 0) + 1
    model_recommends_escalation = bool(m1.get("escalation_recommended"))
    if conf < CONFIDENCE_THRESHOLD and s2.ai_rounds < MAX_AI_ATTEMPTS:
        response_text = (
            response_text
            + "\n\n----\n"
            + "I'm not very confident in this fix yet (confidence: "
            + f"{conf:.0%}, threshold {CONFIDENCE_THRESHOLD:.0%}). "
            "Add more detail for another try, or connect to a specialist when ready."
        )
    elif conf < CONFIDENCE_THRESHOLD and s2.ai_rounds >= MAX_AI_ATTEMPTS:
        response_text = (
            response_text
            + f"\n\n(Confidence: {conf:.0%} — this was your last free AI round before choosing.)"
        )

    db.add(Message(session_id=session_id, sender="ai", content=response_text))
    if s2.ai_rounds >= MAX_AI_ATTEMPTS:
        s2.needs_escalation_choice = 1
        if "Connect to specialist" not in response_text:
            response_text += (
                "\n\nYou've reached the limit of **two** AI fix attempts. "
                "If the issue is still happening, choose **Connect to specialist**."
            )
    db.commit()
    msgs = _messages_ordered(db, session_id)
    return {
        "session_id": session_id,
        "messages": [_message_to_dict(m) for m in msgs],
        "last_model": {
            "confidence": conf,
            "ok": m1.get("ok"),
            "recommended_steps": m1.get("recommended_steps", []),
            "escalation_recommended": model_recommends_escalation,
            "retrieval_score": m1.get("retrieval_score"),
        },
        "suggest_specialist": conf < CONFIDENCE_THRESHOLD or model_recommends_escalation,
        "needs_escalation_choice": bool(s2.needs_escalation_choice),
    }


class ActionIn(BaseModel):
    action: str  # "retry" | "escalate"


@router.post("/api/sessions/{session_id}/action")
def post_action(
    session_id: str,
    body: ActionIn,
    u: User = Depends(get_user),
    db: Session = Depends(get_db),
) -> Any:
    s = _load_session_for_user(db, session_id, u.id)
    if body.action not in ("retry", "escalate"):
        raise HTTPException(400, "action must be 'retry' or 'escalate'")
    if s.state == "escalated":
        raise HTTPException(400, "Already escalated")

    if body.action == "retry":
        s.ai_rounds = 0
        s.needs_escalation_choice = 0
        db.add(
            Message(
                session_id=session_id,
                sender="system",
                content="Alright — we'll try the AI again from here. Describe what happens when you try the last steps.",
            )
        )
        db.commit()
        return {"ok": True, "session_id": session_id, "reset": True}

    # escalate
    msgs = _messages_ordered(db, session_id)
    user_texts = [m.content for m in msgs if m.sender == "user"]
    transcript = "\n".join(
        f"{m.sender}: {m.content}" for m in msgs if m.sender in {"user", "ai", "system"}
    )
    issue_summary = summarize_conversation(user_texts) or (user_texts[-1] if user_texts else "Support request")
    # Clean for storage
    issue_summary = re.sub(r"#{2,}\s*|\*\*", " ", issue_summary)[:2000].strip()
    if not issue_summary:
        issue_summary = "User requested specialist assistance."

    triage = run_model2_details(transcript or issue_summary)
    dept = str(triage.get("department") or DEPARTMENTS[0])
    dept_conf = float(triage.get("confidence") or 0.0)
    routing_reason = str(triage.get("reason") or "")
    triage_scores = triage.get("scores") or {}
    pub = public_ticket_id()
    t = Ticket(
        public_id=pub,
        user_id=u.id,
        session_id=session_id,
        issue_summary=issue_summary,
        department=dept,
        department_confidence=dept_conf,
        routing_reason=routing_reason,
        triage_scores=json.dumps(triage_scores),
        status="open",
    )
    db.add(t)
    s.state = "escalated"
    s.needs_escalation_choice = 0
    db.add(
        Message(
            session_id=session_id,
            sender="system",
            content=(
                f"**Ticket created:** `{pub}`\n"
                f"**Summary:** {issue_summary}\n"
                f"**Routed to:** {dept} ({dept_conf:.0%} confidence)\n"
                f"**Routing note:** {routing_reason or 'Classified from the support conversation.'}\n"
                f"(Your `user_id` is **{u.id}** — our team will use this in follow-up.)\n\n"
                "This chat session is complete. Thank you!"
            ),
        )
    )
    db.commit()
    db.refresh(t)
    return {
        "ok": True,
        "ticket": {
            "id": t.id,
            "public_id": t.public_id,
            "user_id": t.user_id,
            "issue_summary": t.issue_summary,
            "department": t.department,
            "department_confidence": t.department_confidence,
            "routing_reason": t.routing_reason,
            "triage_scores": json.loads(t.triage_scores or "{}"),
            "status": t.status,
            "session_id": session_id,
        },
    }


# ─── admin ────────────────────────────────────────────────────────────────────


@router.get("/api/admin/tickets")
def admin_tickets(
    u: User = Depends(get_admin),
    db: Session = Depends(get_db),
) -> Any:
    st = (
        select(Ticket)
        .order_by(Ticket.id.desc())
        .options(selectinload(Ticket.user))
    )
    rows = list(db.scalars(st).all())
    return {
        "tickets": [
            {
                "id": t.id,
                "public_id": t.public_id,
                "user_id": t.user_id,
                "user_email": t.user.display_email or t.user.email,
                "department": t.department,
                "department_confidence": t.department_confidence,
                "routing_reason": t.routing_reason,
                "triage_scores": json.loads(t.triage_scores or "{}"),
                "status": t.status,
                "created_at": t.created_at.isoformat() if t.created_at else None,
                "issue_excerpt": (t.issue_summary or "")[:120],
            }
            for t in rows
        ]
    }


@router.get("/api/admin/tickets/{ticket_id}")
def admin_ticket_detail(
    ticket_id: int,
    u: User = Depends(get_admin),
    db: Session = Depends(get_db),
) -> Any:
    t = db.get(Ticket, ticket_id)
    if not t:
        raise HTTPException(404, "Ticket not found")
    usr = db.get(User, t.user_id)
    session_id = t.session_id
    msgs = _messages_ordered(db, session_id) if session_id else []
    email_body = _fake_email_body(t, usr)
    return {
        "ticket": {
            "id": t.id,
            "public_id": t.public_id,
            "user_id": t.user_id,
            "user_email": (usr.display_email or usr.email) if usr else "",
            "user_name": usr.name if usr else "",
            "department": t.department,
            "department_confidence": t.department_confidence,
            "routing_reason": t.routing_reason,
            "triage_scores": json.loads(t.triage_scores or "{}"),
            "status": t.status,
            "issue_summary": t.issue_summary,
            "session_id": session_id,
            "created_at": t.created_at.isoformat() if t.created_at else None,
        },
        "messages": [_message_to_dict(m) for m in msgs],
        "fake_outbound_email": {
            "to": (usr.display_email or usr.email) if usr else "",
            "subject": f"Re: {t.public_id} — steps to try",
            "body": email_body,
        },
    }


def _fake_email_body(t: Ticket, usr: User | None) -> str:
    name = (usr.name or "there") if usr else "there"
    return (
        f"Hi {name},\n\n"
        f"Thanks for contacting support. This message refers to **{t.public_id}** for **user_id {t.user_id}**."
        f"\n\n**Issue summary (from the chat):**\n{t.issue_summary or 'N/A'}\n\n"
        f"**Team:** {t.department}\n\n"
        f"Please try the following next steps and reply to this message if the issue continues.\n\n"
        f"(This is a demo draft — not sent.)\n"
    )


def _build_insights(t: Ticket) -> dict[str, Any]:
    """Run the RAG pipeline for a ticket and return the insights payload."""
    from api import AnalyzeRequest, analyze

    desc = (t.issue_summary or "")[:2000] or "support ticket"
    r = analyze(AnalyzeRequest(description=desc, top_k_tickets=4, top_k_kb=3, use_reranking=True))
    return {
        "ticket_id": t.id,
        "public_id": t.public_id,
        "user_id": t.user_id,
        "resolution": r.get("resolution", ""),
        "similar_tickets": r.get("similar_tickets", []),
        "kb_articles": r.get("kb_articles", []),
        "timings": r.get("timings", {}),
    }


@router.get("/api/admin/tickets/{ticket_id}/insights")
def admin_insights(
    ticket_id: int,
    u: User = Depends(get_admin),
    db: Session = Depends(get_db),
) -> Any:
    t = db.get(Ticket, ticket_id)
    if not t:
        raise HTTPException(404, "Ticket not found")

    # Return cached insights immediately — this endpoint never runs the LLM.
    if t.insights_cache:
        try:
            return json.loads(t.insights_cache)
        except Exception:
            t.insights_cache = None
            db.commit()

    raise HTTPException(404, "Insights have not been generated for this ticket yet.")


@router.post("/api/admin/tickets/{ticket_id}/insights/generate")
def generate_insights(
    ticket_id: int,
    u: User = Depends(get_admin),
    db: Session = Depends(get_db),
) -> Any:
    """Generate recommended actions on explicit admin request, then cache them."""
    t = db.get(Ticket, ticket_id)
    if not t:
        raise HTTPException(404, "Ticket not found")
    if t.insights_cache:
        try:
            return json.loads(t.insights_cache)
        except Exception:
            t.insights_cache = None
            db.commit()

    payload = _build_insights(t)
    t.insights_cache = json.dumps(payload)
    db.commit()
    return payload


@router.post("/api/admin/tickets/{ticket_id}/insights/refresh")
def refresh_insights(
    ticket_id: int,
    u: User = Depends(get_admin),
    db: Session = Depends(get_db),
) -> Any:
    """Force-regenerate insights for a ticket, overwriting the DB cache."""
    t = db.get(Ticket, ticket_id)
    if not t:
        raise HTTPException(404, "Ticket not found")

    payload = _build_insights(t)
    t.insights_cache = json.dumps(payload)
    db.commit()
    return payload


# POST /api/admin/create-ticket is handled directly on the FastAPI app in api.py
# to avoid route-ordering conflicts with GET /api/admin/tickets/{ticket_id}.
