"""SQLAlchemy models for the Intelligent Ticketing app (SQLite for dev, Postgres-ready)."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), default="")
    role: Mapped[str] = mapped_column(String(32), default="user")  # user | admin
    display_email: Mapped[str] = mapped_column(
        String(255), default=""
    )  # fake "contact" for templates

    sessions: Mapped[List["SupportSession"]] = relationship(back_populates="user")
    tickets: Mapped[List["Ticket"]] = relationship(back_populates="user")


class SupportSession(Base):
    __tablename__ = "support_sessions"

    id: Mapped[str] = mapped_column(
        String(40), primary_key=True
    )  # uuid string for URLs
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    state: Mapped[str] = mapped_column(
        String(32), default="active"
    )  # active | escalated | closed
    ai_rounds: Mapped[int] = mapped_column(default=0)
    needs_escalation_choice: Mapped[int] = mapped_column(default=0)  # 0/1

    user: Mapped["User"] = relationship(back_populates="sessions")
    messages: Mapped[List["Message"]] = relationship(
        back_populates="session",
    )
    ticket: Mapped[Optional["Ticket"]] = relationship(
        back_populates="session", uselist=False
    )


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    public_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    session_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("support_sessions.id"), unique=True, index=True
    )
    issue_summary: Mapped[str] = mapped_column(Text, default="")
    department: Mapped[str] = mapped_column(String(255), default="")
    department_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    routing_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    triage_scores: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="open")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    insights_cache: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default=None)

    user: Mapped["User"] = relationship(back_populates="tickets")
    session: Mapped["SupportSession"] = relationship(back_populates="ticket")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("support_sessions.id", ondelete="CASCADE"), index=True
    )
    sender: Mapped[str] = mapped_column(String(32), index=True)  # user | ai | system
    content: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    session: Mapped["SupportSession"] = relationship(back_populates="messages")
