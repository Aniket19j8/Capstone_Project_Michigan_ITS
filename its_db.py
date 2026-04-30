from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event, func, select, text
from sqlalchemy.orm import Session, sessionmaker

from its_models import Base, User

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "data" / "its_app.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Set DATABASE_URL env var to switch to PostgreSQL on AWS:
#   export DATABASE_URL="postgresql://user:pass@rds-host:5432/itsdb"
# Defaults to local SQLite for development.
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH.as_posix()}")

_is_sqlite = DATABASE_URL.startswith("sqlite")

if _is_sqlite:
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False, "timeout": 30},
        echo=False,
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _record) -> None:  # type: ignore[no-untyped-def]
        """WAL mode + busy_timeout so Ollama waits don't lock SQLite."""
        cur = dbapi_connection.cursor()
        try:
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA busy_timeout=30000")
        finally:
            cur.close()
else:
    # PostgreSQL — use connection pooling (QueuePool default), no SQLite pragmas.
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,  # drop stale connections before use
        echo=False,
    )


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    # SQLite migration shim: add insights_cache column to DBs that predate this schema.
    # PostgreSQL uses create_all which handles the column automatically.
    if _is_sqlite:
        with engine.connect() as conn:
            for stmt in (
                "ALTER TABLE tickets ADD COLUMN insights_cache TEXT",
                "ALTER TABLE tickets ADD COLUMN department_confidence FLOAT",
                "ALTER TABLE tickets ADD COLUMN routing_reason TEXT",
                "ALTER TABLE tickets ADD COLUMN triage_scores TEXT",
            ):
                try:
                    conn.execute(text(stmt))
                    conn.commit()
                except Exception:
                    pass  # Column already exists — safe to ignore
    with SessionLocal() as db:
        n = db.scalar(select(func.count()).select_from(User)) or 0
        if int(n) < 1:
            db.add(
                User(
                    email="user@demo.com",
                    name="Demo User",
                    role="user",
                    display_email="user@demo.com",
                )
            )
            db.add(
                User(
                    email="admin@demo.com",
                    name="Demo Admin",
                    role="admin",
                    display_email="admin@demo.com",
                )
            )
            db.commit()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
