# Intelligent Ticketing System — Implementation Plan (UI-First)

This document explains **how** we will evolve this repository toward your two-view Intelligent Ticketing System, **why** we are sequencing work this way, and what we need to decide before building.

> **Status:** Decisions below are **locked** for implementation. Remaining open items are optional polish.

---

## 0. Locked product decisions (your answers)

| Topic | Decision |
|--------|----------|
| **Departments (Model 2)** | **Fixed set of 4** — hardcoded or config (not admin-CRUD for v1). |
| **Model 1 confidence** | **Fixed threshold** — use **0.5** as in the spec; not admin-tunable for now. |
| **“Not working” / connect intent** | **Hybrid:** maintain a **keyword/phrase list** (fast, predictable) and optionally layer **semantic meaning** (e.g. embedding similarity to intent phrases) or a **small classifier** for edge cases. |
| **Voice** | Use whatever is **most reliable in practice:** default to **Web Speech API** in the browser for **live transcription** with no extra server dependencies and broad access (HTTPS/localhost). If quality is insufficient, add **faster-whisper** (or similar) in FastAPI as an optional path — your repo already has `faster-whisper` in `requirements.txt`. |
| **Auth & email (v1)** | **Demo-only:** no real email delivery; “outbound” can be **fake/pre-filled** text with **ticket_id** and **user_id** (and any other required fields) **consistently linked** in API responses and admin UI. |
| **Data store — timing** | **Full** relational DB (PostgreSQL, Alembic, KB `embeddings` column, full tables) targets **final delivery**. **First milestones** still enforce **correct mapping:** every ticket references **`user_id`**, messages reference **`ticket_id`**, and admin views resolve the same graph — we can use **SQLite** (or a thin repository layer) for early milestones and **migrate to PostgreSQL** for final cloud deployment without changing the **API contract**. |
| **Deployment** | **Local validation first** (prove flows end-to-end), then **cloud** (env-based `VITE_API_BASE`, CORS, managed Postgres or container). |

**Invariant from day one (even before full DB):** `user_id` ↔ `ticket_id` and message threading must be **traceable in one store** (no orphan tickets, no client-only random IDs for persisted entities).

---

## 1. What already exists in this repo

| Layer | What you have today |
|--------|----------------------|
| **Frontend** | React + TypeScript + Vite (`frontend/`), single main screen: ticket description, optional **Web Speech API** voice input with live partial transcript, submit → animated pipeline → results. |
| **Backend** | FastAPI `api.py`: `GET /api/status`, `POST /api/analyze` — hybrid RAG (ChromaDB + BM25 + rerank) + Ollama LLM, returns resolution blueprint, similar tickets, KB articles, timings. |
| **Data** | Processed tickets/KB, vector stores under `data/`, not a normalized **transactional** DB for live tickets and chat messages. |
| **Stack note** | `requirements.txt` has no PostgreSQL driver yet; persistence today is file/vector oriented. |

**Important inconsistency to fix when we implement:** `frontend/src/api.ts` uses base URL `http://localhost:8001`, while `README.md` and typical `uvicorn` run use port **8000**. We will align one canonical port and document it (or use `VITE_API_BASE` env for flexibility).

---

## 2. What you are asking for (target)

- **User view:** Login (demo) → **chat-first** support UI with text + **voice** (transcribed in-thread) → **Model 1** (issue resolution with **confidence**; retry rules; max 2 AI attempts) → optional **“Connect to Specialist”** → create ticket with **ticket number, AI summary, Model 2 department** → end user flow.
- **Admin view:** Login → **left:** ticket queue → **right/center split:** details + full chat, **RAG/LLM insights** (similar tickets, KB, dupes, recommendations), **editable outbound email**, **“save to knowledge base”** for continuous learning.
- **Database:** Users, Tickets, Messages, Knowledge Base (with embeddings for RAG) — **correct foreign-key mapping** for user → ticket → messages → department.
- **Your priority:** **UI first**; full production DB (Postgres, migrations, KB embeddings in DB) lands at **final delivery**, while **early milestones** still persist **user → ticket → messages** with correct IDs (e.g. SQLite or equivalent), **fake** outbound email, and **Model 1 / Model 2** via existing RAG/LLM or stubs with stable API shapes.

---

## 3. Why “UI first” *with* correct IDs early, full DB at the end

- **UI-first** means we ship **navigation, layouts, and interaction flows** (user chat + admin dashboard) so demos are credible before every ML detail is final.
- **Data must not be an afterthought:** If we only mock state in React, we risk **wrong message ordering, missing `ticket_id` on messages, and broken admin views**. We add a **persistence layer with proper FK-style relationships from milestone 1** (SQLite is acceptable) and **move to PostgreSQL + Alembic + full schema** for **final delivery** and cloud.
- **Full KB table with `embeddings` in PostgreSQL** and any **pgvector** work align with **final delivery**, not necessarily the first local demo.

**Persistence path (matches your “DB = final” preference)**

1. **Early / local:** **SQLAlchemy (or SQLModel) + SQLite** file — `users`, `tickets`, `messages` (and later `knowledge_base`) with `user_id` on tickets, `ticket_id` on messages, **4 fixed departments** (enum or string column). Seed demo user/admin. **No real email**; store display email on user if needed for fake templates.
2. **Final / cloud:** **PostgreSQL** (managed or Docker), **Alembic** migrations, optional **KB + embeddings** in DB or keep Chroma for RAG with **operational** data in Postgres only — we pick one clear split before deploy.
3. Expose **REST (FastAPI) routes** for: demo auth, chat turns, escalation (create ticket + link messages), admin queue/detail, **fake email** payload built from `ticket_id`, `user_id`, summary, department.
4. **Model 1 (resolution + confidence):** wrap or extend **`/api/analyze`**, structured JSON for `response` + `confidence` (fixed 0.5 rule).
5. **Model 2 (department):** **4 fixed labels**; LLM or classifier over issue text — swap implementation without changing the UI contract.

This keeps the **UI contract stable** while models improve behind `POST /api/...` endpoints.

---

## 4. Proposed front-end structure (Vite + React, current repo)

The README mentions Next.js as an option; **this repo is already Vite**. Migrating to Next.js is a **large** change (routing, SSR, deployment). For capstone speed and your “UI first” goal, the pragmatic path is:

- **Stay on Vite** and add **`react-router-dom`** (or **TanStack Router**): routes like `/` (marketing/demo product shell), `/support` (user chat), `/admin` (queue + detail).
- Reuse your **NeuralBg / design tokens** where they fit; **split** “demo product chrome” (dynamic branding) from **chat** and **admin** shells so the same components can be **embedded in another product** later (iframe or npm package is a follow-on).

**Layout mapping**

- **User:** Chat thread component + composer (text + mic using existing `useSpeechRecognition`) + **decision buttons** (Retry / Connect) driven by app state, not ad-hoc alerts.
- **Admin:** **CSS grid** — `queue | insights | details` (or `queue | details+insights` on small screens with tabs). Reuse patterns from `ResultPanel` for “similar tickets / KB / dupes / recommendations” but fed from **admin ticket context APIs**, not from a one-off analyze form.

---

## 5. Backend API shape (high level)

We will keep existing `GET/POST` analyze behavior for RAG, and **add** resource-oriented routes, for example:

- `POST /api/auth/login` (demo) → session or JWT
- `POST /api/sessions` or `POST /api/chat/turn` → stores message, returns Model 1 reply + confidence + `attempts_remaining`
- `POST /api/tickets` (escalation) → creates ticket row, runs Model 2, returns ticket number, summary, department
- `GET /api/admin/tickets` + `GET /api/admin/tickets/{id}` + `PATCH ...`
- `POST /api/admin/tickets/{id}/knowledge` → optional KB expansion + embedding job

Exact paths can be adjusted; the **principle** is: **one source of truth in DB**, **idempotent** ticket creation, **message table** for full chat for admin.

---

## 6. Phased delivery (suggested)

| Phase | Outcome | Why this order |
|--------|---------|----------------|
| **A. UX skeleton** | Routes, user chat UI, admin queue + detail shells | Fast visual alignment with your spec. |
| **B. API + light DB** | SQLite (or similar), users/tickets/messages, seed users, `user_id`–`ticket_id` integrity | Proves end-to-end mapping before cloud DB. |
| **C. Chat + Model 1** | Turn-by-turn storage, **fixed 0.5** confidence + **2-try** + keyword/semantic escalation | Core user story. |
| **D. Escalation + Model 2 (4 depts)** | Ticket creation, summary, department on screen | User flow “done” state. |
| **E. Admin intelligence** | Similar tickets / KB / dupes / LLM + **fake** email (linked IDs) | Reuse RAG from existing pipeline. |
| **F. Final delivery DB** | PostgreSQL, migrations, full KB/embedding story as specified, **cloud** deploy | Scales and matches course requirements. |
| **G. Hardening** | `VITE_API_BASE`, CORS, port fix, smoke checks | Local → cloud. |

---

## 7. How this maps to your “Intelligent + embeddable” story

- **Embeddable widget later:** If we build the support UI as a **self-contained route tree + API client module**, a partner site can load `/support` in an iframe or import a packaged component bundle with `apiBase` configuration.
- **1000+ tickets scale:** Your existing **indexing and hybrid retrieval** path is the right direction; **operational** tables (tickets/messages) move to **PostgreSQL** at final delivery, while **search/KB** can stay on Chroma through mid milestones unless we consolidate earlier.

---

## 8. Optional follow-ups (not blocking)

- **Next.js vs Vite:** Still defaulting to **Vite + react-router** unless you need SSR; say if judges require Next.
- **Branding:** **One** demo product name/theme vs **JSON-driven** theme (logo, colors) — can add after core flows work.
- **Real outbound email / SMTP** — out of scope until you say otherwise; v1 = fake + copyable body with correct **ticket** / **user** links.

---

## 9. Next implementation steps (aligned to your decisions)

1. **Phase A + B:** Routes, user + admin UIs, FastAPI + SQLite, demo auth, all entities with **`user_id` / `ticket_id` / message** linkage.  
2. **Phase C–E:** Model 1, escalation detection (keywords + optional semantic), Model 2 (4 departments), admin RAG + fake email.  
3. **Phase F–G:** PostgreSQL, full schema as in your spec, **cloud** configuration and CORS.

Say when to start coding in the repo; the locked decisions above are the build spec.
