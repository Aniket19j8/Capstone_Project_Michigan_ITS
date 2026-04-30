# Intelligent Ticketing — What Changed, Why, and How to Run

This document describes **how the project differed before these updates**, **what we added and why**, **the current state**, and **how to run everything locally (and for cloud later)**.

It complements the planning doc [INTELLIGENT_TICKETING_UI_PLAN.md](INTELLIGENT_TICKETING_UI_PLAN.md) (goals and phased roadmap).

---

## 1. Where things were before

| Area | Before |
|------|--------|
| **User experience** | A single Vite page: one “submit issue → RAG pipeline → blueprint” flow (`App.tsx` only). No customer chat, no two-step AI + escalation, no ticket confirmation screen, no admin console. |
| **Frontend repo files** | Key tooling was **missing in-repo** (no `package.json` / `tsconfig` checked in), so a fresh clone could not `npm install` or build without recreating that metadata. The README assumed those files existed. The API base URL in `api.ts` pointed at **port 8001** while the backend docs used **8000** — an easy source of “frontend can’t reach API” issues. |
| **Backend** | `api.py` exposed **`GET /api/status`** and **`POST /api/analyze`** (hybrid RAG + Ollama). There was **no** transactional app database for real users, chat sessions, or tickets — only processed data, Chroma, and CSV/JSON for the research pipeline. |
| **Product goals** | The capstone spec called for a **User view** (chat, voice, Model 1 with confidence, specialist handoff, Model 2 department routing) and an **Admin view** (queue, full chat, RAG/LLM insights, fake email draft) with **clear `user_id` → `ticket_id` → `messages` mapping** — none of that existed in the running app. |

---

## 2. What we added (summary)

### 2.1 Backend — app data + ticketing API

| File / change | Purpose |
|---------------|---------|
| `its_models.py` | SQLAlchemy models: **`User`**, **`SupportSession`**, **`Message`**, **`Ticket`**. Tickets reference **`user_id`** and a unique **`session_id`**; messages belong to a session. |
| `its_db.py` | SQLite at **`data/its_app.db`**, `SessionLocal`, **`init_db()`** (create tables + seed **user@demo.com** / **admin@demo.com**). Designed so you can point the same code at **PostgreSQL** later with minimal model changes. |
| `its_brain.py` | **Model 1** (Ollama + JSON for `response` + `confidence`, RAG via `get_chat_rag_context`), **Model 2** (4 fixed departments + LLM + keyword fallback), escalation phrase lists, public ticket id (`ITS-…`). |
| `its_routes.py` | FastAPI routes: **demo auth**, **sessions + chat + actions**, **admin list/detail/insights** (insights reuse existing **`analyze()`** in `api.py`). |
| `api.py` | **`get_chat_rag_context()`** for the chat model; **`@app.on_event("startup")` → `init_db()`**; **`app.include_router(its_routes)`** so ticketing APIs mount alongside the existing RAG API. |
| `requirements.txt` | **`sqlalchemy>=2.0.0`** for the ORM layer. |

**Why this shape:** we needed **durable, relational links** for demos and grading, without waiting for the “final” PostgreSQL milestone — SQLite gives real FK-style integrity today; the plan’s full Postgres deployment remains a **later delivery** step.

### 2.2 Frontend — routes, new screens, API client

| File / change | Purpose |
|---------------|---------|
| `package.json` | Declares **React, Vite, TypeScript, react-router-dom, tailwind, lucide-react, react-markdown** so the project is installable and buildable. |
| `tsconfig.json`, `tsconfig.app.json`, `tsconfig.node.json` | Standard Vite + TS build setup. |
| `src/main.tsx` | Wraps the app in **`BrowserRouter`**. |
| `src/App.tsx` | **Routes** instead of a single monolith page. |
| `src/api.ts` | Single **`VITE_API_BASE`** (default **`http://localhost:8000`**) plus helpers for **login, sessions, messages, admin**, and the original **status/analyze** calls. |
| `src/pages/HomePage.tsx` | Entry links: **user chat**, **admin**, **RAG playground**. |
| `src/pages/LoginPage.tsx` | Demo login (**password: `demo`**), redirects by role. |
| `src/pages/UserSupportPage.tsx` | Chat UI, Web Speech (existing hook), **Try again / Connect to specialist** after rules fire, shows ticket + department when escalated. |
| `src/pages/AdminPage.tsx` | **Queue | RAG center | details + fake email** layout. |
| `src/pages/RagDemoPage.tsx` | The **old** one-shot “Submit & Analyze” RAG experience, kept at route **`/rag`**. |
| `src/vite-env.d.ts` | TypeScript support for `import.meta.env` / **`VITE_API_BASE`**. |

**Why:** the spec required **two main views** and a **chat-first** flow while preserving the **original RAG demo** for backward compatibility and benchmarks.

---

## 3. Where we are now

- **User path:** sign in as **`user@demo.com`** / **`demo`** → support chat with **Model 1** (with confidence; fixed **0.5** threshold in code) → up to **two** AI rounds → then **Try AI again** or **Connect to specialist** → **ticket** row with **`public_id`**, summary, **department (Model 2)**, and **`user_id` visible in the copy for the demo.  
- **Admin path:** sign in as **`admin@demo.com`** / **`demo`** → ticket queue → select ticket → **right:** user id, email, issue, chat; **center:** RAG/LLM insights (same family of output as `/api/analyze`); **fake** outbound email text (not sent).  
- **Data:** SQLite file **`data/its_app.db`** holds users, sessions, messages, and tickets with consistent IDs.  
- **RAG + LLM health:** still depends on your existing stack (**Ollama** on **`qwen3:4b`**, Chroma/retrievers for full retrieval). If retriever/LLM is down, chat may still return lower-confidence or fallback text — check **`GET /api/status`**.

---

## 4. Running instructions (local)

### 4.1 Prerequisites

- **[uv](https://docs.astral.sh/uv/)** (Python env + install; install once: e.g. `pip install uv` or see the docs).  
- **Python 3.10+** (match your course environment; uv can install a pin if you add one later).  
- **Node.js + npm** (for the Vite app).  
- **Ollama** with **`qwen3:4b`** (and `ollama serve`) for best results.  
- If you use the full RAG index: run your existing data pipeline (e.g. scripts `01`–`03` and Chroma) as in the main [README](README.md).

### 4.2 Environment variables (Vite / API URL)

The React app reads **`VITE_API_BASE`** (see `frontend/src/api.ts`). If it is not set, it defaults to **`http://localhost:8000`**.

1. In **`frontend`**, copy the example file and edit:

   ```bash
   cd frontend
   copy .env.example .env
   ```

   (On macOS/Linux: `cp .env.example .env`.)

2. Set the API base inside **`frontend/.env`**:

   ```env
   VITE_API_BASE=http://localhost:8000
   ```

   For a remote API, use your real origin, e.g. `https://api.example.com` (no trailing slash required).

3. **Restart** `npm run dev` after changing `.env` so Vite picks it up.

4. **One-off in the shell** (no file): you can set the variable for a single `npm` run, e.g. **PowerShell**:

   ```powershell
   $env:VITE_API_BASE = "http://127.0.0.1:8000"
   npm run dev
   ```

   **cmd.exe:**

   ```bat
   set VITE_API_BASE=http://127.0.0.1:8000
   npm run dev
   ```

   **bash / zsh:**

   ```bash
   export VITE_API_BASE=http://127.0.0.1:8000
   npm run dev
   ```

For **`npm run build`**, set **`VITE_API_BASE`** the same way (or in `.env`) so production bundles call the right API.

### 4.3 Backend (using **uv** instead of `pip`)

From the **repository root**:

```bash
uv venv
```

Activate the venv, then install dependencies with **uv** (use the same PowerShell or bash you use in class):

- **Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1`  
- **Windows (cmd):** `.venv\Scripts\activate.bat`  
- **macOS / Linux:** `source .venv/bin/activate`

Then:

```bash
uv pip install -r requirements.txt
uv run uvicorn api:app --reload --port 8000
```

`uv run` uses the project’s **`.venv`** when present, so the second line runs Uvicorn with the packages you installed.

If you prefer to use `python` explicitly after `uv pip install`:

```bash
python -m uvicorn api:app --reload --port 8000
```

(That still uses the activated venv’s Python.)

- Health / RAG status: `GET http://localhost:8000/api/status`  
- Legacy analyze: `POST http://localhost:8000/api/analyze`  
- Ticketing APIs are under the same host, e.g. `http://localhost:8000/api/auth/login`, `.../api/sessions`, etc.

**First run:** startup calls **`init_db()`**, which creates **`data/its_app.db`** and seeds demo users if the DB is empty.

### 4.4 Frontend

```bash
cd frontend
npm install
npm run dev
```

- Default Vite URL: **`http://localhost:5173`**.  
- With **`frontend/.env`** in place, **`VITE_API_BASE`** points the UI at your API; see **§4.2** above.  
- Rebuild when deploying: `npm run build` with the correct **`VITE_API_BASE`** for that environment.

### 4.5 Demo accounts

| Email | Role | Password |
|--------|------|----------|
| `user@demo.com` | `user` | `demo` |
| `admin@demo.com` | `admin` | `demo` |

### 4.6 Typical dev commands (reference)

| Goal | Command (from repo root, venv activated) |
|------|--------|
| API only | `uv run uvicorn api:app --reload --port 8000` |
| UI only (after `npm install` in `frontend/`) | `cd frontend && npm run dev` |
| Production build of UI | `cd frontend && npm run build` → serve `frontend/dist`; set **`VITE_API_BASE`** at **build** time. |

---

## 5. Cloud / next steps (brief)

- Set **`VITE_API_BASE`** to your deployed API URL and **rebuild** the frontend.  
- On the server, use **`uvicorn` + HTTPS**; configure **CORS** if the UI is on a different origin (your `api.py` already uses permissive CORS for dev — tighten for production).  
- **Final delivery** in the plan: migrate to **PostgreSQL** + formal migrations; this doc’s “where we are now” is intentionally **SQLite + same API** so you can swap the database URL without redesigning the UI.

---

*Generated to match the implementation on branch `Feature_hallucination_and_Frontend_updated_03_04_2026`.*
