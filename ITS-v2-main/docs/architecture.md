# Architecture

Capstone ITS v2 is a FastAPI application with a user-facing chat triage agent and an admin dashboard. The backend uses LangChain for model and retrieval calls, LangGraph for the stateful agent loop, Pydantic for every LLM structured output, SQLite for tickets and graph checkpoints, and Pinecone for vector search.

> **History note.** An earlier version of this document described an 8-node deterministic LangGraph workflow (`input_guardrails → assess_requirements → ask_follow_up → classify_and_guard → retrieve_and_answer → validate_answer → finalize_resolution → create_ticket`). The current implementation in `app/graph.py` is a simpler **3-node ReAct agent** — the LLM owns reasoning and Python only owns side effects and safety. The reasoning + classification + clearance filtering all happen inside the agent step. We made this simplification deliberately for lower latency, fewer brittle deterministic branches, and clean composition with `langgraph-checkpoint-sqlite`. The lessons from the original 8-node design (per-step Pydantic structured output, follow-up loop, answer validation with redaction) are still present — they are now implemented inside the tools, the guardrail node, and the post-processing in `run_chat_turn`.

## Runtime Components

- **FastAPI** serves the chat page, admin dashboard, static assets, and JSON APIs (see `app/main.py`).
- **LangGraph** owns the multi-turn workflow and persists thread state with `langgraph-checkpoint-sqlite` at `data/langgraph_checkpoints.sqlite`.
- **LangChain** wraps the configured chat model, Pinecone retrieval, and FlashRank reranking via `app/rag.py` and `app/rag_retrieve.py`.
- **Pydantic** validates graph state, ticket metadata, guardrail decisions, KB references, and the final `ChatTurnResult` returned to the client.
- **SQLite + SQLAlchemy** stores tickets, ticket messages, KB links, duplicate relations, projects, project members, and chat messages. The schema lives in `app/db.py`; migrations in `app/db_migrations.py`.
- **Pinecone** stores KB vectors in `its-knowledge-base` and ticket vectors in `its-tickets` (defaults; configurable).
- **OpenAI / Ollama / HuggingFace** are switchable behind `app/llm.py` (`LLM_PROVIDER`, `EMBEDDING_PROVIDER`).

## API Surface

Implemented in `app/main.py` (cookie-session auth via `app/auth.py`):

- `GET  /api/health` — basic service probe.
- `GET  /auth/me`, `POST /auth/login`, `POST /auth/register`, `POST /auth/logout`, `POST /auth/change-password`, `POST /auth/reset-password` — session lifecycle.
- `GET  /api/projects`, `POST /api/projects`, `GET /api/projects/{id}/members`, `POST /api/projects/{id}/members`, `DELETE /api/projects/{id}/members/{uid}` — projects + membership scoping.
- `GET  /api/tags`, `GET /api/users` — auxiliary lookups.
- `GET  /api/chat/threads`, `GET /api/chat/history` — thread + message history persisted in SQLite (separate from LangGraph checkpoints, so history survives checkpoint rotation).
- `POST /api/chat` — invokes `app.graph.run_chat_turn`; returns a `ChatTurnResult` with `thread_id`, `response`, `route` (`self_resolution | ticket_created | blocked | follow_up`), optional `ticket_id`, and `linked_kb_articles`.
- `GET  /api/tickets`, `POST /api/tickets`, `GET /api/tickets/{id}`, `GET /api/tickets/{id}/insights` — ticket queue, ticket detail, and on-demand insights (Pinecone duplicate candidates + RAG suggestions + LLM-suggested fixes).
- `GET  /api/admin/insights`, `POST /api/admin/analytics` — admin-scoped analytics.
- `GET  /api/kb/doc` — serves a KB document with clearance check, falling back to the legacy KB dir at `LEGACY_KB_DIR` when needed.
- `GET  /openapi.json`, `GET /docs` — admin-only; non-admin users see `403`.
- `GET  /`, `/login`, `/register`, `/admin`, `/{frontend_path:path}` — serve the React SPA from `frontend/dist` when present, else fall back to Jinja templates under `app/templates/`.

## SQL Schema

Defined in `app/db.py`:

- `users` — credentials, role, clearance.
- `sessions` — cookie sessions.
- `projects`, `project_members` — project scoping for tickets and chat.
- `tickets` — status, timestamps, user/thread ids, app/environment, clearance, category, priority, summary, keywords, plus serialized Pydantic intelligence/resolution/guardrail/conversation/raw context blobs.
- `ticket_messages` — normalized message audit trail per ticket.
- `ticket_kb_links` — KB articles linked to a ticket with relevance and clearance metadata.
- `duplicate_ticket_links` — ticket-to-ticket duplicate candidates (filled by `vector_search_tickets`).
- `chat_messages` — per-thread user/assistant turns, written by `add_chat_turn_messages` from `app/graph.py`.
- LangGraph checkpoint tables are managed by `SqliteSaver` in `data/langgraph_checkpoints.sqlite`.

## Pydantic Contracts

Defined in `app/schemas.py`:

- `HelpdeskAgentState` (`TypedDict`) — `messages` (LangChain `BaseMessage` list with `add_messages` reducer), `user_id`, `thread_id`, `app_name`, `environment`, `user_clearance`, `is_blocked`, `route`.
- `ChatTurnResult` — `thread_id`, `response`, `route`, optional `ticket_id`, `linked_kb_articles` (list of `KBArticleRef`).
- `TicketCreate` / `TicketRead` / `TicketIntelligence` — strict ticket schemas.
- `GuardrailDecision`, `AnswerValidation`, `RequirementAssessment`, `IssueClassification`, `SelfResolutionAnswer` — structured-output contracts for the guardrail and post-processing layers.

## LangGraph Agent (current implementation)

```
START
  └── guardrail_node      ── (block?) ──► END (route="blocked", security ticket filed)
        │
        ▼
      agent_node  ◄─────────────┐
        │                       │
        │ (chooses tool calls)  │
        ▼                       │
      tools_node ───────────────┘
        │
        └── (no more tool calls) ──► END (route resolved by run_chat_turn)
```

Three nodes in `app/graph.py`:

1. **`guardrail_node`** — runs deterministic guardrails (`app/guardrails.evaluate_input_safety`) for prompt injection, hidden-prompt extraction, privileged-access requests, and credential disclosure. On block, files a security ticket, sets `is_blocked=True`, and emits a sanitized refusal `AIMessage`.
2. **`agent_node`** — binds the chat model to five tools and runs one ReAct step. The `_REQUEST_CTX` `ContextVar` carries `user_id`, `thread_id`, `display_name`, `user_role`, `user_clearance`, `project_id`, `project_ids`, `app_name`, `environment`, `kb_refs`, `ticket_id`, and `messages_snapshot` so tools can act on behalf of the right user without re-passing them.
3. **`tools_node`** — executes any tool calls emitted by `agent_node` and returns `ToolMessage`s back to the agent.

Routing:

- `_route_after_guardrail` ends the graph when blocked, otherwise hands off to `agent`.
- `_route_after_agent` routes to `tools` while the last `AIMessage` carries `tool_calls`, otherwise ends.
- `MAX_RECURSION` is set per-invocation via `_chat_invoke_config` to keep the loop bounded.

After the graph returns, `run_chat_turn`:

- Picks the last non-tool-call `AIMessage` as the user-visible response.
- Runs `redact_sensitive_text` on the response (emails, phones, SSNs, API keys, passwords, tokens, private keys).
- Resolves the route label: `blocked` if the guardrail tripped, else `ticket_created` if a ticket id was set, else `self_resolution` if KB references were collected, else `follow_up`.
- Returns a fully-typed `ChatTurnResult`.

## Tools Available to the Agent

All defined in `app/graph.py` and decorated with `@tool`:

| Tool | Purpose |
|---|---|
| `search_knowledge_base(query)` | Pinecone hybrid RAG over `its-knowledge-base` with clearance/category/app/env filters; returns ranked KB chunks and pushes them onto `kb_refs` so the chat layer can render citations. |
| `search_existing_tickets(query)` | SQL search across SQLite tickets with project scoping; used when the user asks if a ticket already exists. |
| `analyze_ticket_data(question)` | Read-only SQL agent (`app.admin_analytics`) for ticket counts, totals, breakdowns, trends, and bounded lists. |
| `vector_search_tickets(query, status?, priority?)` | Pinecone vector search over `its-tickets` (`app/ticket_vector.py`) with optional status/priority metadata filters; surfaces duplicates and similar incidents. |
| `create_helpdesk_ticket(summary, category, priority, ...)` | Persists a ticket in SQLite with the conversation snapshot, links KB references and duplicate candidates, and sets `ticket_id` in the request context. |

## Guardrails

- HTTP/Pydantic validation limits payload shape and size on every endpoint.
- `app/guardrails.py` detects prompt injection, hidden-prompt extraction, privileged-access requests, credential disclosure, and sensitive output patterns.
- `redact_sensitive_text` sanitizes emails, phone numbers, SSNs, API keys, passwords, tokens, and private keys before any chat response is returned.
- KB retrieval applies clearance/category/app/environment metadata filters in Pinecone (`build_pinecone_filter`) so standard users never see internal or restricted KB metadata.

## RAG Pipeline

Defined in `app/rag.py`, `app/rag_ingest.py`, `app/rag_retrieve.py`, plus the ingest scripts:

1. `scripts/ingest_kb.py` loads Markdown files from `kb/` (or `LEGACY_KB_DIR`) with LangChain `DirectoryLoader` + `TextLoader`.
2. Front matter is parsed only to normalize metadata: `category`, `clearance` / `clearance_level`, `app_name`, `environment`.
3. Content is split with `MarkdownHeaderTextSplitter` then `MarkdownTextSplitter`.
4. Chunks are stored in Pinecone (`its-knowledge-base`) using the embedding model returned by `app.llm.get_embedding_model()`.
5. At query time `HybridRAGPipeline` builds a Pinecone metadata filter from user clearance, category, app, and environment, then runs similarity search and reranks with LangChain's `FlashrankRerank`.
6. The retrieved chunks become `kb_refs` on the agent context and are returned to the client as `linked_kb_articles` for citation rendering.

## Ingest Scripts

- `scripts/ingest_kb.py` — KB → Pinecone (`its-knowledge-base`).
- `scripts/ingest_tickets.py` — CSV → SQLite (with flexible column aliases including `ticket id`, `title`, `description`, `category`, `severity`, `status`, `resolution`, `embedding text`, `unified id`, `external record id`); optional Pinecone upsert into `its-tickets`. Supports `--dry-run`, `--start-record`, `--end-record`, `--db-batch-size`, `--vector-batch-size`, `--skip-vector-index`.
- `scripts/seed_projects.py`, `scripts/create_admin.py`, `scripts/migrate_db.py` — bootstrap users, projects, and migrations.

## Evaluation Scripts

Two scripts under `scripts/` produce v2-vs-research-stack comparison artifacts in `evaluation_v2/`:

- `scripts/eval_v2_retrieval.py` — measures Pinecone+OpenAI retrieval latency and KB grounding behavior; compares to the documented research-stack baselines.
- `scripts/eval_v2_agent.py` — runs `run_chat_turn` against a fixed scenario set covering self-resolution, ticket creation, OOS, and prompt injection; reports turn latency, route distribution, KB citation rate, ticket-creation success, and guardrail precision.

Both support `--mode live` (real Pinecone/OpenAI calls) and `--mode mock` (cached numbers so charts always render). See the root `README.md` and this folder’s `README.md` for how v2 fits the wider project.

## Runbook

```bash
uv sync
cp .env.example .env
uv run python scripts/ingest_kb.py
uv run python scripts/ingest_tickets.py
uv run uvicorn app.main:app --reload
```

Model construction is centralized in `app/llm.py`:

- OpenAI chat and embeddings: `LLM_PROVIDER=openai`, `EMBEDDING_PROVIDER=openai`.
- Ollama chat: `LLM_PROVIDER=ollama`, `OLLAMA_MODEL=...`, `OLLAMA_BASE_URL=...`.
- HuggingFace embeddings: `EMBEDDING_PROVIDER=huggingface` after installing the `huggingface` extra.
- Pinecone: set `PINECONE_API_KEY`; defaults are `PINECONE_KB_INDEX_NAME=its-knowledge-base` and `PINECONE_TICKET_INDEX_NAME=its-tickets`.

## Privacy-first deployment of the same product

Without OpenAI or Pinecone the same FastAPI + React + LangGraph stack runs locally:

- `LLM_PROVIDER=ollama` with `qwen3:4b` (optionally + the LoRA helpdesk adapter from the research stack).
- `EMBEDDING_PROVIDER=huggingface` with `sentence-transformers/all-MiniLM-L6-v2` or any local SentenceTransformer.
- Vector store swapped to self-hosted ChromaDB (or Qdrant for >1M vectors) by replacing the Pinecone client in `app/rag_retrieve.py`.

Same evaluation scripts, same UI, lower throughput, no data leaves the network.
