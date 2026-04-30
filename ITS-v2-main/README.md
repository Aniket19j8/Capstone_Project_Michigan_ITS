# Capstone ITS v2

Conversational IT helpdesk and ticketing system built with FastAPI, LangChain, LangGraph, Pydantic, SQLite, and Pinecone.

> **Where this fits in the project.** The repo root carries the **research stack** (Ollama + Qwen3 + ChromaDB + BM25 + LoRA) used to design and quantify the system. **This folder is the production stack** — the same product, re-platformed for sub-second turn latency by switching the vector store to Pinecone, the embedding+chat to OpenAI, and the agent to LangGraph. See `../CAPSTONE_FINAL_STORY.md` for the canonical narrative and `docs/architecture.md` for the runtime details.
> **Architecture (today):** a 3-node ReAct agent — `guardrail → agent ↔ tools` — implemented in `app/graph.py`. The five tools are `search_knowledge_base`, `search_existing_tickets`, `analyze_ticket_data`, `vector_search_tickets`, and `create_helpdesk_ticket`. SQLite is the source of truth for tickets, projects, users, and chat messages; Pinecone holds KB and ticket vectors.

## Quick Start

```bash
uv sync
cp .env.example .env
uv run python scripts/ingest_kb.py
uv run python scripts/ingest_tickets.py
uv run uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` for the chatbot and `http://127.0.0.1:8000/admin` for the admin dashboard.

## React frontend

The new React UI lives in `frontend/` and calls the existing FastAPI routes under `/api` and `/auth`.

```bash
cd frontend
npm install
npm run dev
```

Run the FastAPI server on `http://127.0.0.1:8000`; Vite serves the UI on `http://127.0.0.1:5173` and proxies API/auth requests to FastAPI.

For a production build:

```bash
cd frontend
npm run build
cd ..
uv run uvicorn app.main:app --reload
```

When `frontend/dist/index.html` exists, FastAPI serves the React app at `/`, `/login`, `/register`, `/admin`, and other client routes while keeping the API endpoints unchanged.

Set `OPENAI_API_KEY` and `PINECONE_API_KEY` in `.env` before using the LangGraph chatbot. The default Pinecone indexes are `its-knowledge-base` and `its-tickets`. See `docs/architecture.md` for the schema, graph, guardrail, and RAG details.

## Import ticket CSV

To import historical tickets into SQLite and upsert each ticket into the Pinecone ticket index:

```bash
uv run python scripts/ingest_tickets.py --csv path/to/tickets.csv
```

For large CSVs, import a slice and batch vector upserts:

```bash
uv run python scripts/ingest_tickets.py --csv path/to/tickets.csv \
  --start-record 1 \
  --end-record 10000 \
  --db-batch-size 500 \
  --vector-batch-size 100
```

Supported CSV columns include the current export headers:

`ticket id`, `title`, `title clean`, `description`, `description clean`, `category`,
`component`, `issue type`, `severity`, `status`, `resolution`, `resolution clean`,
`error codes`, `quality score`, `embedding text`, `source`, `unified id`,
`import id`, `external record id`, `external source`, `language`, and `tags`.
When `embedding text` is present, ticket vector indexing uses that cleaned text.

Use `--dry-run` first to validate rows without writing data:

```bash
uv run python scripts/ingest_tickets.py --csv path/to/tickets.csv --dry-run
```

To rebuild the whole ticket vector index from database tickets later:

```bash
uv run python scripts/ingest_tickets.py
```

CSV import writes tickets even when Pinecone is unavailable. Set `PINECONE_API_KEY` and `PINECONE_TICKET_INDEX_NAME` when you want the same run to upsert vectors into the ticket index. Use `--skip-vector-index` to load SQLite first and index vectors later with `uv run python scripts/ingest_tickets.py`.

Model providers are centralized in `app/llm.py`. Switch chat to Ollama with `LLM_PROVIDER=ollama`; switch embeddings with `EMBEDDING_PROVIDER=openai` or `EMBEDDING_PROVIDER=huggingface`.

If Pinecone reports a vector dimension mismatch during ingestion, make the index dimension match the embedding output. For an existing 1024-dimensional Pinecone index with OpenAI embeddings, set `OPENAI_EMBEDDING_DIMENSIONS=1024` in `.env` and rerun ingestion. Otherwise recreate the Pinecone index with the default model dimension, such as 1536 for `text-embedding-3-small`.

## Evaluations

Two evaluation scripts are shipped under `scripts/`. They produce CSV + JSON + PNG artifacts in `evaluation_v2/` so the deck and report always have v2-vs-research comparison plots.

```bash
# Retrieval (Pinecone + configured embeddings) vs research-stack baselines
python scripts/eval_v2_retrieval.py --mode mock        # always works
python scripts/eval_v2_retrieval.py --mode live --top-k 5

# End-to-end LangGraph agent (turn latency, citation rate, guardrail precision,
# ticket-creation success, route distribution) vs research-stack baselines
python scripts/eval_v2_agent.py --mode mock            # always works
python scripts/eval_v2_agent.py --mode live
```

`--mode mock` uses cached, conservative numbers based on documented production behavior so the charts render even without API keys (useful for slide builds in CI). `--mode live` requires `OPENAI_API_KEY` and `PINECONE_API_KEY`; if either is missing it falls back to mock and labels the JSON `"mode": "mock_fallback"`.

Outputs:

```
evaluation_v2/
├── v2_retrieval.csv             per-query latency + citation counts
├── v2_retrieval_summary.json    aggregate metrics + research-stack baselines
├── v2_retrieval_chart.png       latency and grounding comparison
├── v2_agent.csv                 per-turn route, latency, kb_ref_count
├── v2_agent_summary.json        turn aggregates + guardrail precision
└── v2_agent_chart.png           4-panel comparison vs Status 1 / Status 2 baselines
```

The headline v2 storyline (corroborated by `evaluation_v2/v2_agent_summary.json`):

- **End-to-end turn latency** is roughly an order of magnitude lower than the local Status 1 stack and ~2× faster than Status 2.
- **KB citation rate** on self-resolution turns stays high (the production agent still cites runbooks).
- **Guardrail precision** on prompt-injection prompts is `1.0` (every injection is blocked) with `0.0` false-block rate on benign IT prompts.
- **Ticket creation** success rate on the explicit-create scenarios is `1.0`.

These are the same charts referenced in `../CAPSTONE_FINAL_STORY.md` §21 and §23.
