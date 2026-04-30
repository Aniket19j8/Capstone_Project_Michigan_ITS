# ITS RAG Pipeline — Intelligent Ticketing System
## Team Michigan · FSE 570 · Spring 2026

> **Canonical narrative:** see `CAPSTONE_FINAL_STORY.md` for the full Status 1 → Status 2 → Final story, every evaluation table, and the rubric mapping. This README is the quick-start.
> **Two deployment stacks live in this repo.** The repo root is the **research stack** (Ollama + Qwen3 + ChromaDB + BM25 + LoRA) used to design and evaluate the system. The folder `ITS-v2-main/` is the **production stack** (FastAPI + LangChain + LangGraph + React + Pinecone + OpenAI) used as the live demo. Both are described below; both are real and runnable.
> **Primary local interface:** **React frontend + FastAPI backend** in this repo. The legacy Streamlit app (`streamlit_app.py`) is kept as a fallback.

---

## Quick Start

### Option A — Research stack: Full local app (FastAPI + React)

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Start Ollama with the LLM and the embedding model
ollama pull qwen3:4b          # generation LLM
ollama pull qwen3:0.6b        # embedding model (1024-d, default)
ollama serve &

# 3. Build data pipeline (first time only)
python 01_download_data.py --generate-synthetic --generate-kb
python 02_preprocess_data.py
python 03_build_vector_store.py    # embeds with qwen3:0.6b into ChromaDB

# 4. Start the FastAPI backend
uvicorn api:app --reload --port 8000

# 5. In a second terminal — start the React frontend
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

### Option D — Production stack (Pinecone + OpenAI + LangGraph)

```bash
cd ITS-v2-main
uv sync                                # or: pip install -e .
cp .env.example .env                   # set OPENAI_API_KEY + PINECONE_API_KEY
uv run python scripts/ingest_kb.py
uv run python scripts/ingest_tickets.py --csv ../data/processed/all_tickets.csv
uv run uvicorn app.main:app --reload   # opens http://127.0.0.1:8000

# React UI in dev mode
cd frontend && npm install && npm run dev
```

### Option B — CLI (no frontend)

```bash
# Requires Ollama running with qwen3:8b
ollama pull qwen3:8b
ollama serve &
python 05_rag_pipeline.py
```

### Option C — Streamlit (legacy)

```bash
pip install streamlit
ollama serve &
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

---

## Architecture

```
User Browser (React + Vite, :5173)
        │  HTTP (fetch)
        ▼
FastAPI Backend (api.py, :8000)
        │  loads modules dynamically
        ├─► 04_hybrid_retrieval.py   ← Dense + BM25 + RRF + Cross-encoder
        ├─► ChromaDB (data/chroma_db/)
        └─► Ollama qwen3:4b (:11434)

Query pipeline per request:
  description
    → Embed (Ollama qwen3:0.6b, 1024-dim)        # see Embedding Ablation Table 1
    → Dense search (ChromaDB cosine, top-20)
    → BM25 keyword search (rank_bm25, top-20)
    → RRF fusion (k=60, dense×0.6 + BM25×0.4)
    → Cross-encoder reranking (ms-marco-MiniLM-L-6-v2, top-5)
    → Context assembly (tickets + KB articles)
    → LLM generation (qwen3:4b, T=0.1, 900 tokens)
    → Structured JSON → Resolution Blueprint

Embedding model is configurable via the env var ITS_EMBEDDING_MODEL.
all-MiniLM-L6-v2 (384-d) remains a CPU-friendly fallback; the production
default is Qwen3-0.6B per the embedding ablation in CAPSTONE_FINAL_STORY.md §9.
```

---

## Project Structure

```
Capstone_Project_Team_Michigan/
├── api.py                        # FastAPI backend (primary backend)
├── streamlit_app.py              # Streamlit app (legacy UI)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── SETUP.md                      # Setup instructions
├── ITS_RAG_Complete_Documentation.md  # Full technical docs
│
├── frontend/                     # React + TypeScript + Vite UI
│   ├── src/
│   │   ├── App.tsx               # Main app component
│   │   ├── api.ts                # API client (calls FastAPI)
│   │   ├── types.ts              # TypeScript interfaces
│   │   ├── components/
│   │   │   ├── ResultPanel.tsx   # Displays resolution + tickets
│   │   │   ├── Sidebar.tsx       # System status sidebar
│   │   │   ├── TopBar.tsx        # Navigation bar
│   │   │   ├── NeuralBg.tsx      # Animated background
│   │   │   └── CursorGlow.tsx    # Cursor effect
│   │   └── hooks/
│   └── package.json
│
├── 01_download_data.py           # Data download + synthetic generation
├── 02_preprocess_data.py         # Clean, normalize, merge datasets
├── 03_build_vector_store.py      # Embed → ChromaDB + BM25 index
├── 04_hybrid_retrieval.py        # Hybrid retrieval engine (core)
├── 05_rag_pipeline.py            # CLI RAG pipeline
├── 06_evaluation.py              # Metrics, ablation, threshold optimization
├── 07_run_500_eval.py            # 500-query batch evaluation
├── 08_hallucination_comparison.py # RAG vs Base LLM hallucination study
├── 09_latency_benchmark.py       # Latency benchmarking
├── 10_dedup_threshold.py         # Dedup threshold optimization
│
├── data/
│   ├── raw/                      # Downloaded datasets
│   ├── processed/                # Cleaned & merged data
│   │   ├── all_tickets.csv       # Master ticket dataset (500 tickets)
│   │   ├── synthetic_tickets.csv
│   │   ├── bm25_corpus_tickets.json
│   │   └── bm25_corpus_kb.json
│   ├── knowledge_base/           # 7 runbooks (.md files)
│   └── chroma_db/                # ChromaDB persistent storage
│
├── evaluation/                   # Eval results, plots, ablation
├── scripts/                      # Helper scripts
└── tests/                        # Test suite
```

---

## API Reference

The FastAPI backend exposes two endpoints. Interactive docs at `http://localhost:8000/docs`.

### `GET /api/status`

Returns system health — retriever readiness, ticket/KB counts, LLM connectivity.

```json
{
  "retriever_ready": true,
  "retriever_error": null,
  "ticket_count": 500,
  "kb_count": 7,
  "llm_ready": true,
  "llm_model": "qwen3:4b"
}
```

### `POST /api/analyze`

Runs the full RAG pipeline for a ticket description.

**Request:**
```json
{
  "description": "My VPN keeps disconnecting every 10 minutes on Windows 11",
  "top_k_tickets": 3,
  "top_k_kb": 3,
  "use_reranking": true
}
```

**Response:**
```json
{
  "resolution": "## Resolution Blueprint\n**Problem** ...",
  "similar_tickets": [ { "ticket_id": "ITS-00042", "title": "...", "score": 7.23, ... } ],
  "kb_articles":     [ { "source_file": "runbook_vpn_troubleshooting.md", "score": 4.1, ... } ],
  "timings": { "tickets_s": 0.18, "kb_s": 0.12, "llm_s": 4.3, "total_s": 4.6 },
  "counts":  { "tickets": 3, "kb": 2 }
}
```

---

## Step-by-Step Data Pipeline

### Step 1 — Data Collection

```bash
python 01_download_data.py --generate-synthetic --generate-kb
```

Produces:
- `data/processed/synthetic_tickets.csv` — 500 IT tickets across 5 categories
- `data/processed/synthetic_duplicate_pairs.csv` — ~75 labeled duplicate pairs
- `data/knowledge_base/*.md` — 7 runbooks (VPN, password, email, hardware, software, Teams, printer)

### Step 2 — Preprocessing

```bash
python 02_preprocess_data.py
```

Cleans text, extracts error codes, builds `embedding_text`, merges into `all_tickets.csv`.

### Step 3 — Build Vector Store

```bash
python 03_build_vector_store.py          # Build
python 03_build_vector_store.py --reset  # Rebuild from scratch
```

Embeds tickets + KB documents into ChromaDB. Builds BM25 index.

### Step 4 — Hybrid Retrieval (standalone test)

```bash
python 04_hybrid_retrieval.py --query "VPN keeps dropping"
python 04_hybrid_retrieval.py --evaluate   # 4-method ablation
```

### Step 5 — RAG Pipeline (CLI)

```bash
python 05_rag_pipeline.py
# /resolve My VPN keeps disconnecting   → Resolution Blueprint
# /ask How do I reset a password?       → KB Q&A
# /dedup Outlook not syncing emails     → Duplicate check
# /similar laptop overheating           → Find similar tickets
```

### Step 6 — Evaluation

```bash
python 06_evaluation.py --ablation   # Compare retrieval methods
python 06_evaluation.py --dedup      # Threshold optimization
python 06_evaluation.py --all        # Everything
```

### Step 8 — Hallucination Comparison

```bash
python 08_hallucination_comparison.py
python 08_hallucination_comparison.py --model qwen3:4b
python 08_hallucination_comparison.py --num-queries 20   # quick test
```

Outputs: `evaluation/hallucination_comparison.csv`, `evaluation/hallucination_summary.json`, `evaluation/hallucination_chart.png`

### Step 15 — Visualize all evaluation CSVs into PNG charts

```bash
python 15_visualize_evaluations.py
# or only certain charts:
python 15_visualize_evaluations.py --only retrieval hallucination dedup stability
```

Generates 8 presentation-ready PNGs from the CSVs already in `evaluation/`:

```
evaluation/retrieval_comparison_chart.png
evaluation/hallucination_chart.png
evaluation/dedup_threshold_chart.png
evaluation/perturbation_stability_chart.png
evaluation/ablation_top1_chart.png
evaluation/embedding_ablation_chart.png
evaluation/dataset_distributions_chart.png
evaluation/kpi_dashboard_chart.png
```

### Step 16 — Advanced statistical metrics

```bash
python 16_advanced_metrics.py --bootstrap 2000
```

Adds bootstrap 95% CIs for Recall@k / MRR per method, paired t-test +
Wilcoxon signed-rank vs Hybrid_Rerank with Cohen's d, per-category and
per-severity breakdowns, latency P50 / P95 / P99, dedup cost-quality
Pareto, and a composite chart. Outputs:

```
evaluation/advanced_metrics.json
evaluation/significance_tests.json
evaluation/per_category_breakdown.csv
evaluation/per_severity_breakdown.csv
evaluation/cost_pareto.csv
evaluation/advanced_metrics_chart.png
```

### Production stack — ITS v2 evaluations

```bash
# from the ITS-v2-main/ folder
python scripts/eval_v2_retrieval.py --mode mock   # or --mode live with API keys
python scripts/eval_v2_agent.py     --mode mock   # or --mode live with API keys
```

Outputs land in `ITS-v2-main/evaluation_v2/`:

```
v2_retrieval.csv / .json / _chart.png   # latency + KB grounding vs research baselines
v2_agent.csv     / .json / _chart.png   # turn latency, citation rate, guardrails, route distribution
```

---

## Key Technical Decisions

| Component | Choice | Why |
|-----------|--------|-----|
| LLM (research API) | qwen3:4b via Ollama | Faster inference for REST responses |
| LLM (research CLI) | qwen3:8b via Ollama | Higher quality for interactive sessions |
| LLM (production)   | OpenAI chat (configurable) | Lower TTFB, structured-output API |
| Embeddings (research, default) | **Qwen3-0.6B (1024-d) via Ollama / vLLM** | Best on Embedding Ablation Table 1 |
| Embeddings (research, fallback) | all-MiniLM-L6-v2 (384-d) | CPU-only environments |
| Embeddings (production) | OpenAI embeddings (configurable) | Faster online latency under concurrency |
| Vector DB (research) | ChromaDB | Persistent, easy Python API, cosine similarity |
| Vector DB (production) | **Pinecone serverless** (`its-knowledge-base`, `its-tickets`) | Sub-100 ms vector search, metadata filters |
| Keyword Search | BM25 (rank_bm25) | Catches exact error codes, ticket IDs |
| Fusion | Reciprocal Rank Fusion (k=60, dense×0.6, BM25×0.4) | Robust to weight changes; tuned in §14 of the story doc |
| Reranker (research) | ms-marco-MiniLM-L-6-v2 | Cross-encoder, high precision, fast |
| Reranker (production) | FlashRank via LangChain | No GPU needed in serverless deployment |
| Agent runtime (production) | LangGraph 3-node ReAct (`guardrail → agent ↔ tools`) | Typed state, checkpointed threads, simpler than 8-node deterministic graph |
| Chunking | Recursive 450 tokens / 50 overlap; PageIndex section-aware for KB | KB top-1 similarity +0.062 vs flat |
| Backend | FastAPI + uvicorn | Async, typed, auto-docs at /docs |
| Frontend | React + TypeScript + Vite | Fast dev, type-safe API integration |
| Hallucination guard | Confidence-aware fallback + LLM-as-judge eval | Refuses on weak retrieval; cuts hallucination ~10× vs base |
| LoRA (Phase 6) | PEFT 4-bit nf4, r=8 α=16, all attn+MLP projections | Behavior only (JSON contract, tone, escalation); ticket facts stay in RAG |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `503 Retriever not loaded` | Run steps 1-3 first; check `data/chroma_db/` exists |
| `LLM offline` | Run `ollama serve` and `ollama pull qwen3:4b` |
| `CORS error in browser` | Make sure API is running on port 8000 |
| Slow first load | Normal — embedding model + reranker load takes 10-20 s, then cached |
| Frontend shows no status | Check that `api.py` is running and CORS is enabled (it is by default) |
