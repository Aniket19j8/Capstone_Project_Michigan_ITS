# Intelligent Ticketing System (ITS)

Hey — this is our capstone for **FSE 570**, **Spring 2026**, **Team Michigan**.

**GitHub:** [github.com/Aniket19j8/Capstone_Project_Michigan_ITS](https://github.com/Aniket19j8/Capstone_Project_Michigan_ITS)  
Clone: `git clone https://github.com/Aniket19j8/Capstone_Project_Michigan_ITS.git`

We got tired of the usual story: helpdesks drown in tickets that are basically repeats, plain LLMs invent error codes, and triage sends work to the wrong queue. So we built a retrieval-augmented assistant: grab similar past tickets and real runbooks, cite them, and only then let the model speak. We ran the thing end-to-end on scaled data, measured recall, hallucinations, latency, dedup thresholds, and even a small LoRA pass for answer *shape* (not for memorizing tickets).

There are **two** runnable projects in this repo. Same idea, different deployment:

1. **Root folder** — the stack we used for research and class demos: **Ollama** (Qwen3 chat + Qwen3-0.6B embeddings), **ChromaDB**, **BM25**, **RRF**, **cross-encoder rerank**, optional **LoRA**. Numbered scripts `01_*` … `16_*`, plus **FastAPI** (`api.py`) and a **React** UI under `frontend/`.
2. **`ITS-v2-main/`** — the “ship it” version: **FastAPI**, **LangChain / LangGraph** agent, **Pinecone**, **OpenAI** (embeddings + chat), **React** SPA, SQLite for app state. Same product narrative, faster cloud path.

If you just want something pixels-on-screen in five minutes, start with **Streamlit** (`streamlit_app.py`). It was our first UI; the README below still points there on purpose.

---

## Google Drive (manual data — we don’t auto-download the capstone export)

Shared folder (download what you need and merge into this repo’s `data/` tree):

**[Team data folder on Google Drive](https://drive.google.com/drive/folders/1ov-D3L0DOdmUN7Xfq2jrB7tJwGFUq25J?usp=sharing)**

What’s in there (as of our upload):

| In Drive | Put locally under |
| --- | --- |
| `processed/` | `data/processed/` (CSVs, manifests — includes the big ticket export) |
| `chroma_db/` | `data/chroma_db/` (optional: skip re-embedding if you use this snapshot; `.gitignore` ignores it in git) |
| `knowledge_base/` | `data/knowledge_base/` |
| `lora/` | `data/lora/` (LoRA artifacts / manifests if you run Phase 6) |

Run `python 01_download_data.py` if you still want synthetic stubs, or go straight to `02_preprocess_data.py` / `03_build_vector_store.py` depending on what you pulled.

**Can someone with “Viewer” download?** Usually **yes**: people shared as **Viewer** can open files and use **Download** (or download a whole folder as a zip), unless the owner enabled **“Viewers cannot download, print, or copy”** on those files. If someone only sees “Request access,” widen the link to **Anyone with the link** or add them to the share.

---

## Who built this

**Aniket, Anu, Kaushil, Tanmay, Tushar** — Team Michigan, MS Data Science, Wayne State (capstone spring ‘26). We split pipelines, evals, the v2 port, and the UI work between us; the README is mine in voice but the repo is everybody’s.

---

## Why we built it (short)

We needed something defensible for the rubric: **heterogeneous data**, **measurable retrieval**, **honest hallucination numbers**, and a line to production. The root stack is the proof. `ITS-v2-main` is what you’d show if someone asks “okay but can this live in the real world with auth and an agent?”

---

## Repo layout (what matters)

```
.
├── streamlit_app.py          # quickest demo (after Ollama + Chroma are built)
├── api.py                    # FastAPI backend for the React app at frontend/
├── 01_download_data.py       # local-first data prep; no surprise network
├── 02_preprocess_data.py     # clean / merge / stats
├── 03_build_vector_store.py  # embed → Chroma + BM25 corpora
├── 04_hybrid_retrieval.py    # dense + BM25 + RRF + rerank
├── 05_rag_pipeline.py        # CLI RAG
├── 06–16_*.py                # evals, benchmarks, LoRA prep/train, viz, stats
├── department_mapping.py     # triage-style department hints
├── its_*.py                  # small helpers used by routes/brain
├── frontend/                 # Vite + React for the root FastAPI API
├── data/
│   ├── raw/                  # optional public mirrors if you opt in in 01
│   ├── processed/            # CSVs, manifests, BM25 JSON
│   ├── knowledge_base/       # markdown runbooks / articles
│   └── chroma_db/            # local vector store (gitignored — rebuild with 03)
├── evaluation/               # CSV/JSON/plots from the numbered scripts
├── scripts/                  # fetch helpers (Jira, GitHub, etc.) if you want them
├── tests/
├── requirements.txt
├── requirements-lora.txt     # extra deps if you run the LoRA scripts
└── ITS-v2-main/              # second project (Pinecone + OpenAI + LangGraph)
```

---

## Requirements

**Root stack**

```bash
pip install -r requirements.txt
```

Optional: `pip install datasets` only if you run `01_download_data.py --with-public-datasets`. Ollama is separate — install from ollama.com and pull models (see below).

**LoRA phase**

```bash
pip install -r requirements-lora.txt
```

**Root React frontend**

```bash
cd frontend && npm install
```

**ITS v2** (see `ITS-v2-main/README.md` — uses `uv` or pip install there)

---

## How to start — project A: root stack (research + FastAPI + React)

Typical first-time flow:

```bash
# Ollama — LLM + embedder (1024-d default for qwen3:0.6b)
ollama pull qwen3:4b
ollama pull qwen3:0.6b
ollama serve

# Data: register your Drive CSV if you have it; always safe locally (no HF unless you ask)
python 01_download_data.py

python 02_preprocess_data.py
python 03_build_vector_store.py
```

**FastAPI + React (what we treated as the “main” app for the class UI):**

```bash
uvicorn api:app --reload --port 8000
# other terminal
cd frontend && npm run dev
# UI → http://localhost:5173   API docs → http://localhost:8000/docs
```

**Streamlit (first paint, super simple):**

```bash
pip install streamlit
streamlit run streamlit_app.py
# http://localhost:8501
```

**CLI RAG**

```bash
python 05_rag_pipeline.py
```

---

## How to start — project B: `ITS-v2-main` (cloud-shaped)

From the subfolder:

```bash
cd ITS-v2-main
uv sync
cp .env.example .env
# add OPENAI_API_KEY, PINECONE_API_KEY, etc.

uv run python scripts/ingest_kb.py
uv run python scripts/ingest_tickets.py --csv ../data/processed/all_tickets.csv
uv run uvicorn app.main:app --reload

cd frontend && npm install && npm run dev
```

Details and admin URLs: `ITS-v2-main/README.md`.

---

## Data (what we actually used)

- **Unified heterogeneous tickets**: Jira-style exports, GitHub issues, Stack-style text, synthetic rows for stress tests, then larger “enterprise-shaped” CSVs scaled toward **~80k** rows for the final Chroma / eval story.
- **Knowledge base**: markdown runbooks under `data/knowledge_base/` (VPN, password, email, hardware, etc.) plus a few article-style files; preprocessing + chunking happen in `02` / `03`.
- **Dedup labels**: small synthetic duplicate pair files for threshold sweeps.
- **LoRA JSON splits**: built from filtered resolved tickets — behavior training only; facts stay in RAG.

No secrets in the repo: big CSVs and chroma dirs stay local or come from Drive.

---

## Models (high level)

| Piece | Root / research | ITS v2 |
| --- | --- | --- |
| Chat | Qwen3 via Ollama (e.g. `qwen3:4b`) | OpenAI chat (configurable) |
| Embeddings | Qwen3-0.6B via Ollama (default) | OpenAI embeddings |
| Vectors | ChromaDB | Pinecone |
| Keyword leg | BM25 + RRF with dense | Handled in LC tooling / indexes |
| Rerank | cross-encoder (sentence-transformers) | FlashRank-style path in v2 |
| Optional fine-tune | LoRA on JSON / tone (see `12–14`) | n/a in tree as shipped |

---

## Architecture (two lines)

**Root:** browser → React → `api.py` → `04_hybrid_retrieval` (Chroma + BM25 + rerank) → Ollama for generation → JSON-ish blueprint back to UI.

**v2:** browser → React → FastAPI → LangGraph agent (`guardrail → agent ↔ tools`) → Pinecone + SQLite; guardrails strip sensitive patterns on the way out.

More detail: `ITS-v2-main/docs/architecture.md`.

---

## Results (headline numbers we stood behind)

These came from the evaluation folder and class slides — your machine may vary slightly, but the ballpark is what we reported:

- **Retrieval:** recall@5 around **0.99** band, MRR high (**~0.99**) on the hybrid + rerank setup at scale.
- **Hallucinations:** RAG grounded runs much lower judged hallucination rate than “base LLM alone” (we used an LLM-as-judge protocol — see `08_hallucination_comparison.py` and outputs under `evaluation/`).
- **Dedup:** best threshold landed near **τ ≈ 0.78** with strong F1 on our labeled pairs.
- **Latency:** mean end-to-end in the hundreds of ms on the tuned stack (see `09_latency_benchmark.py`); v2 targets fast cloud turns.
- **Embeddings:** ablations favored **Qwen3-0.6B** (1024-d) for our ticket domain.

Open the CSVs and PNGs in `evaluation/` if you want the receipts.

---

## Recommendations (if you deploy for real)

- Keep **grounding + citations** non-negotiable; if retrieval confidence is weak, **escalate** instead of improvising.
- Run **dedup** before assigning engineers — saves a lot of duplicate investigations.
- **Redact** before cloud LLMs if you use v2 with sensitive tenants; we put guardrails in v2 for a reason.
- Revisit **thresholds** on your own ticket distribution; ours are a data point, not universal law.

---

## Future work

- Stronger **multilingual** evals (we had mixed language tags in the big export).
- **Online learning** from analyst thumbs up/down without poisoning the vector store.
- **cheaper/local v2** profiles (smaller OpenAI models or routed Ollama) for cost sensitivity.
- Deeper **security** review on tool-calling paths in the agent.

---

## Eval / extras

Examples:

```bash
python 06_evaluation.py --ablation
python 08_hallucination_comparison.py
python 15_visualize_evaluations.py
python 16_advanced_metrics.py --bootstrap 2000
```

v2 eval scripts live under `ITS-v2-main/scripts/`.

---

That’s the whole picture. Clone it, drop the Drive files, run `01` → `03`, pick Streamlit or FastAPI+React, and poke `evaluation/` if you’re grading us. Good luck out there.
