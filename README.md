# ITS RAG Pipeline — Complete Setup & Run Guide
## Intelligent Ticketing System - Team Michigan, FSE 570

---

## Quick Start (5 commands to get running)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install & start Ollama with your LLM
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:8b        # or: ollama pull llama3.1:8b
ollama serve &               # runs in background

# 3. Download data & generate synthetic tickets + knowledge base
cd scripts
python 01_download_data.py --generate-synthetic --generate-kb

# 4. Preprocess & build vector store
python 02_preprocess_data.py
python 03_build_vector_store.py

# 5. Run the RAG pipeline
python 05_rag_pipeline.py
```

---

## Project Structure

```
its_rag/
├── requirements.txt              # All Python dependencies
├── README.md                     # This file
├── configs/
│   └── config.env                # All configurable parameters
├── scripts/
│   ├── config_loader.py          # Config utility
│   ├── 01_download_data.py       # Data download + synthetic generation
│   ├── 02_preprocess_data.py     # Clean, normalize, merge all datasets
│   ├── 03_build_vector_store.py  # Embed → ChromaDB + BM25 index
│   ├── 04_hybrid_retrieval.py    # Hybrid retrieval engine (the core)
│   ├── 05_rag_pipeline.py        # Full RAG: retrieval → LLM → response
│   ├── 06_evaluation.py          # Metrics, ablation, threshold optimization
│   ├── fetch_jira.py             # (generated) Apache JIRA API fetcher
│   └── fetch_github_issues.py    # (generated) GitHub Issues fetcher
├── data/
│   ├── raw/                      # Downloaded datasets
│   ├── processed/                # Cleaned & merged data
│   ├── knowledge_base/           # Runbooks & KB articles (.md files)
│   └── chroma_db/                # ChromaDB persistent storage
└── evaluation/                   # Eval results, plots, ablation
```

---

## Step-by-Step Guide

### Step 1: Data Collection

```bash
# Generate synthetic tickets (500) + knowledge base (7 runbooks)
python 01_download_data.py --generate-synthetic --generate-kb

# Fetch real Apache JIRA tickets (needs internet)
python scripts/fetch_jira.py

# Fetch GitHub Issues (needs internet, optionally set GITHUB_TOKEN)
export GITHUB_TOKEN=ghp_your_token
python scripts/fetch_github_issues.py
```

**What you get:**
- `data/processed/synthetic_tickets.csv` — 500 IT tickets across 5 categories
- `data/processed/synthetic_duplicate_pairs.csv` — ~75 labeled duplicate pairs
- `data/knowledge_base/*.md` — 7 runbooks (VPN, password, email, hardware, software, Teams, printer)
- `data/raw/jira_issues.csv` — Real Apache JIRA tickets (if fetched)
- `data/raw/github_issues.csv` — Real GitHub issues (if fetched)

### Step 2: Preprocessing

```bash
python 02_preprocess_data.py
```

**What it does:**
- Cleans text (removes URLs, paths, normalizes whitespace)
- Extracts error codes from descriptions
- Computes quality scores per ticket
- Creates `embedding_text` field (optimized for semantic search)
- Merges all sources into `all_tickets.csv`
- Generates `data_stats.json` with full statistics

### Step 3: Build Vector Store

```bash
python 03_build_vector_store.py          # Build
python 03_build_vector_store.py --reset  # Rebuild from scratch
```

**What it does:**
- Loads `all-MiniLM-L6-v2` embedding model (384 dimensions)
- Embeds all tickets → ChromaDB `its_tickets` collection
- Chunks KB documents (section-aware, 450 tokens, 50 overlap)
- Embeds KB chunks → ChromaDB `its_knowledge_base` collection
- Builds BM25 tokenized corpus for keyword search
- Runs verification queries to confirm everything works

### Step 4: Hybrid Retrieval

```bash
# Interactive search
python 04_hybrid_retrieval.py

# Single query
python 04_hybrid_retrieval.py --query "VPN keeps dropping"

# Ablation comparison (dense vs BM25 vs hybrid vs hybrid+rerank)
python 04_hybrid_retrieval.py --evaluate
```

**Pipeline: Query → Dense (ChromaDB) + BM25 → RRF Fusion → Cross-Encoder Rerank → Top-K**

### Step 5: RAG Pipeline

```bash
# Interactive mode
python 05_rag_pipeline.py

# Commands in interactive mode:
#   /resolve My VPN keeps disconnecting    → Resolution Blueprint
#   /ask How do I reset a password?         → KB Q&A
#   /dedup Outlook not syncing emails       → Duplicate check
#   /similar laptop overheating             → Find similar tickets
```

**Requires Ollama running:** `ollama serve`

### Step 6: Evaluation

```bash
python 06_evaluation.py --ablation   # Compare retrieval methods
python 06_evaluation.py --dedup      # Threshold optimization
python 06_evaluation.py --all        # Everything
```

**Outputs:** `evaluation/ablation_results.csv`, `evaluation/dedup_threshold_analysis.png`

---

## Key Technical Decisions

| Component | Choice | Why |
|-----------|--------|-----|
| LLM | Qwen3-8B via Ollama | Best JSON extraction + multi-turn at 8B size |
| Embeddings | all-MiniLM-L6-v2 | Fast, 384-dim, proven for semantic search |
| Vector DB | ChromaDB | Persistent, easy Python API, cosine similarity |
| Keyword Search | BM25 (rank_bm25) | Catches exact error codes, IDs |
| Fusion | Reciprocal Rank Fusion | Standard method, no tuning needed |
| Reranker | ms-marco-MiniLM-L-6-v2 | Cross-encoder, high precision, fast |
| Chunking | Recursive, 450 tokens | 85-90% recall in benchmarks |
| Orchestration | Direct Python (can add LangChain) | Simpler for MVP |

---

## For Status 1 Presentation (March 5)

Show these working:
1. **Data stats** — run `02_preprocess_data.py`, show the stats JSON
2. **Vector search demo** — run `03_build_vector_store.py`, show verification queries
3. **Hybrid vs Dense comparison** — run `04_hybrid_retrieval.py --evaluate`
4. **RAG resolution demo** — run `05_rag_pipeline.py`, show a Resolution Blueprint
5. **Evaluation framework** — show metrics defined in `06_evaluation.py`
