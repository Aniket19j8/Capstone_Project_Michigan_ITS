# ITS RAG — Setup Guide

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | Tested on 3.11 |
| Node.js | 18+ | For React frontend |
| Ollama | Latest | Local LLM serving |
| Git | Any | |

---

## 1. Python Environment

```bash
# From the project root
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2. Ollama + LLM

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh      # Linux/macOS
# Windows: download installer from https://ollama.com

# Pull the models
ollama pull qwen3:4b    # used by FastAPI backend (faster)
ollama pull qwen3:8b    # used by CLI pipeline (higher quality, optional)

# Start Ollama daemon
ollama serve &
```

Verify: `curl http://localhost:11434/api/tags`

---

## 3. Build the Data Pipeline (first time only)

```bash
# Generate synthetic tickets + knowledge base
python 01_download_data.py --generate-synthetic --generate-kb

# Preprocess and merge all sources
python 02_preprocess_data.py

# Embed into ChromaDB + build BM25 index
python 03_build_vector_store.py
```

Expected output:
- `data/processed/all_tickets.csv` (500 tickets)
- `data/processed/bm25_corpus_tickets.json`
- `data/processed/bm25_corpus_kb.json`
- `data/chroma_db/` (ChromaDB persistent store)

---

## 4. Start the FastAPI Backend

```bash
# From project root (with venv active and ollama running)
uvicorn api:app --reload --port 8000
```

Verify at `http://localhost:8000/api/status` — should return `retriever_ready: true`.

API docs: `http://localhost:8000/docs`

---

## 5. Start the React Frontend

```bash
cd frontend
npm install        # first time only
npm run dev        # starts at http://localhost:5173
```

The frontend calls the backend at `http://localhost:8000` by default (configured in `frontend/src/api.ts`).

---

## 6. (Optional) Run the Streamlit App

```bash
pip install streamlit
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

---

## File Requirements Checklist

Before starting the backend, verify:

```
data/
├── chroma_db/                   ← ChromaDB files (created by step 3)
├── processed/
│   ├── all_tickets.csv          ← created by step 2
│   ├── bm25_corpus_tickets.json ← created by step 3
│   └── bm25_corpus_kb.json      ← created by step 3
└── knowledge_base/
    ├── runbook_vpn_troubleshooting.md
    ├── runbook_password_reset.md
    ├── runbook_email_issues.md
    ├── runbook_hardware_laptop.md
    ├── runbook_software_installation.md
    ├── kb_article_teams_performance.md
    └── kb_article_printer_setup.md
```

---

## Troubleshooting

**Retriever fails to load**
- Make sure steps 1-3 completed without errors
- Check `data/chroma_db/` and `data/processed/` exist and are non-empty
- Re-run `python 03_build_vector_store.py --reset`

**LLM offline in `/api/status`**
- Run `ollama serve` if it is not already running
- Run `ollama pull qwen3:4b` if the model is not downloaded
- Verify with: `ollama list`

**Frontend CORS error**
- The backend allows `*` origins by default — no configuration needed
- Ensure the API is running on port 8000

**Slow first response**
- The embedding model (`all-MiniLM-L6-v2`) and reranker (`ms-marco-MiniLM-L-6-v2`) load on first request
- Subsequent requests are fast (models are cached in memory)
