# ITS RAG — Streamlit App Setup

## Quick Start (3 commands)

```bash
# 1. Install Streamlit (if not already)
pip install streamlit

# 2. Make sure Ollama is running
ollama serve &

# 3. Run the app
streamlit run streamlit_app.py
```

The app opens at **http://localhost:8501**

## File Structure Required

Place `streamlit_app.py` in your project root (same level as `scripts/` and `data/`):

```
your_project/
├── streamlit_app.py          ← This file
├── scripts/
│   ├── 04_hybrid_retrieval.py
│   ├── 05_rag_pipeline.py
│   └── ...
├── data/
│   ├── chroma_db/            ← ChromaDB files
│   ├── processed/
│   │   ├── bm25_corpus_tickets.json
│   │   ├── bm25_corpus_kb.json
│   │   └── ...
│   └── knowledge_base/
│       ├── runbook_vpn_troubleshooting.md
│       └── ...
└── configs/
```

## What the App Shows

### Tab 1: Submit Ticket (Full Auto Pipeline)
User types a ticket → system automatically runs:
1. Duplicate check (retrieval + LLM)
2. KB search (retrieval)
3. Resolution Blueprint generation (LLM)

### Tab 2: Ask Knowledge Base
User asks a question → retrieves KB articles → LLM generates grounded answer

### Tab 3: Find Similar Tickets
Pure retrieval — compare Dense vs BM25 vs Hybrid vs Hybrid+Rerank

### Tab 4: Ablation Study
Run the 4-method comparison on test queries, see results in a table

### Tab 5: Documentation
Complete technical docs — techniques, metrics, data specs, improvement ideas

## Troubleshooting

- **"Retriever failed"**: Make sure `data/chroma_db/` and `data/processed/` exist with the generated files
- **"LLM offline"**: Run `ollama serve` and ensure `qwen3:8b` is pulled
- **Slow first load**: Normal — loading embedding model + reranker takes 10-20 seconds on first run, then cached
