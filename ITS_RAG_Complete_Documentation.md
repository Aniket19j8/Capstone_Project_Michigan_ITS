# ITS RAG Pipeline — Complete Technical Documentation
## Intelligent Ticketing System · Team Michigan · FSE 570 Spring 2026

> **Branch:** `Feature_hallucination_and_Frontend_updated_03_04_2026`
> Major additions in this branch: FastAPI backend (`api.py`), React + TypeScript + Vite frontend (`frontend/`), and hallucination comparison study (`08_hallucination_comparison.py`).

---

## 1. System Overview

### 1.1 What Was Built

A full-stack RAG (Retrieval-Augmented Generation) system for IT help-desk ticket triage with three interface layers:

| Interface | File | Port | Status |
|-----------|------|------|--------|
| React frontend | `frontend/` | 5173 | **Active (this branch)** |
| FastAPI backend | `api.py` | 8000 | **Active (this branch)** |
| Streamlit app | `streamlit_app.py` | 8501 | Legacy (still functional) |
| CLI pipeline | `05_rag_pipeline.py` | — | Active |

### 1.2 End-to-End Flow

```
User types IT issue description (browser)
  → POST /api/analyze (FastAPI)
      → Embed query (all-MiniLM-L6-v2, 384-dim)
      → Dense search: ChromaDB cosine similarity, top-20 tickets
      → BM25 search: keyword matching, top-20 tickets
      → RRF fusion: merge rankings (k=60, dense×0.6, BM25×0.4)
      → Cross-encoder reranking: ms-marco-MiniLM-L-6-v2, keep top-5
      → Same pipeline for KB: ChromaDB kb collection, keep top-3
      → Relevance filter: drop docs with rerank_score < -2.0
      → Context assembly: format tickets + KB as structured text
      → LLM prompt: qwen3:4b via Ollama (T=0.1, 900 tokens)
      → JSON response: parse blueprint fields
      → Normalize + render blueprint as Markdown
  ← JSON response to browser
      → ResultPanel renders resolution, tickets, KB, email draft
```

### 1.3 When the LLM Is Used

| Operation | Uses LLM? | Model |
|-----------|-----------|-------|
| Dense vector search | No | Embedding model only |
| BM25 keyword search | No | Pure token matching |
| RRF fusion | No | Math formula |
| Cross-encoder reranking | No | Separate small model |
| Resolution blueprint | Yes | qwen3:4b (API) / qwen3:8b (CLI) |
| Duplicate judgment | Yes | qwen3:4b / qwen3:8b |
| KB Q&A | Yes | qwen3:4b / qwen3:8b |
| Hallucination scoring | Yes | qwen3:4b (LLM-as-judge) |

---

## 2. FastAPI Backend (`api.py`)

### 2.1 Endpoints

#### `GET /api/status`

Returns system health. Called by the frontend sidebar on page load and on an interval.

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

#### `POST /api/analyze`

Runs the full pipeline. Request body:

```json
{
  "description": "My VPN keeps disconnecting every 10 minutes on Windows 11",
  "top_k_tickets": 3,
  "top_k_kb": 3,
  "use_reranking": true
}
```

Response includes:
- `resolution` — Markdown-formatted blueprint
- `similar_tickets` — scored ticket list
- `kb_articles` — scored KB article list
- `timings` — per-stage latency breakdown
- `counts` — how many tickets/KB docs were used

Interactive docs: `http://localhost:8000/docs`

### 2.2 Blueprint JSON Schema

The LLM is instructed to return this exact schema. The backend normalizes, validates, and fills in defaults if the LLM output is partial or malformed.

```json
{
  "problem": "1-2 sentence plain-language restatement",
  "resolution": ["paragraph 1 — cause analysis", "paragraph 2 — recommended fix"],
  "troubleshooting_steps": [
    { "step": "Short title", "details": "Full instructions" }
  ],
  "similar_tickets": [
    { "ticket_id": "ITS-00042", "component": "VPN", "status": "Resolved", "summary": "..." }
  ],
  "similar_ticket_resolutions": [
    { "ticket_id": "ITS-00042", "resolution": "..." }
  ],
  "knowledge_base": [
    { "source": "runbook_vpn_troubleshooting.md", "guidance": "Human-readable instructions" }
  ],
  "escalate_if": [
    { "condition": "When to escalate", "owner": "Network Team" }
  ],
  "user_email": {
    "subject": "Short subject",
    "body": "Full self-contained email with troubleshooting steps"
  }
}
```

### 2.3 Hallucination Guards in the API

| Guard | Implementation |
|-------|---------------|
| Relevance threshold | Docs with `rerank_score < -2.0` (tickets) or `< -5.0` (KB) are excluded from context |
| Thinking-token stripping | `<think>...</think>` tokens and chain-of-thought phrases removed before JSON parse |
| JSON extraction | Robust scanner finds first `{` and attempts decode — handles wrapped or noisy LLM output |
| Fallback blueprint | If LLM returns invalid JSON, a structured fallback is built from raw retrieval results |
| Default email | If the LLM omits the email field, a templated professional email is generated |

---

## 3. React Frontend (`frontend/`)

### 3.1 Tech Stack

| Layer | Choice |
|-------|--------|
| Framework | React 18 |
| Language | TypeScript |
| Build tool | Vite |
| Styling | CSS modules + custom CSS |
| HTTP | Native `fetch` |

### 3.2 Key Files

| File | Purpose |
|------|---------|
| `src/App.tsx` | Main component — form, state, submit handler |
| `src/api.ts` | API client — `getStatus()`, `analyze()` |
| `src/types.ts` | TypeScript interfaces for all API shapes |
| `src/components/ResultPanel.tsx` | Renders the full blueprint |
| `src/components/Sidebar.tsx` | System status display |
| `src/components/TopBar.tsx` | App header/navigation |
| `src/components/NeuralBg.tsx` | Animated canvas background |
| `src/components/CursorGlow.tsx` | Cursor glow visual effect |

### 3.3 TypeScript Interfaces (`types.ts`)

```ts
interface SystemStatus {
  retriever_ready: boolean;
  retriever_error: string | null;
  ticket_count: number | null;
  kb_count: number | null;
  llm_ready: boolean;
  llm_model: string;
}

interface SimilarTicket {
  id: string; text: string; score: number;
  title: string; category: string; severity: string;
  status: string; component: string; ticket_id: string;
  source_file: string;
}

interface KbArticle {
  id: string; text: string; score: number;
  title: string; source_file: string;
}

interface AnalyzeResult {
  resolution: string;
  similar_tickets: SimilarTicket[];
  kb_articles: KbArticle[];
  timings: { tickets_s: number; kb_s: number; llm_s: number; total_s: number };
  counts: { tickets: number; kb: number };
}
```

---

## 4. Retrieval Pipeline — Full Specifications

### 4.1 Embedding

| Parameter | Value |
|-----------|-------|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Dimensions | 384 |
| Max sequence length | 256 tokens |
| Similarity metric | Cosine |
| Training data | Pre-trained on 1B+ sentence pairs |

### 4.2 Vector Store (ChromaDB)

| Parameter | Value |
|-----------|-------|
| Database | ChromaDB (persistent, SQLite-backed) |
| Index type | HNSW (Hierarchical Navigable Small World) |
| Distance function | Cosine |
| Ticket collection | `its_tickets` (500 vectors) |
| KB collection | `its_knowledge_base` (7 vectors) |

### 4.3 BM25 Keyword Search

| Parameter | Value |
|-----------|-------|
| Algorithm | BM25 Okapi (`rank_bm25`) |
| Tokenization | Lowercase, alphanumeric split, stopword removal |
| Stopwords removed | 80+ common English words |
| Corpus | 500 ticket documents + 7 KB documents |

### 4.4 Reciprocal Rank Fusion (RRF)

| Parameter | Value |
|-----------|-------|
| Formula | `score(d) = Σ weight_i / (k + rank_i)` |
| k constant | 60 |
| Dense weight | 0.6 |
| BM25 weight | 0.4 |
| Candidates (each source) | Top 20 |

### 4.5 Cross-Encoder Reranking

| Parameter | Value |
|-----------|-------|
| Model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Input | (query, document) pairs |
| Candidates reranked | Top 20 from RRF |
| Final results | Top 5 |
| Approximate speed | ~50 ms per pair |

### 4.6 Knowledge Base Chunking

| Parameter | Value |
|-----------|-------|
| Method | Recursive character splitting, section-aware |
| Target chunk size | 450 tokens |
| Overlap | 50 tokens |
| Section detection | Markdown headers (##, ###) |
| Result | 7 chunks (one per KB document) |

### 4.7 LLM Configuration

| Parameter | API backend | CLI pipeline |
|-----------|-------------|--------------|
| Model | qwen3:4b | qwen3:8b |
| Serving | Ollama (:11434) | Ollama (:11434) |
| Temperature | 0.1 | 0.1 |
| Max tokens | 900 | 2048 |
| Think mode | Disabled (`think: false`) | Disabled |
| Context window | 128K | 128K |

---

## 5. Hallucination Comparison (`08_hallucination_comparison.py`)

### 5.1 Design

The script compares two generation modes on the same set of queries:

| Mode | Description |
|------|-------------|
| Base LLM | No retrieval context — model answers from training data only |
| RAG | Full pipeline — retrieved tickets + KB injected into prompt |

An LLM-as-judge then scores each answer on three dimensions (0–10):

| Dimension | Definition |
|-----------|------------|
| Groundedness | Is the answer supported by the provided context (or correctly says "I don't know")? |
| Specificity | Does it give actionable, specific guidance rather than vague advice? |
| Accuracy | Is the answer factually correct for IT support? |

### 5.2 Key Design Decisions

**Relevance threshold before context injection:** If retrieved docs have `rerank_score < MIN_RERANK_SCORE`, the context is withheld entirely. This prevents the LLM from receiving irrelevant IT tickets for out-of-scope queries (e.g., weather, cooking) and generating hallucinated IT-sounding answers.

**Separate in-scope vs out-of-scope reporting:** Aggregate stats are not dragged down by out-of-scope queries. The judge is explicitly told that "I don't have information about this" is a *correct* answer for OOS queries.

**Thinking-token stripping:** `<think>...</think>` blocks from qwen3 are removed before JSON parsing.

### 5.3 Outputs

```
evaluation/
├── hallucination_comparison.csv    # Per-query scores (base vs RAG)
├── hallucination_summary.json      # Aggregate statistics
└── hallucination_chart.png         # Bar chart comparison
```

### 5.4 Usage

```bash
python 08_hallucination_comparison.py
python 08_hallucination_comparison.py --model qwen3:4b
python 08_hallucination_comparison.py --num-queries 20   # quick test
```

---

## 6. Data Specifications

### 6.1 Ticket Data (500 tickets)

| Field | Description |
|-------|-------------|
| `ticket_id` | Unique identifier (e.g., ITS-00042) |
| `title` | Short issue summary |
| `description` | User's natural language description |
| `category` | Incident, Network, Account, Software, Hardware |
| `component` | VPN, Outlook, Laptop, MFA, Teams, etc. |
| `severity` | Critical, High, Medium, Low |
| `status` | Open, In Progress, Resolved, Closed |
| `resolution` | How it was fixed (48% of tickets have this) |
| `environment` | Windows 11, macOS 14, Ubuntu 22.04, etc. |
| `assigned_team` | IT Operations, Network Engineering, Help Desk, etc. |
| `embedding_text` | Optimized concatenation for vector embedding |

**Distribution:**
- Categories: Incident 108 · Network 105 · Account 98 · Software 95 · Hardware 94
- Severities: Medium 247 · Low 141 · High 87 · Critical 25
- Tickets with resolution: 241 (48%)
- Labeled duplicate pairs: ~69

### 6.2 Knowledge Base (7 documents)

| File | Topics |
|------|--------|
| `runbook_vpn_troubleshooting.md` | Connection drops, MTU, split tunneling |
| `runbook_password_reset.md` | AD lockout, MFA, identity verification |
| `runbook_email_issues.md` | Outlook sync, cache, profiles, shared mailboxes |
| `runbook_hardware_laptop.md` | BSOD codes, overheating, boot failures |
| `runbook_software_installation.md` | Licenses, permissions, disk space |
| `kb_article_teams_performance.md` | Cache clearing, memory reduction |
| `kb_article_printer_setup.md` | Queue stuck, offline, setup steps |

---

## 7. Evaluation Results

### 7.1 Retrieval Ablation

| Query | Dense | BM25 | Hybrid | Hybrid+Rerank |
|-------|-------|------|--------|---------------|
| "My computer won't start" | ✅ Laptop | ❌ Salesforce | ✅ Laptop | ✅ Laptop |
| "error code 0x80070005" | ✅ Hex match | ✅ Hex match | ✅ Hex match | ✅ Hex match |
| "Cannot access resources remotely" | ✅ VPN | ✅ Firewall | ✅ Firewall | ✅ VPN |
| "Outlook not syncing after update" | ✅ | ✅ | ✅ | ✅ |
| "Teams consuming too much memory" | ✅ | ✅ | ✅ | ✅ |
| "account locked need reset MFA" | ✅ | ✅ | ✅ | ✅ |
| "BSOD IRQL_NOT_LESS_OR_EQUAL" | ~ partial | ❌ | ~ partial | ~ partial |
| "printer offline error" | ✅ | ✅ | ✅ | ✅ |

**Key findings:**
- BM25 alone fails on semantic queries ("My computer won't start" → returned Salesforce)
- Dense alone misses keyword-specific matches (version numbers, error codes)
- Hybrid consistently matches or beats either method alone
- Reranking helps on ambiguous queries but needs more data for consistent gains

### 7.2 Resolution Quality

| Test | Result |
|------|--------|
| `/ask print queue stuck` | Correct 4-step fix from KB, cited source |
| `/dedup Outlook not syncing` | Correctly identified as RELATED (not duplicate) with reasoning |
| Out-of-scope: "toilet flush not working" | Correctly refused IT context, general advice only |
| Out-of-scope: "truck ran over me" | Correctly redirected to medical help |
| `/similar laptop BSOD` | 5 relevant BSOD tickets ranked correctly |

---

## 8. Goal Alignment

**Goal:** Move engineers' focus from problem reporting to problem resolution.

| Before ITS | After ITS |
|------------|-----------|
| Manually search past tickets | Instant retrieval of similar tickets |
| Read through runbooks | Relevant KB sections surfaced automatically |
| Write resolution from scratch | LLM generates Resolution Blueprint |
| Duplicate tickets waste triage time | Duplicates flagged before assignment |
| New issues have no starting point | Full KB support for unseen issues |
| Knowledge lives in engineers' heads | Captured in KB, accessible via RAG |

---

## 9. How to Extend

### Add More Ticket Data

```bash
python scripts/fetch_jira.py             # Apache JIRA (needs internet)
export GITHUB_TOKEN=ghp_your_token
python scripts/fetch_github_issues.py    # GitHub Issues
python 02_preprocess_data.py
python 03_build_vector_store.py --reset
```

### Add New Knowledge Base Documents

Create a `.md` file in `data/knowledge_base/` following the existing runbook format (markdown headers, numbered steps). Then rebuild the vector store:

```bash
python 03_build_vector_store.py --reset
```

No code changes needed — the pipeline picks up new KB files automatically.

### Add a New Domain

1. Create KB documents for the domain in `data/knowledge_base/`
2. Create domain-specific tickets in the same CSV format as `synthetic_tickets.csv`
3. Re-run scripts 02 and 03

### Potential Improvements

| Improvement | Expected Impact |
|-------------|----------------|
| Query expansion (LLM rewrites vague queries) | +10-15% recall |
| Metadata filtering in ChromaDB | More category-specific results |
| Fine-tune LLM on ticket JSON format (QLoRA) | +10-15% JSON reliability |
| Feedback loop (resolved tickets → KB) | Knowledge grows over time |
| Scale ticket corpus (500 → 5000+) | Better duplicate detection |
| Authentication on API | Required for production deployment |
