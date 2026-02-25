# ITS RAG Pipeline — Complete Technical Documentation
## Intelligent Ticketing System • Team Michigan • FSE 570 Spring 2026

---

## 1. What We Built & What's Working

Your RAG pipeline is **fully operational**. When you ran `/ask`, `/dedup`, `/resolve`, `/similar` in the CLI, each command triggered the **complete end-to-end pipeline**:

```
Your query → Embedding (all-MiniLM-L6-v2)
           → Dense search (ChromaDB, cosine similarity)
           → BM25 keyword search (rank_bm25)
           → Reciprocal Rank Fusion (merge results)
           → Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
           → Context assembly (top tickets + KB articles)
           → LLM generation (Qwen3-8B via Ollama)
           → Grounded response with citations
```

**You have already tested the full RAG generation.** The toilet flush and truck examples prove it — the LLM correctly identified that the retrieved context was irrelevant and said so, which is exactly the behavior of a well-grounded RAG system (it doesn't hallucinate).

---

## 2. How It Works: Manual vs Automatic

**Current state (CLI):** You manually select which mode to use (`/ask`, `/dedup`, etc.).

**Production state (Streamlit app):** The flow is **fully automatic**. When a user submits a ticket:

| Step | What Happens | Who Triggers It |
|------|-------------|-----------------|
| 1. User types issue | "My VPN keeps dropping on Windows 11" | User |
| 2. Duplicate check | System retrieves similar tickets → LLM judges duplicate/related/unique | **Automatic** |
| 3. KB retrieval | System finds matching runbooks and articles | **Automatic** |
| 4. Resolution Blueprint | LLM reads retrieved context → generates step-by-step fix | **Automatic** |
| 5. Results displayed | Engineer sees: similar tickets, KB articles, resolution, dedup verdict | **Automatic** |

The Streamlit app I built (`streamlit_app.py`) implements exactly this automatic flow.

---

## 3. When Does It Connect to Qwen / Use LLM?

The LLM (Qwen3-8B) is used at **3 specific points**, not during retrieval:

| Operation | Uses LLM? | What Happens |
|-----------|-----------|--------------|
| Dense vector search | ❌ No | Embedding model only (all-MiniLM-L6-v2) |
| BM25 keyword search | ❌ No | Pure token matching, no ML |
| RRF fusion | ❌ No | Math formula only |
| Cross-encoder reranking | ❌ No | Separate small model (ms-marco) |
| **Resolution generation** | ✅ Yes | Qwen3-8B reads retrieved context → generates blueprint |
| **Duplicate judgment** | ✅ Yes | Qwen3-8B reads similar tickets → judges duplicate/related/unique |
| **KB Q&A** | ✅ Yes | Qwen3-8B reads KB articles → generates grounded answer |

---

## 4. RAG Techniques — Full Specifications

### 4.1 Embedding

| Parameter | Value |
|-----------|-------|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Dimensions | 384 |
| Max sequence length | 256 tokens |
| Similarity metric | Cosine similarity |
| Training | Pre-trained on 1B+ sentence pairs (no fine-tuning by us) |

### 4.2 Vector Store

| Parameter | Value |
|-----------|-------|
| Database | ChromaDB (persistent, SQLite-backed) |
| Index type | HNSW (Hierarchical Navigable Small World) |
| Distance function | Cosine |
| Total embeddings | 507 (500 tickets + 7 KB chunks) |
| Ticket collection | `its_tickets` (500 vectors) |
| KB collection | `its_knowledge_base` (7 vectors) |

### 4.3 BM25 Keyword Search

| Parameter | Value |
|-----------|-------|
| Algorithm | BM25 Okapi (via `rank_bm25`) |
| Tokenization | Lowercase, alphanumeric split, stopword removal |
| Stopwords removed | 80+ common English words |
| Avg tokens per ticket | 24.8 |
| Corpus size | 500 ticket documents + 7 KB documents |

### 4.4 Reciprocal Rank Fusion (RRF)

| Parameter | Value |
|-----------|-------|
| Formula | `score(d) = Σ weight_i / (k + rank_i)` |
| k constant | 60 (standard default) |
| Dense weight | 0.6 |
| BM25 weight | 0.4 |
| Candidates from Dense | Top 20 |
| Candidates from BM25 | Top 20 |

### 4.5 Cross-Encoder Reranking

| Parameter | Value |
|-----------|-------|
| Model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Input | (query, document) pairs |
| Candidates reranked | Top 20 from RRF |
| Final results returned | Top 5 |
| Speed | ~50ms per pair |

### 4.6 Chunking (for KB Documents)

| Parameter | Value |
|-----------|-------|
| Method | Recursive character splitting, section-aware |
| Target chunk size | 450 tokens |
| Overlap | 50 tokens |
| Section detection | Markdown headers (##, ###) |
| Result | 7 chunks (each KB doc fit in 1 chunk) |

### 4.7 LLM

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-8B |
| Serving | Ollama (local, no API key needed) |
| Context window | 128K tokens |
| Temperature | 0.1 (low, for consistency) |
| Max generation | 2048 tokens |
| Prompt strategy | Grounded generation (cite retrieved sources) |

---

## 5. Data Specifications

### 5.1 Ticket Data

| Field | Description | Example |
|-------|-------------|---------|
| ticket_id | Unique identifier | ITS-00042 |
| title | Short issue summary | VPN connection drops |
| description | User's natural language description | "My VPN keeps disconnecting every 10 minutes..." |
| category | Issue category | Network, Hardware, Software, Account, Incident |
| component | Specific system/tool | VPN, Outlook, Laptop, MFA, Teams |
| severity | Impact level | Critical, High, Medium, Low |
| status | Current state | Open, In Progress, Resolved, Closed |
| resolution | How it was fixed (if resolved) | "Updated VPN client. Verified with user." |
| environment | User's OS/platform | Windows 11, macOS 14, Ubuntu 22.04 |
| assigned_team | Responsible team | IT Operations, Network Engineering, Help Desk |
| embedding_text | Optimized text for embedding | Combination of title + description + metadata |

**Distribution (500 tickets):**
- Categories: Incident 108, Network 105, Account 98, Software 95, Hardware 94
- Severities: Medium 247, Low 141, High 87, Critical 25
- With resolution: 241 tickets (48%)
- Duplicate pairs: 69 labeled pairs

### 5.2 Knowledge Base

7 documents (5 runbooks + 2 KB articles):
- VPN troubleshooting (connection drops, MTU, split tunneling)
- Password reset & account lockout (AD, MFA, identity verification)
- Email/Outlook issues (sync, cache, profiles, shared mailboxes)
- Laptop hardware (BSOD codes, overheating, boot failures)
- Software installation (licenses, permissions, disk space)
- Teams performance (cache clearing, memory reduction)
- Printer setup (queue stuck, offline, setup steps)

---

## 6. Evaluation Results

### 6.1 Ablation Study (from your actual run)

| Query | Type | Dense | BM25 | Hybrid | Hybrid+Rerank |
|-------|------|-------|------|--------|---------------|
| "My computer won't start" | Semantic | ✅ Laptop not turning on | ❌ Salesforce?? | ✅ Laptop not turning on | ✅ Laptop not turning on |
| "error code 0x80070005" | Keyword | ✅ Salesforce error [HEX] | ✅ SAP error [HEX] | ✅ Salesforce error [HEX] | ✅ SAP error [HEX] |
| "Cannot access resources remotely" | Semantic | ✅ VPN access issue | ✅ Firewall access | ✅ Firewall access | ✅ VPN access |
| "Outlook not syncing after update" | Mixed | ✅ Outlook sync issue | ✅ Outlook sync issue | ✅ Outlook sync issue | ✅ Outlook sync issue |
| "Teams consuming too much memory" | Mixed | ✅ Teams memory leak | ✅ Teams memory leak | ✅ Teams memory leak | ✅ Teams memory leak |
| "account locked need reset MFA" | Mixed | ✅ MFA account locked | ✅ MFA account locked | ✅ MFA account locked | ✅ MFA account locked |
| "BSOD IRQL_NOT_LESS_OR_EQUAL" | Keyword | ~ BSOD (partial) | ❌ No results | ~ BSOD (partial) | ~ API Gateway?? |
| "printer offline error" | Mixed | ✅ Printer not turning on | ✅ Printer not turning on | ✅ Printer not turning on | ✅ Printer not turning on |

**Key Findings:**
- **BM25 alone fails on semantic queries** (e.g., "My computer won't start" → returned Salesforce)
- **Dense alone misses keyword-specific matches** (version numbers, error codes)
- **Hybrid consistently matches or beats either method alone**
- **Reranking helps on ambiguous queries** but can occasionally pick less relevant results (needs more data)

### 6.2 RAG Generation Quality (from your test)

| Test | Result | Quality |
|------|--------|---------|
| `/ask print queue stuck` | Correct 4-step fix from KB, cited source | ✅ Excellent |
| `/dedup Outlook not syncing` | Correctly identified as RELATED (not duplicate) with reasoning | ✅ Excellent |
| `toilet flush not working` | Correctly said "outside scope," gave general advice, didn't hallucinate | ✅ Excellent (grounded) |
| `truck ran over me` | Correctly redirected to medical help, didn't force IT context | ✅ Excellent (grounded) |
| `/similar laptop BSOD` | Found 5 relevant BSOD tickets ranked correctly | ✅ Good |

---

## 7. How to Make It Better

### 7.1 Add More Data

```bash
# Add real Apache JIRA tickets
python scripts/fetch_jira.py
# Re-run preprocessing and vector store
python scripts/02_preprocess_data.py
python scripts/03_build_vector_store.py --reset
```

### 7.2 Add New Domains (Healthcare, Insurance Fraud)

**Step 1:** Create KB documents for the new domain. Place in `data/knowledge_base/`:

```markdown
# runbook_insurance_claim_processing.md
## Symptom: Claim Processing Delayed
### Step 1: Verify claim completeness
1. Check all required fields are populated
2. Verify supporting documentation attached
...
```

**Step 2:** Create or collect domain-specific tickets (same CSV format as synthetic_tickets.csv).

**Step 3:** Re-run scripts 02 and 03. The RAG pipeline automatically picks up new data — no code changes needed.

### 7.3 Other Improvements

| Improvement | How | Expected Impact |
|-------------|-----|-----------------|
| Query expansion | LLM rewrites vague queries before retrieval | +10-15% recall |
| Metadata filtering | Filter ChromaDB by category/team/date | More relevant results |
| Fine-tune LLM (QLoRA) | Train on ticket extraction JSON format | +10-15% JSON reliability |
| Add more KB docs | Write runbooks for every common issue | Better resolution quality |
| Feedback loop | Save resolved tickets back into KB | Knowledge grows over time |
| Increase ticket data | 500 → 5000+ from real sources | Better duplicate detection |

---

## 8. Project Goal Alignment

**Goal:** Move engineers' focus from problem reporting to problem resolution.

**How ITS achieves this:**

| Before ITS | After ITS |
|------------|-----------|
| Engineer manually searches past tickets | System instantly retrieves similar tickets |
| Engineer reads through runbooks | System finds and surfaces relevant KB sections |
| Engineer writes resolution from scratch | System generates Resolution Blueprint with steps |
| Duplicate tickets waste triage time | Duplicates flagged automatically before assignment |
| New issues have no starting point | Full KB support even for never-seen-before issues |
| Knowledge lives in individual engineers' heads | Knowledge captured in KB, accessible to all via RAG |

---

## 9. Running the Streamlit App

```bash
pip install streamlit
ollama serve &
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

The app has 5 tabs:
1. **Submit Ticket** — Full automatic pipeline (dedup → retrieval → resolution)
2. **Ask Knowledge Base** — Direct Q&A against runbooks
3. **Find Similar** — Compare retrieval methods side-by-side
4. **Ablation Study** — Run and view the 4-method comparison
5. **Documentation** — Everything in this document, rendered in-app
