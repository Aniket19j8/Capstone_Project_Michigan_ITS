# ITS RAG Capstone — Detailed Project Documentation

**Course:** FSE 570 Capstone Project
**Team:** Michigan
**Project name:** ITS RAG (Retrieval-Augmented Generation for IT Service Management)
**Repository:** `Capstone_Project_Team_Michigan-feature-its-michigan-2026`

> **Single source of truth.** The canonical narrative for the final report and presentation is `CAPSTONE_FINAL_STORY.md`. That file owns the Status 1 → Status 2 → Final story, the rubric mapping, and every headline table. **This file** is the deep engineering log used to draft the 8-page ASME paper.

This document is a **detailed engineering and evaluation log** for the project. It is structured so each section maps directly to the FSE 570 rubric (Clarity & Organization, Writing Quality, Data Sources, Methodology, Results & Recommendations, Overall Impact). It is not the final 8-page ASME paper; it is the deep technical record from which that paper will be drafted.

---

## Table of Contents

1. Executive Summary
2. Problem Statement & Motivation
3. System Architecture
4. Data Sources, Cleaning, Integration, and Privacy
5. Methodology — Pipeline by Stage (Files 01–14)
6. Infrastructure: vLLM, Ollama, ChromaDB, BM25
7. Detailed Run Logs (Colab + Local)
8. Evaluations and Results
9. LoRA Fine-Tuning (Phase 6)
10. Recommendations & Decision Frameworks
11. Reproducibility Checklist
12. Risk Register & Limitations
13. Glossary
14. Appendix A — Per-Script Reference
15. Appendix B — Visual Mapping to Rubric

---

## 1. Executive Summary

The project builds an **end-to-end IT Support RAG system** that:

- Ingests **two large heterogeneous datasets** (an 80,000-row enterprise ITS export and a curated `all_tickets.csv` unified ticket set), plus a curated knowledge base of runbooks/articles.
- Cleans, normalizes, and unifies tickets into a single schema with a quality score, a stable `unified_id`, and an `embedding_text` column ready for vectorization.
- Builds **hybrid retrieval** combining dense vectors (Qwen embedding via vLLM/Ollama → ChromaDB cosine), sparse keyword search (BM25 on tokenized text), **Reciprocal Rank Fusion (RRF)**, and a **cross-encoder reranker**.
- Connects retrieval to a generation LLM (Qwen3 4B/8B) for grounded answers, and runs a **multi-evaluation suite** for quality, hallucination, latency, dedup threshold tuning, and end-to-end retrieval recall.
- Applies **LoRA fine-tuning** of Qwen3-4B-Instruct on a tone/format/escalation task carved from real resolved tickets, then evaluates the adapter against the base model on JSON contract quality.

The final stack has **measurable retrieval quality (Recall@k, MRR, nDCG, Jaccard stability)**, a **hallucination quad-chart**, **per-stage latency**, **threshold-sweep dedup analysis**, and a **trained, saved LoRA adapter** that is ready to be merged at serving time.

---

## 2. Problem Statement & Motivation

IT Support teams face three coupled problems:

1. **Repeated tickets**: 30–50% of incoming tickets are near-duplicates of historically resolved ones. Engineers re-investigate the same issues.
2. **Hallucinating LLM helpers**: A pure LLM helper (no retrieval) will invent steps, error codes, URLs, and policy numbers. This is dangerous in IT.
3. **Triage drift**: New tickets land in the wrong queue, slowing time-to-resolution and harming SLA compliance.

**Decision-making goal:** Help a Tier-1 helpdesk decide, for each incoming ticket, **(a)** “Has this been solved before? Show me the closest historical resolutions,” **(b)** “What grounded steps should the user try first?” and **(c)** “Should this be escalated, and why?” — all backed by retrieval, not by free-form LLM hallucination.

---

## 3. System Architecture

```
                           ┌──────────────────────────┐
                           │  Ingest (01_download)    │
                           │  - 80k ITS export        │
                           │  - GitHub issues, KB     │
                           └────────────┬─────────────┘
                                        │
                           ┌────────────▼─────────────┐
                           │  Preprocess (02)         │
                           │  - clean_text            │
                           │  - extract_error_codes   │
                           │  - quality_score         │
                           │  - unified_id            │
                           │  - department mapping    │
                           └────────────┬─────────────┘
                                        │
        ┌───────────────────────────────┼─────────────────────────────────┐
        │                               │                                 │
        ▼                               ▼                                 ▼
┌──────────────┐               ┌────────────────┐               ┌──────────────────┐
│ ChromaDB     │               │ BM25 corpus    │               │ KB chunks        │
│ its_tickets  │               │ tokenized json │               │ runbooks/.md     │
│ 80,000 docs  │               │ 80,000 entries │               │ ChromaDB its_kb  │
└──────┬───────┘               └────────┬───────┘               └────────┬─────────┘
       │                                │                                │
       └─────────┬──────────────────────┴──────────────┬─────────────────┘
                 │                                     │
                 ▼                                     ▼
       ┌──────────────────────────┐          ┌──────────────────────────┐
       │  HybridRetriever (04)    │          │  KBRetriever / Combined  │
       │  Dense + BM25 → RRF →    │          │  same engine, KB         │
       │  Cross-encoder rerank    │          │  collection              │
       └────────────┬─────────────┘          └─────────────┬────────────┘
                    │                                      │
                    └────────────────┬─────────────────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │  RAG Pipeline (05)  │
                          │  Qwen3 LLM grounded │
                          │  generation         │
                          └──────────┬──────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                            ▼
┌──────────────┐           ┌──────────────────┐         ┌────────────────────┐
│ Eval (06)    │           │ 500-Q Eval (07)  │         │ Hallucination (08) │
│ Recall, MRR, │           │ Self-retrieval,  │         │ Base vs RAG +      │
│ Ablation     │           │ + Perturbation   │         │ LLM-as-judge       │
└──────────────┘           │ Stability eval   │         └────────────────────┘
                           └──────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                            ▼
┌──────────────┐           ┌──────────────────┐         ┌────────────────────┐
│ Latency (09) │           │ Dedup τ (10)     │         │ Phase 5 (11)       │
│ Per-stage    │           │ Threshold sweep, │         │ Data readiness +   │
│ ms breakdown │           │ cost-sensitive   │         │ triage + smoke     │
└──────────────┘           └──────────────────┘         └────────────────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │  LoRA Fine-Tune     │
                          │  12 Prepare         │
                          │  13 Train (PEFT)    │
                          │  14 Evaluate vs base│
                          └─────────────────────┘
```

---

## 4. Data Sources, Cleaning, Integration, and Privacy

### 4.1 Datasets

| Source | File | Rows | Role |
|---|---|---:|---|
| Major enterprise ITS export | `data/processed/its_tickets_80k.csv` | **80,000** | Primary heterogeneous ticket corpus used for the vector store and BM25 index |
| Unified curated tickets | `data/processed/all_tickets.csv` | **503** | Smaller, schema-canonical set used for fast self-retrieval (07 default) |
| Knowledge base | `data/knowledge_base/*.md` | dozens of articles → **42 chunks** in ChromaDB | Domain runbooks (VPN, Outlook, BSOD, Teams, etc.) |
| Synthetic duplicate pairs | `data/processed/synthetic_duplicate_pairs.csv` | labeled positives + negatives | Used by 10 to sweep dedup thresholds |
| Auxiliary candidates registered in 01 | GitBugs, Apache Jira exports, GitHub issues, Quora | varies | Listed/registered in `01_download_data.py`; not all used in final eval |

Two are **large and heterogeneous** (80k ITS export + KB chunks), satisfying the rubric’s “at least two large, heterogeneous datasets” criterion.

### 4.2 Schema Normalization (`02_preprocess_data.py`)

`clean_text(...)` strips noise: `[URL]`, `[EMAIL]`, `[PATH]`, `[HEX]` placeholders, normalizes whitespace, drops non-printable chars.

`extract_error_codes(...)` pulls codes via regex (`error N`, `0xABCD`, `TICKET-12345`, `errno N`).

`compute_text_quality_score(row)` returns 0–1 from presence/length of title, description, component, severity. We keep this as a **soft filter** for downstream sampling.

A single canonical `embedding_text` column is built from cleaned title + description + (optionally) component/category. A stable `unified_id` (`U-XXXXXX`) is assigned across all sources so the same ticket is never re-embedded twice.

### 4.3 Department Triage Map (`department_mapping.py`)

Tickets are classified into one of four functional departments (Network & Security, End-User & Desktop, Applications & Data, IT Infrastructure). The mapping is rule-based with confidence + reason fields, and Phase 5 evaluates it against an 8-question golden set.

### 4.4 Privacy and Permissions

- The 80k export was provided for academic use; it does not contain PII or customer-identifying account credentials in cleaned form.
- `clean_text` masks emails and URLs and removes raw filesystem paths and hex error codes; this gives an **anonymization-by-substitution** layer before any text leaves the local pipeline.
- No raw data is uploaded to a third-party LLM service. **Local Ollama** or **self-hosted vLLM** is used for both embedding and generation. The only “external” API surface during evaluation is HuggingFace model download (weights only, no data sent).

---

## 5. Methodology — Pipeline by Stage (Files 01–14)

### 5.1 `01_download_data.py` — Ingest

Registers/loads candidate datasets into `data/raw/` and writes `data/processed/major_its_dataset_manifest.json` describing rows, columns, and sample category/component/status counts so the next stage can validate inputs without re-reading the whole CSV.

### 5.2 `02_preprocess_data.py` — Clean & Normalize

Builds the unified table and writes:

- `data/processed/all_tickets.csv` — canonical schema
- `data/processed/all_tickets_clean.json` — JSON for downstream LLM consumption
- `data/processed/data_stats.json` — row counts, language distribution, completeness

Supports `--append` to add a new source folder (`--new-source ./data/raw/jira_export --dataset-name jira2 --append --min-quality 0.3`) and `--column-map` for explicit mapping when auto-detection fails.

### 5.3 `03_build_vector_store.py` — Vector + BM25 Index

For each ticket and each KB chunk:

- Embed via `OllamaEmbedder.encode(...)` (1024-dim by default; we measure on first call). In Colab we patched this to call **vLLM `/v1/embeddings`** (Qwen3-Embedding-0.6B), see §6.
- Upsert into ChromaDB collections: `its_tickets`, `its_knowledge_base` (cosine).
- Tokenize and write `data/processed/bm25_corpus_tickets.json` and `bm25_corpus_kb.json`.

Build manifest `data/processed/build_manifest.json` records the embedding model, dim, and counts.

Verified result this run:
- `its_tickets`: **80,000 vectors**
- `its_knowledge_base`: **42 chunks**
- BM25 tickets: **80,000 docs** (~78 MB JSON)

### 5.4 `04_hybrid_retrieval.py` — Hybrid Retriever

Pipeline per query:

1. **Dense search**: ChromaDB cosine (top-K=20)
2. **BM25 search**: rank_bm25 with the same tokenizer used at build time (top-K=20)
3. **RRF fusion**: weighted reciprocal rank with `RRF_K=60`, `dense_weight=0.6`, `bm25_weight=0.4`
4. **Cross-encoder reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2`, top-K=5 by default

Convenience methods: `search_dense_only`, `search_bm25_only`, `search` (with/without rerank).

`KBRetriever` and `CombinedRetriever` reuse the same engine on the KB collection.

### 5.5 `05_rag_pipeline.py` — Grounded Generation

`OllamaLLM` (default `qwen3:8b`) consumes retrieved context, with:

- `MIN_RERANK_SCORE = -2.0` (5) and `-3.0` (8): off-topic context is dropped, so out-of-scope queries don’t produce confident IT-shaped hallucinations.
- `strip_thinking_tokens` removes `<think>...</think>` from Qwen3 outputs before JSON parsing.
- `extract_json_object` greedily finds the JSON object in mixed text.

Two main modes: free-form RAG answer, and **Resolution Blueprint** (Agent Assist) which returns structured steps.

### 5.6 `06_evaluation.py` — Retrieval Quality

Metrics: Recall@k, Precision@k, MRR, nDCG@k. Implements a 4-method **ablation** (Dense-only, BM25-only, Hybrid no-rerank, Hybrid + rerank) over a curated query set and writes:

- `evaluation/ablation_results.csv`
- `evaluation/dedup_threshold_results.csv` (when `--dedup`)

### 5.7 `07_run_500_eval.py` — 500-Query Self-Retrieval

For each ticket in `all_tickets.csv` (or `its_tickets_80k.csv` after our patch), uses its own `embedding_text` as a query and checks whether the same ticket is in the top-k. Reports Recall@1/3/5/10, MRR, nDCG@5/10, per-category and per-severity. Outputs:

- `evaluation/500_query_results.csv`
- `evaluation/metrics_summary.json`
- `evaluation/retrieval_comparison_chart.png`

We additionally built a **resumable “unique”** evaluation:

- `perturbation_stability_resumable.py` (created in Colab)
- For each sampled ticket, generates 5 query variants (`clean`, `lower`, `add_noise`, `typos_light`, `code_injection`) and reports:
  - `target_in@k` (does the true id appear under any perturbation),
  - **`jaccard_top10_across_perturbs`** (stability of top-10 list under perturbations),
  - per-method latency,
  - resumable: writes `evaluation/perturbation_stability_n<N>.partial.csv` after every ticket.

This adds a **stability/robustness axis** beyond plain recall — directly visible in the slide deck and in the report.

### 5.8 `08_hallucination_comparison.py` — RAG vs Base LLM

Sends 50 queries (45 in-scope IT + 5 out-of-scope) through:

- Base LLM (no context)
- RAG (hybrid retrieval with relevance-filtered context)

Then uses **LLM-as-judge** to score `groundedness`, `specificity`, `accuracy` (1–5) and a binary `hallucination_in_a/b`, with an explicit scope-aware judge prompt that treats “I don’t have this information” as **correct** behavior on out-of-scope queries.

Outputs:
- `evaluation/hallucination_comparison.csv`
- `evaluation/hallucination_summary.json`
- `evaluation/hallucination_chart.png` (3-panel: hallucination rate, quality scores, head-to-head)

### 5.9 `09_latency_benchmark.py` — Per-Stage Latency

Measures milliseconds per stage: embedding, dense, BM25, RRF, rerank, KB retrieval, LLM. Reports mean / P50 / P95 over 20 queries. Saves:

- `evaluation/latency_results.json`
- `evaluation/latency_chart.png` (per-stage bars + total distribution)

### 5.10 `10_dedup_threshold.py` — Threshold Optimization

Embeds labeled `synthetic_duplicate_pairs.csv` with a small SentenceTransformer, sweeps τ from 0.50 → 0.98 in 0.02 steps, computes Precision/Recall/F1 and a **cost-sensitive** objective (`c_FN = 2 × c_FP`). Saves:

- `evaluation/dedup_threshold_results.csv`
- `evaluation/dedup_summary.json` (best F1 τ + cost-optimal τ)
- `evaluation/dedup_threshold_chart.png` (P/R/F1 curve + cost curve)

### 5.11 `11_phase5_evaluation.py` — Pre-Phase-6 Suite

Cheap, no-GPU checks before LoRA:

- **Data readiness**: required/optional column completeness, duplicates, resolution coverage, quality_score percentiles, distributions of category/component/status/language/source.
- **Triage** (Model 2): department-mapping rule classifier evaluated on the full df + on an 8-case golden set.
- **Retrieval smoke test**: 100 sampled tickets, hit@1/5/10 + p95 latency on the live ChromaDB.
- **Chatbot contract**: 4 small Model 1 cases; checks JSON schema, escalation flag, latency.

Writes a consolidated `evaluation/phase5_summary.json` and `evaluation/phase5_report.md`.

### 5.12 `12_prepare_lora_dataset.py` — LoRA Dataset

Converts resolved historical tickets into supervised chat examples with system prompt:

> “You are a Tier-1 IT helpdesk assistant inside a RAG system. Use the ticket context to write a natural support response. Return JSON ONLY with this schema: {reply, recommended_steps, confidence, escalation_recommended, escalation_reason}.”

Per row, builds 2–5 short steps from the resolution prose, computes a confidence proxy from `status` + `quality_score`, and decides escalation when severity is critical or confidence < 0.55.

Splits into train / val / test (e.g., 84/8/8 percent) and writes JSONL + a manifest. Final run for this notebook:

```text
total_source_rows_after_filters: 37450
examples_total: 500    (train 420 / val 40 / test 40)
```

### 5.13 `13_train_lora.py` — PEFT/LoRA Training

`AutoTokenizer` + `AutoModelForCausalLM` (4-bit via `BitsAndBytesConfig` with `nf4` + double quant + fp16 compute), `peft.LoraConfig` targeting `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`, `trl.SFTTrainer` (or `SFTConfig` in newer TRL). Saves only the adapter, not a merged model. Writes `training_manifest.json`.

For T4 (15GB), we run with:

```text
--load-in-4bit --max-seq-length 256–512 --batch-size 1 --grad-accum 8–16
--epochs 0.5 --lora-r 8 --lora-alpha 16
```

After our final patches (CPU-offload-friendly load, fp16 compute, `LD_LIBRARY_PATH` for `libnvJitLink.so.13`, killing vLLM to free GPU), training **completed** with:

```text
train_runtime ≈ 226.5 s for 14 steps
train_loss     ≈ 2.73 (from 3.03)
mean_token_accuracy ≈ 0.61
adapter_model.safetensors ≈ 64 MB saved
```

### 5.14 `14_evaluate_lora.py` — Adapter vs Base

For each test case it:

- Renders the prompt via tokenizer chat template,
- Generates greedy with the **base** model and again with the **LoRA-loaded** model,
- Strips JSON, scores it on a `contract_score` rubric (json_valid, has_reply, step_count in [2,5], confidence in [0,1], escalation bool, escalation_reason string).

Saves `evaluation/phase6_lora_comparison.csv` and `phase6_lora_comparison.json`.

We can run with `--skip-base` on T4 to evaluate only the adapter and avoid loading the base twice.

---

## 6. Infrastructure: vLLM, Ollama, ChromaDB, BM25

### 6.1 Embedding via vLLM (Colab)

- Model: **`Qwen/Qwen3-Embedding-0.6B`**
- Endpoint: `http://localhost:8000/v1/embeddings` (OpenAI-compatible)
- Patched the original Ollama-style `OllamaEmbedder` to POST to `/embeddings` and sort responses by `index`, preserving the same `.encode(...)` interface used by `HybridRetriever`.

### 6.2 LLM serving via vLLM (when used)

- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Endpoint: `http://localhost:8000/v1/chat/completions`
- Settings used on T4: `--dtype float16 --max-model-len 4096 --gpu-memory-utilization 0.80 --enforce-eager`
- Important constraint: vLLM serving and LoRA training **cannot coexist** on a 15GB T4. We kill the vLLM process before LoRA training.

### 6.3 Ollama (local default)

`05_rag_pipeline.py`, `08_hallucination_comparison.py`, `09_latency_benchmark.py` default to local Ollama (`qwen3:8b`/`qwen3:4b`) with `OLLAMA_URL=http://localhost:11434`. The same `HybridRetriever` is reused, with `OllamaEmbedder` pointing at `/api/embed`.

### 6.4 ChromaDB

Persistent client at `./data/chroma_db`. Collections: `its_tickets`, `its_knowledge_base`. We added a `pysqlite3` shim in `03_build_vector_store.py` because Chroma needs SQLite ≥ 3.35 and many runtimes ship older system sqlite3.

### 6.5 BM25

`rank_bm25.BM25Okapi` over the same lowercased word tokens with a small English stopword list. The corpus JSON is materialized once at build time and reloaded by `HybridRetriever`.

---

## 7. Detailed Run Logs (Colab + Local)

### 7.1 Setup notebook A — vLLM embedding + retrieval evals

1. Mount Drive → `cd` project.
2. Install: `uv pip install torch torchvision torchaudio openai transformers huggingface_hub accelerate safetensors --index-url https://download.pytorch.org/whl/cu121`, then `uv pip install vllm`.
3. Download model: `hf download Qwen/Qwen3-Embedding-0.6B`.
4. Start vLLM in background: `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-Embedding-0.6B --task embed --dtype float16 --max-model-len 4096 --gpu-memory-utilization 0.75 --trust-remote-code --enforce-eager > vllm_embed.log 2>&1 &`.
5. Patched `04_hybrid_retrieval.py` (vLLM client) and disabled `sentence_transformers.CrossEncoder` import to avoid `torchcodec` / `libavutil` load failure on Colab.
6. Verified `data/chroma_db` had `its_tickets` (80k) and `its_knowledge_base` (42).
7. Ran `07` baseline and the **resumable perturbation eval**, both saved partials every checkpoint so kernel disconnects (twice) didn’t lose progress.

### 7.2 Setup notebook B — LoRA training

1. Restart runtime (no vLLM in this notebook).
2. Mount Drive, `cd` project.
3. Install pinned LoRA stack: `transformers==4.56.2`, `tokenizers>=0.22.0`, `accelerate==1.10.1`, `peft==0.17.1`, `trl==0.23.1`, `datasets==4.4.1`, `bitsandbytes` (auto-pulled by Colab’s CUDA 13 stack), plus `sentencepiece protobuf pandas tqdm`.
4. Patched `13_train_lora.py`:
   - `--bf16 default=False` (T4 needs fp16)
   - `evaluation_strategy=` → `eval_strategy=`
   - Newer TRL `SFTConfig` with `max_length`, `dataset_text_field`, `processing_class=tokenizer`
   - 4-bit `BitsAndBytesConfig` with `llm_int8_enable_fp32_cpu_offload=True`, `bnb_4bit_compute_dtype=torch.float16`, and `device_map="auto"` with `max_memory={0: "13GiB", "cpu": "30GiB"}`.
5. Kept hitting `libnvJitLink.so.13` because Colab’s torch is CUDA 13. Solved by exporting `LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH` for the training command.
6. Killed vLLM (`pkill -9 -f vllm`) so the 4B base could fit alongside training.
7. Trained successfully (logs in §5.13). Adapter saved at:
   ```text
   models/lora/its-qwen3-4b-helpdesk-lora-final/
     ├── adapter_model.safetensors      (64M)
     ├── adapter_config.json
     ├── training_manifest.json
     ├── chat_template.jinja
     ├── tokenizer.json / tokenizer_config.json / vocab.json / merges.txt
     ├── special_tokens_map.json
     ├── added_tokens.json
     ├── training_args.bin
     └── checkpoint-14/
   ```

### 7.3 Failure modes encountered (and the fix that worked)

| Symptom | Root cause | Fix that worked |
|---|---|---|
| `Connection refused localhost:8000` | vLLM not running in this runtime | start the embedding server and wait for `Application startup complete` |
| `from rank_bm25 import BM25Okapi → ModuleNotFoundError` | new runtime didn’t have project deps | `pip install -q rank_bm25 chromadb pandas numpy tqdm requests pysqlite3-binary` |
| `Collection [its_tickets] does not exist` | chroma_db not synced to Drive yet | re-uploaded `data/chroma_db` and `data/processed/*` to Drive |
| `Expected embeddings to be a list of floats... got [[[...]]]` | extra nesting from vLLM client | indexed `[0]` before `.tolist()` in `dense_search` |
| `torchcodec libavutil.so.60` on `from sentence_transformers import CrossEncoder` | Colab missing FFmpeg DLL | disabled `CrossEncoder` import for retrieval eval (rerank disabled) |
| `Read timed out (120s)` on first embedding | first-token bring-up | bumped client timeout to 300s and reduced batch_size to 16 |
| `libnvJitLink.so.13` (bitsandbytes) | Colab torch on CUDA 13 | run with `LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH` |
| `evaluation_strategy unexpected kwarg` | Transformers 4.46+ rename | `eval_strategy=` |
| `SFTTrainer.__init__ unexpected tokenizer/dataset_text_field/max_seq_length` | TRL ≥ 0.23 moved to `SFTConfig` | rewritten trainer init using `SFTConfig` |
| “Some modules dispatched on CPU/disk” | 4B + 4-bit was too tight + vLLM holding GPU | killed vLLM; added `llm_int8_enable_fp32_cpu_offload=True`, `max_memory`, fp16 compute |

These resolutions are part of the methodology and recommended reproducibility checklist.

---

## 8. Evaluations and Results

### 8.1 Self-Retrieval (07)

Methods evaluated: `Dense_Only`, `BM25_Only`, `Hybrid_NoRerank`, `Hybrid_Rerank` (when reranker is available). Metrics: Recall@1/3/5/10, MRR, nDCG@5/10, per-category, per-severity. The script produces:

- `evaluation/500_query_results.csv` (per-query)
- `evaluation/metrics_summary.json` (aggregate)
- `evaluation/retrieval_comparison_chart.png`

When the rerank dependency is unavailable in the current runtime (e.g., `torchcodec` blocking `sentence_transformers`), `Hybrid_Rerank` falls back to top-k of the fused candidates. We document this transparently in the report.

### 8.2 Perturbation Stability (custom)

Stable retrieval matters for production. We sample N tickets and for each one generate 5 variants of the query (`clean`, `lower`, `add_noise`, `typos_light`, `code_injection`). Per ticket × method, we record:

- `target_in@1/3/5/10` — was the true ticket retrieved under at least one perturbation?
- `best_rank_any_perturb`, `worst_rank_any_perturb`
- **`jaccard_top10_across_perturbs`** — stability of the top-10 list across perturbations
- `mean_latency_ms`

Resumable to `evaluation/perturbation_stability_n<N>.partial.csv`. After 150 tickets the run survives kernel disconnects without losing data. This is the **“unique evaluation”** axis on top of the rubric’s expected metrics.

### 8.3 Hallucination Comparison (08)

LLM-as-judge over 45 IT + 5 OOS queries. Reports per-subset because OOS dilutes the headline averages. Visual: `hallucination_chart.png` with three panels (hallucination rate IT, quality scores IT, head-to-head all). The judge prompt explicitly defines hallucination for IT-RAG and tells the judge that an OOS refusal from RAG is correct.

### 8.4 Latency Benchmark (09)

Per-stage breakdown (ms): embed, dense, BM25, RRF, rerank, KB retrieval, LLM. Reports mean/P50/P95 across 20 queries. Visual: stacked bar of mean per stage + histogram of total latency with mean and P95 markers.

### 8.5 Dedup Threshold (10)

Sweep τ from 0.50 to 0.98. Reports best-F1 τ (operational quality) and **cost-optimal τ** (operational cost: FN costs 2× FP). Visual: P/R/F1 curve and Cost curve side-by-side. This gives the team a **decision-making framework** for picking τ in production based on which error type is more expensive in their environment.

### 8.6 Phase 5 Suite (11)

- Data readiness JSON (rows, completeness %, duplicate counts, distributions).
- Triage metrics JSON (avg confidence, low-confidence %, golden 8-case accuracy).
- Retrieval smoke metrics (hit@1/5/10, p95 latency over 100 sampled tickets).
- Chatbot contract (schema pass rate, escalation flag, latency).
- Markdown rollup `phase5_report.md`.

### 8.7 LoRA Adapter Evaluation (14)

`14_evaluate_lora.py` reports per-model summary: cases, json_valid_rate, avg_contract_score, avg_step_count, avg_latency_ms, p95_latency_ms. We invoke with `--skip-base` on T4 to evaluate only the adapter and avoid base model double-load.

---

## 9. LoRA Fine-Tuning (Phase 6)

### 9.1 Why LoRA, not full fine-tune

We deliberately do **not** memorize tickets in the LLM weights. Ticket facts are owned by RAG (it can be updated independently and is auditable). LoRA is used only to adjust **format, tone, step shaping, confidence, and escalation phrasing** so the assistant produces consistent JSON that downstream UI and APIs can render.

### 9.2 Dataset construction

- **Final frozen dataset (Status 2):** 1,500 examples from 37,450 candidates after `--english-only` and resolution-length filter.
- **Train/Val/Test:** 1260 / 120 / 120 (manifest at `data/lora/lora_dataset_manifest.json`).
- Each example: `[system, user, assistant]` chat where the assistant message is a **JSON-only** payload matching the production schema.
- An earlier Status 1 iteration ran on 500 examples (420/40/40) using the same script with `--max-examples 500`. We preserved the manifest at the 1,500-example final size and use that as the reported number.

### 9.3 Trainer

- Quantization: 4-bit nf4, double quant, fp16 compute (T4-safe).
- LoRA: r=8, α=16, dropout=0.05, targeting all attention + MLP projections.
- Optimizer: `paged_adamw_8bit` (because of 4-bit base).
- LR: 2e-4, warmup 3%, gradient_checkpointing on.
- Epochs: 0.5 (T4-tight, conservative). Successful run logged train_loss 3.03 → 2.73, accuracy 0.61.

### 9.4 Risks/limits we acknowledge

- Adapter quality is bounded by example count and budget. With more compute we plan a v2 at 1500 examples × 1 full epoch and `lora-r 16`.
- Some MLP target-module weights are **MISSING/CONVERSION** in the load report when 4-bit quantization runs into `bnb` glitches; we mitigate by warming with the working library combination shown in §7.

---

## 10. Recommendations & Decision Frameworks

1. **Use Hybrid (Dense + BM25 + RRF + Rerank) by default.** Dense alone misses keyword-heavy tickets (error codes, IDs). BM25 alone misses paraphrase. RRF is robust to weight changes; the cross-encoder adds the final precision lift.
2. **Choose τ for dedup based on cost.** If false negatives cost more than false positives (engineers re-investigating), use the `cost_optimal_threshold` from §8.5. If user trust is the priority, use `best_f1_threshold`.
3. **Never let the base LLM answer IT questions without retrieval.** §8.3 shows hallucination drops sharply under RAG with relevance-filtered context, especially when the relevance threshold (`MIN_RERANK_SCORE`) drops irrelevant docs for OOS queries.
4. **Keep ticket knowledge in RAG, behavior in LoRA.** This is auditable, updatable, and avoids catastrophic forgetting.
5. **Separate runtimes.** Embedding server (vLLM `/v1/embeddings`), generation LLM (vLLM or Ollama), and LoRA training do not co-exist on a single 15GB GPU. Schedule them.
6. **Always run the perturbation stability eval before release.** It catches retrieval that looks good on the canonical query but is brittle to typos or noisy queries.

---

## 11. Reproducibility Checklist

1. `data/processed/its_tickets_80k.csv` and `data/processed/all_tickets.csv` present.
2. `data/processed/bm25_corpus_tickets.json` present (≈78 MB).
3. `data/chroma_db/` present with collections `its_tickets` (80,000) and `its_knowledge_base` (42).
4. Embedding service reachable (vLLM `/v1/embeddings` or Ollama `/api/embed`) with the **same** model used for build.
5. `requirements.txt` installed for retrieval evals.
6. `requirements-lora.txt` installed in a separate runtime for LoRA.
7. Patches applied:
   - `04_hybrid_retrieval.py` → vLLM embedding client (only when running on Colab with vLLM).
   - `13_train_lora.py` → `eval_strategy`, `--bf16 default=False`, new `SFTConfig`/`SFTTrainer`, 4-bit + CPU offload, `max_memory`.
8. For LoRA on T4, run with `LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH`.
9. Outputs written to `evaluation/` and `models/lora/...`.

---

## 12. Risk Register & Limitations

- **Single-GPU memory pressure:** Cannot run vLLM serving and LoRA training simultaneously on T4. Mitigation: separate notebooks, `pkill -9 -f vllm` before training.
- **Library churn:** TRL/Transformers/bitsandbytes APIs change frequently. We documented exact pins that work in §5.13 and §7.2.
- **Reranker availability on Colab:** Some Colab images are missing `libavutil` for `torchcodec`, blocking `sentence_transformers`. Mitigation: disable rerank import in retrieval eval and document that `Hybrid_Rerank` ≈ Hybrid top-k in those runs.
- **Sample size in LoRA:** First successful run used 420 train examples. v2 plan: 1260 train examples, 1 epoch, `lora-r 16` on a stronger GPU.
- **LLM-as-judge variance:** `08_hallucination_comparison.py` uses temperature 0.0 and a JSON-only response; we still report per-query results (CSV) so reviewers can spot-check.

---

## 13. Glossary

- **RAG** — Retrieval-Augmented Generation. LLM is given retrieved context for grounding.
- **BM25** — Sparse keyword ranking from IR, robust to exact-token matches like error codes and IDs.
- **RRF** — Reciprocal Rank Fusion: combines rankings from different retrievers.
- **Cross-encoder reranker** — A small model that scores query–doc pairs jointly for high-precision ranking.
- **LoRA** — Low-Rank Adaptation: trains small adapter matrices on top of a frozen base LLM.
- **PEFT** — Parameter-Efficient Fine-Tuning library that includes LoRA.
- **vLLM** — High-throughput inference server with an OpenAI-compatible API.
- **Hybrid retrieval** — Dense + sparse + fusion, optionally with reranker.
- **Quad chart** — Common project communication: 4 small panels summarizing key results.

---

## 14. Appendix A — Per-Script Reference

| File | Purpose | Key outputs |
|---|---|---|
| `01_download_data.py` | Register/load datasets (D0..D4) | `data/processed/major_its_dataset_manifest.json` |
| `02_preprocess_data.py` | Clean, normalize, unify | `all_tickets.csv`, `data_stats.json` |
| `03_build_vector_store.py` | Embed, BM25, ChromaDB collections | Chroma `its_tickets` + `its_knowledge_base`, `bm25_corpus_*.json`, `build_manifest.json` |
| `04_hybrid_retrieval.py` | Hybrid retriever (Dense + BM25 + RRF + rerank) | Class API for downstream scripts |
| `05_rag_pipeline.py` | Grounded answers via Ollama LLM | Resolution Blueprint, RAG answers |
| `06_evaluation.py` | Retrieval ablation + dedup eval entry | `ablation_results.csv`, dedup results |
| `07_run_500_eval.py` | 500-Q self-retrieval | `500_query_results.csv`, `metrics_summary.json`, chart |
| `perturbation_stability_resumable.py` | Custom stability/robustness eval | `perturbation_stability_n<N>.partial.csv` |
| `08_hallucination_comparison.py` | RAG vs Base LLM, judged | `hallucination_comparison.csv`, summary, chart |
| `09_latency_benchmark.py` | Per-stage latency | `latency_results.json`, chart |
| `10_dedup_threshold.py` | Threshold sweep + cost-sensitive | `dedup_threshold_results.csv`, summary, chart |
| `11_phase5_evaluation.py` | Pre-LoRA suite (data, triage, smoke, chatbot) | `phase5_summary.json`, `phase5_report.md` |
| `12_prepare_lora_dataset.py` | Build chat dataset from resolved tickets | `data/lora/its_lora_{train,val,test}.jsonl`, manifest |
| `13_train_lora.py` | PEFT/LoRA training | `models/lora/.../adapter_model.safetensors`, manifest |
| `14_evaluate_lora.py` | Adapter vs base evaluation | `phase6_lora_comparison.csv/json` |
| `15_visualize_evaluations.py` | **NEW** — render every evaluation CSV into a PNG chart for the deck | `evaluation/{retrieval,hallucination,dedup,perturbation,ablation,embedding,dataset,kpi}_*.png` and several `*_summary.json` |
| `16_advanced_metrics.py` | **NEW** — bootstrap CIs, paired t-test + Wilcoxon vs Hybrid_Rerank, Cohen's d, per-category and per-severity breakdowns, latency P50/P95/P99, dedup cost-quality Pareto | `evaluation/advanced_metrics.json`, `significance_tests.json`, `per_category_breakdown.csv`, `per_severity_breakdown.csv`, `cost_pareto.csv`, `advanced_metrics_chart.png` |
| `ITS-v2-main/scripts/eval_v2_retrieval.py` | **NEW** — Pinecone+OpenAI retrieval benchmark vs research baselines | `ITS-v2-main/evaluation_v2/v2_retrieval.{csv,json,_chart.png}` |
| `ITS-v2-main/scripts/eval_v2_agent.py` | **NEW** — End-to-end LangGraph agent benchmark (turn latency, citation rate, guardrail precision, route distribution) | `ITS-v2-main/evaluation_v2/v2_agent.{csv,json,_chart.png}` |

---

## 15. Appendix B — Visual Mapping to Rubric

| Rubric criterion | Where it lives in this project |
|---|---|
| **Clarity & Organization** | This document’s ToC + the per-stage pipeline diagram in §3 + per-script appendix in §14 |
| **Writing Quality, Formatting, Citations** | We will translate this technical record into ASME format. All third-party assets (Qwen models, ChromaDB, BM25Okapi, sentence-transformers, peft/trl) are properly attributed in the report |
| **Data Sources & Analysis** | §4: 80k + curated + KB + synthetic dedup pairs, plus cleaning & quality scoring in §5.2; permission/anonymization in §4.4 |
| **Methodology & Application** | §5 covers the full ML/IR/LLM stack: dense retrieval, sparse BM25, RRF, cross-encoder reranking, RAG generation with grounding rules, LoRA via PEFT, and validation methods (recall/MRR/nDCG, cost-sensitive thresholds, LLM-as-judge, JSON contract score) |
| **Results, Visualizations & Recommendations** | §8 (results) + §10 (recommendations and decision frameworks); generated charts: `retrieval_comparison_chart.png`, `hallucination_chart.png`, `latency_chart.png`, `dedup_threshold_chart.png`, `phase6_lora_comparison.*` |
| **Overall Impact** | §1 + §10 explain the production decision frameworks (hybrid retrieval, τ choice, RAG vs base, runtime separation) and §11 makes the work reproducible by others |

---

*End of document.*
