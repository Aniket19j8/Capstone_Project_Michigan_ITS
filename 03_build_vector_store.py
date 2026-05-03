"""
03 — turn cleaned rows + KB markdown into Chroma collections and BM25 sidecars.

Ollama (or whatever ITS_EMBEDDING_MODEL says) does embeddings; we chunk KB docs,
skip chunking for tickets, and write bm25_corpus_*.json for step 04.
Annoying cluster note: Chroma wants a newer sqlite — the shim below swaps in pysqlite3.

Usage:
  python 03_build_vector_store.py
  python 03_build_vector_store.py --reset
  python 03_build_vector_store.py --verify-only
"""
from __future__ import annotations  # Python 3.9: allows str | None in type hints (3.10+ syntax)

# Chroma needs SQLite >= 3.35.0. Many cluster images ship older system sqlite3; use
# pysqlite3-binary (see requirements) and register it before *any* import that may
# load sqlite3 — notably `pandas` imports sqlite3 on load, so chromadb must come
# before pandas, right after this shim.
import sys
sys.modules.pop("sqlite3", None)  # drop cached stdlib sqlite if something pre-loaded it
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import re
import json
import pickle
import argparse
import datetime
from pathlib import Path

import chromadb
from chromadb.config import Settings

import numpy as np
import pandas as pd
from tqdm import tqdm
import requests

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def default_embedding_model() -> str:
    return (os.environ.get("ITS_EMBEDDING_MODEL", "qwen3:0.6b").strip() or "qwen3:0.6b")


def resolve_ollama_base_url(cli_url: str | None = None) -> str:
    """where the local model server lives — env wins for docker weirdness"""
    if cli_url and cli_url.strip():
        u = cli_url.strip().rstrip("/")
        return u if u.startswith("http") else f"http://{u}"
    env_first = os.environ.get("ITS_OLLAMA_URL", "").strip()
    if env_first:
        u = env_first.rstrip("/")
        return u if u.startswith("http") else f"http://{u}"
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").strip()
    if host.startswith("http"):
        return host.rstrip("/")
    return f"http://{host}".rstrip("/")


EMBEDDING_MODEL = default_embedding_model()
EMBEDDING_DIM = 1024  # hint only; OllamaEmbedder measures real dim on first call
BATCH_SIZE = max(1, _env_int("ITS_EMBED_BATCH_SIZE", 32))
CHROMA_PERSIST_DIR = "./data/chroma_db"
PROCESSED_DIR = Path("./data/processed")
KB_DIR = Path("./data/knowledge_base")

CHUNK_SIZE = 450
CHUNK_OVERLAP = 50

TICKET_COLLECTION = "its_tickets"
KB_COLLECTION = "its_knowledge_base"


def chunk_document(text, title="", doc_id="", chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """KB only — slice on ## headers when we can, else fall back to dumb windows"""
    chunks = []

    sections = re.split(r'\n(?=#{1,3}\s)', text)

    current_chunk = ""
    current_section = title

    for section in sections:
        header_match = re.match(r'^(#{1,3})\s+(.+?)$', section, re.MULTILINE)
        section_title = header_match.group(2) if header_match else current_section

        words = section.split()
        if len(words) <= chunk_size:
            if len(current_chunk.split()) + len(words) <= chunk_size:
                current_chunk += "\n\n" + section
            else:
                if current_chunk.strip():
                    chunks.append({
                        "text": current_chunk.strip(),
                        "section": current_section,
                        "doc_id": doc_id,
                    })
                current_chunk = section
                current_section = section_title
        else:
            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "section": current_section,
                    "doc_id": doc_id,
                })
                current_chunk = ""

            paragraphs = section.split("\n\n")
            para_chunk = ""
            for para in paragraphs:
                if len(para_chunk.split()) + len(para.split()) <= chunk_size:
                    para_chunk += "\n\n" + para
                else:
                    if para_chunk.strip():
                        chunks.append({
                            "text": para_chunk.strip(),
                            "section": section_title,
                            "doc_id": doc_id,
                        })
                    para_chunk = para
            if para_chunk.strip():
                current_chunk = para_chunk
                current_section = section_title

    if current_chunk.strip():
        chunks.append({
            "text": current_chunk.strip(),
            "section": current_section,
            "doc_id": doc_id,
        })

    if overlap > 0 and len(chunks) > 1:
        for i in range(1, len(chunks)):
            prev_words = chunks[i - 1]["text"].split()
            overlap_text = " ".join(prev_words[-overlap:])
            chunks[i]["text"] = f"[...] {overlap_text}\n\n{chunks[i]['text']}"

    return chunks


def tokenize_for_bm25(text):
    """Lowercase word tokens, drop a small English stopword set."""
    if not isinstance(text, str):
        return []
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                 "being", "have", "has", "had", "do", "does", "did", "will",
                 "would", "could", "should", "may", "might", "shall", "can",
                 "to", "of", "in", "for", "on", "with", "at", "by", "from",
                 "as", "into", "through", "during", "before", "after", "and",
                 "but", "or", "nor", "not", "so", "yet", "both", "either",
                 "neither", "each", "every", "all", "any", "few", "more",
                 "most", "other", "some", "such", "no", "only", "own", "same",
                 "than", "too", "very", "just", "because", "if", "when", "that",
                 "this", "these", "those", "it", "its", "i", "me", "my", "we",
                 "our", "you", "your", "he", "him", "his", "she", "her", "they",
                 "them", "their", "what", "which", "who", "whom"}
    return [t for t in tokens if t not in stopwords and len(t) > 1]


class OllamaEmbedder:
    """Thin wrapper: .encode() batches to POST /api/embed like sentence-transformers."""

    def __init__(self, model: str = "qwen3:0.6b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        probe = self._embed_batch(["ping"])
        self._dim = len(probe[0])

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, texts, batch_size: int = 128, show_progress_bar: bool = False) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        all_embs = []
        for i in range(0, len(texts), batch_size):
            all_embs.extend(self._embed_batch(texts[i : i + batch_size]))
        return np.array(all_embs)

    def _embed_batch(self, texts):
        resp = self._session.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]


def build_vector_store(
    reset: bool = False,
    append: bool = False,
    batch_size: int = BATCH_SIZE,
    embedding_model: str | None = None,
    ollama_base_url: str | None = None,
):
    """Embed tickets + KB into Chroma, dump BM25 JSON files, write build_manifest."""

    model_id = embedding_model or EMBEDDING_MODEL
    base = ollama_base_url or resolve_ollama_base_url()

    print("[1/5] Loading embedding model...")
    model = OllamaEmbedder(model_id, base_url=base)
    actual_dim = model.get_sentence_embedding_dimension()
    print(f"  Ollama: {base}")
    print(f"  Model: {model_id}  |  dim: {actual_dim}")

    print("\n[2/5] Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    if reset:
        for name in (TICKET_COLLECTION, KB_COLLECTION):
            try:
                client.delete_collection(name)
                print(f"  Deleted collection: {name}")
            except Exception:
                pass

    ticket_collection = client.get_or_create_collection(
        name=TICKET_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    kb_collection = client.get_or_create_collection(
        name=KB_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"  Tickets in ChromaDB before build: {ticket_collection.count()}")
    print(f"  KB chunks in ChromaDB before build: {kb_collection.count()}")

    print("\n[3/5] Processing and embedding tickets...")
    tickets_path = PROCESSED_DIR / "all_tickets.csv"
    if not tickets_path.exists():
        print("  ❌ all_tickets.csv not found. Run 02_preprocess_data.py first.")
        return

    df = pd.read_csv(tickets_path).fillna("")
    print(f"  Loaded {len(df)} tickets from CSV")

    if not reset and ticket_collection.count() > 0:
        existing_ids = set(ticket_collection.get(include=[])["ids"])
        df_new = df[~df["unified_id"].isin(existing_ids)].reset_index(drop=True)
        skipped = len(df) - len(df_new)
        if skipped:
            print(f"  Skipping {skipped} already-indexed tickets; embedding {len(df_new)} new ones")
        df = df_new

    if len(df) == 0:
        print("  ✅ Nothing new to embed for tickets.")
    else:
        texts = df["embedding_text"].tolist()
        ids   = df["unified_id"].tolist()
        metadatas = [
            {
                "ticket_id":     str(row.get("ticket_id", "")),
                "title":         str(row.get("title_clean", ""))[:200],
                "category":      str(row.get("category", "")),
                "component":     str(row.get("component", "")),
                "department":    str(row.get("department", "")),
                "severity":      str(row.get("severity", "")),
                "status":        str(row.get("status", "")),
                "source":        str(row.get("source", "")),
                "source_system": str(row.get("source_system", row.get("external_source", ""))),
                "language":      str(row.get("language", "")),
                "quality_score": float(row.get("quality_score", 0)),
            }
            for _, row in df.iterrows()
        ]

        print(f"  Embedding {len(texts)} tickets (batch size: {batch_size})...")
        for i in tqdm(range(0, len(texts), batch_size), desc="  Tickets"):
            sl = slice(i, i + batch_size)
            ticket_collection.upsert(
                ids=ids[sl],
                embeddings=model.encode(texts[sl], show_progress_bar=False).tolist(),
                documents=texts[sl],
                metadatas=metadatas[sl],
            )

        print(f"  ✅ {ticket_collection.count()} tickets now in ChromaDB")

    print("\n  Building BM25 corpus for tickets (full CSV)...")
    df_full = pd.read_csv(tickets_path).fillna("")
    all_texts = df_full["embedding_text"].tolist()
    all_ids   = df_full["unified_id"].tolist()
    bm25_corpus_tickets = [tokenize_for_bm25(t) for t in all_texts]

    bm25_data = {"corpus": bm25_corpus_tickets, "ids": all_ids, "documents": all_texts}
    bm25_path = PROCESSED_DIR / "bm25_corpus_tickets.json"
    with open(bm25_path, "w") as f:
        json.dump(bm25_data, f)
    print(f"  ✅ BM25 corpus saved: {bm25_path}  ({len(all_ids)} docs)")

    print("\n[4/5] Processing and embedding knowledge base...")
    all_chunks = []
    chunk_counter = 0

    for filepath in sorted(KB_DIR.rglob("*.md")):
        with open(filepath, "r") as f:
            content = f.read()

        title = filepath.stem.replace("_", " ").title()
        doc_type = "runbook" if "runbook" in filepath.stem else "kb_article"

        chunks = chunk_document(content, title=title, doc_id=filepath.stem)
        print(f"  {filepath.name}: {len(chunks)} chunks")

        for j, chunk in enumerate(chunks):
            chunk_id = f"KB-{chunk_counter:05d}"
            all_chunks.append({
                "id": chunk_id,
                "text": chunk["text"],
                "doc_id": chunk["doc_id"],
                "doc_title": title,
                "doc_type": doc_type,
                "section": chunk["section"],
                "chunk_index": j,
                "source_file": filepath.name,
            })
            chunk_counter += 1

    if all_chunks:
        kb_texts = [c["text"] for c in all_chunks]
        kb_ids = [c["id"] for c in all_chunks]
        kb_metas = [{
            "doc_id": c["doc_id"],
            "doc_title": c["doc_title"],
            "doc_type": c["doc_type"],
            "section": c["section"],
            "chunk_index": c["chunk_index"],
            "source_file": c["source_file"],
        } for c in all_chunks]

        if not reset and kb_collection.count() > 0:
            existing_kb_ids = set(kb_collection.get(include=[])["ids"])
            new_chunk_mask = [c["id"] not in existing_kb_ids for c in all_chunks]
            all_chunks_new = [c for c, keep in zip(all_chunks, new_chunk_mask) if keep]
            print(f"  Skipping {len(all_chunks) - len(all_chunks_new)} existing KB chunks; "
                  f"embedding {len(all_chunks_new)} new ones")
            kb_texts  = [c["text"] for c in all_chunks_new]
            kb_ids    = [c["id"]   for c in all_chunks_new]
            kb_metas  = [{k: c[k] for k in ("doc_id","doc_title","doc_type","section","chunk_index","source_file")}
                         for c in all_chunks_new]
        print(f"  Embedding {len(kb_texts)} KB chunks...")
        for i in tqdm(range(0, len(kb_texts), batch_size), desc="  KB"):
            sl = slice(i, i + batch_size)
            kb_collection.upsert(
                ids=kb_ids[sl],
                embeddings=model.encode(kb_texts[sl], show_progress_bar=False).tolist(),
                documents=kb_texts[sl],
                metadatas=kb_metas[sl],
            )

        print(f"  ✅ {kb_collection.count()} KB chunks indexed in ChromaDB")

        bm25_corpus_kb = [tokenize_for_bm25(t) for t in kb_texts]
        bm25_kb_data = {
            "corpus": bm25_corpus_kb,
            "ids": kb_ids,
            "documents": kb_texts,
        }
        bm25_kb_path = PROCESSED_DIR / "bm25_corpus_kb.json"
        with open(bm25_kb_path, "w") as f:
            json.dump(bm25_kb_data, f)
        print(f"  ✅ BM25 KB corpus saved: {bm25_kb_path}")

        chunks_df = pd.DataFrame(all_chunks)
        chunks_df.to_csv(PROCESSED_DIR / "kb_chunks_index.csv", index=False)

    print("\n[5/5] Verification - test queries...")
    _run_verification(model, ticket_collection, kb_collection)

    manifest = {
        "built_at": datetime.datetime.utcnow().isoformat() + "Z",
        "embedding_model": model_id,
        "ollama_base_url": base,
        "embedding_dim": actual_dim,
        "ticket_count": ticket_collection.count(),
        "kb_chunk_count": kb_collection.count(),
        "reset": reset,
        "batch_size": batch_size,
    }
    manifest_path = PROCESSED_DIR / "build_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest saved: {manifest_path}")

    print("=" * 60)
    print("✅ Vector store build complete!")
    print(f"  Tickets: {ticket_collection.count()} vectors in '{TICKET_COLLECTION}'")
    print(f"  KB:      {kb_collection.count()} vectors in '{KB_COLLECTION}'")
    print(f"  ChromaDB: {CHROMA_PERSIST_DIR}")
    print("\nNext step: python 06_evaluation.py --ablation")
    print("=" * 60)


def _run_verification(model, ticket_collection, kb_collection):
    """Smoke-test a few queries so a broken embedder shows up immediately."""
    test_queries = [
        "VPN connection keeps dropping",
        "Outlook not syncing emails",
        "Laptop blue screen error",
        "Account locked out need password reset",
        "Software installation permission denied",
    ]

    print("\n  --- Ticket Search Test ---")
    for query in test_queries:
        q_emb = model.encode(query).tolist()
        results = ticket_collection.query(query_embeddings=[q_emb], n_results=3)
        print(f"  Q: '{query}'")
        for j, m in enumerate(results["metadatas"][0]):
            print(f"     [{j+1}] {m.get('title', 'N/A')[:70]}")
        print()

    print("  --- KB Search Test ---")
    for query in test_queries[:3]:
        q_emb = model.encode(query).tolist()
        results = kb_collection.query(query_embeddings=[q_emb], n_results=2)
        for j, doc in enumerate(results["documents"][0]):
            print(f"  Q: '{query}' → [{j+1}] {doc[:80].replace(chr(10),' ')}...")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ITS RAG - Build Vector Store")
    parser.add_argument("--reset", action="store_true",
                        help="Delete existing collections and rebuild from scratch.")
    parser.add_argument("--append", action="store_true",
                        help="Only embed IDs not already in ChromaDB (default behaviour).")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, metavar="N",
                        help=f"Texts per Ollama embedding call. Default: {BATCH_SIZE} "
                        f"(env ITS_EMBED_BATCH_SIZE)")
    parser.add_argument(
        "--embedding-model",
        default=default_embedding_model(),
        help="Ollama model for /api/embed (default: env ITS_EMBEDDING_MODEL or qwen3:0.6b).",
    )
    parser.add_argument(
        "--ollama-url",
        default=None,
        help="Ollama base URL, e.g. http://127.0.0.1:11434. Overrides ITS_OLLAMA_URL / OLLAMA_HOST.",
    )
    parser.add_argument("--verify-only", action="store_true",
                        help="Run test queries against existing index; skip all building.")
    args = parser.parse_args()

    ollama_url = resolve_ollama_base_url(args.ollama_url)

    if args.verify_only:
        print("Verify-only mode — loading existing index...")
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        tc = client.get_collection(TICKET_COLLECTION)
        kc = client.get_collection(KB_COLLECTION)
        m = OllamaEmbedder(args.embedding_model, base_url=ollama_url)
        _run_verification(m, tc, kc)
    else:
        build_vector_store(
            reset=args.reset,
            append=args.append,
            batch_size=args.batch_size,
            embedding_model=args.embedding_model,
            ollama_base_url=ollama_url,
        )
