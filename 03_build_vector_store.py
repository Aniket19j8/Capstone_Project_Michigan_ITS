"""
ITS RAG - Step 3: Build Vector Store & BM25 Index
Embeds all tickets and knowledge base documents, creates ChromaDB collections
and BM25 index for hybrid retrieval.

Creates:
  - ChromaDB collection: its_tickets (ticket embeddings)
  - ChromaDB collection: its_knowledge_base (KB document chunks)
  - data/processed/bm25_corpus_tickets.json (BM25 tokenized corpus)
  - data/processed/bm25_corpus_kb.json (BM25 tokenized KB corpus)

Usage:
  python 03_build_vector_store.py
  python 03_build_vector_store.py --reset  (rebuild from scratch)
"""

import os
import re
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CHROMA_PERSIST_DIR = "./data/chroma_db"
PROCESSED_DIR = Path("./data/processed")
KB_DIR = Path("./data/knowledge_base")

CHUNK_SIZE = 450       # tokens (~300-400 words)
CHUNK_OVERLAP = 50     # token overlap between chunks

TICKET_COLLECTION = "its_tickets"
KB_COLLECTION = "its_knowledge_base"


# ──────────────────────────────────────────────
# Chunking for Knowledge Base documents
# ──────────────────────────────────────────────
def chunk_document(text, title="", doc_id="", chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Recursive character text splitting that respects section boundaries.
    For tickets: no chunking needed (they're short).
    For KB docs/runbooks: chunk by sections, then by paragraphs.
    """
    chunks = []

    # First try to split by markdown headers (## or ###)
    sections = re.split(r'\n(?=#{1,3}\s)', text)

    current_chunk = ""
    current_section = title

    for section in sections:
        # Extract section header if present
        header_match = re.match(r'^(#{1,3})\s+(.+?)$', section, re.MULTILINE)
        section_title = header_match.group(2) if header_match else current_section

        # If section fits in one chunk, keep it together
        words = section.split()
        if len(words) <= chunk_size:
            if len(current_chunk.split()) + len(words) <= chunk_size:
                current_chunk += "\n\n" + section
            else:
                # Save current chunk and start new one
                if current_chunk.strip():
                    chunks.append({
                        "text": current_chunk.strip(),
                        "section": current_section,
                        "doc_id": doc_id,
                    })
                current_chunk = section
                current_section = section_title
        else:
            # Section too large, split by paragraphs
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

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append({
            "text": current_chunk.strip(),
            "section": current_section,
            "doc_id": doc_id,
        })

    # Add overlap context: prepend last N words from previous chunk
    if overlap > 0 and len(chunks) > 1:
        for i in range(1, len(chunks)):
            prev_words = chunks[i - 1]["text"].split()
            overlap_text = " ".join(prev_words[-overlap:])
            chunks[i]["text"] = f"[...] {overlap_text}\n\n{chunks[i]['text']}"

    return chunks


# ──────────────────────────────────────────────
# BM25 Tokenizer
# ──────────────────────────────────────────────
def tokenize_for_bm25(text):
    """Simple tokenization for BM25: lowercase, split, remove stopwords."""
    if not isinstance(text, str):
        return []
    # Lowercase and split on non-alphanumeric
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    # Remove very common stopwords (keep domain-relevant words)
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


# ──────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────
def build_vector_store(reset=False):
    """Build complete vector store and BM25 index."""

    # ── Load embedding model ──
    print("[1/5] Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  Model: {EMBEDDING_MODEL}")
    print(f"  Dimension: {model.get_sentence_embedding_dimension()}")

    # ── Initialize ChromaDB ──
    print("\n[2/5] Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    if reset:
        try:
            client.delete_collection(TICKET_COLLECTION)
        except:
            pass
        try:
            client.delete_collection(KB_COLLECTION)
        except:
            pass
        print("  Existing collections deleted.")

    ticket_collection = client.get_or_create_collection(
        name=TICKET_COLLECTION,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )
    kb_collection = client.get_or_create_collection(
        name=KB_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"  Ticket collection: {TICKET_COLLECTION}")
    print(f"  KB collection: {KB_COLLECTION}")

    # ── Process & Embed Tickets ──
    print("\n[3/5] Processing and embedding tickets...")
    tickets_path = PROCESSED_DIR / "all_tickets.csv"
    if not tickets_path.exists():
        print("  ❌ all_tickets.csv not found. Run 02_preprocess_data.py first.")
        return

    df = pd.read_csv(tickets_path)
    print(f"  Loaded {len(df)} tickets")

    # Prepare texts for embedding
    texts = df["embedding_text"].tolist()
    ids = df["unified_id"].tolist()

    # Prepare metadata for ChromaDB
    metadatas = []
    for _, row in df.iterrows():
        metadatas.append({
            "ticket_id": str(row.get("ticket_id", "")),
            "title": str(row.get("title_clean", ""))[:200],
            "category": str(row.get("category", "")),
            "component": str(row.get("component", "")),
            "severity": str(row.get("severity", "")),
            "status": str(row.get("status", "")),
            "source": str(row.get("source", "")),
            "quality_score": float(row.get("quality_score", 0)),
        })

    # Embed in batches
    BATCH_SIZE = 64
    print(f"  Embedding {len(texts)} tickets (batch size: {BATCH_SIZE})...")

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="  Embedding tickets"):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_ids = ids[i:i + BATCH_SIZE]
        batch_meta = metadatas[i:i + BATCH_SIZE]

        embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()

        ticket_collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_meta,
        )

    print(f"  ✅ {ticket_collection.count()} tickets indexed in ChromaDB")

    # ── Build BM25 index for tickets ──
    print("\n  Building BM25 corpus for tickets...")
    bm25_corpus_tickets = []
    for text in texts:
        bm25_corpus_tickets.append(tokenize_for_bm25(text))

    bm25_data = {
        "corpus": bm25_corpus_tickets,
        "ids": ids,
        "documents": texts,
    }
    bm25_path = PROCESSED_DIR / "bm25_corpus_tickets.json"
    with open(bm25_path, "w") as f:
        json.dump(bm25_data, f)
    print(f"  ✅ BM25 corpus saved: {bm25_path}")

    # ── Process & Embed Knowledge Base ──
    print("\n[4/5] Processing and embedding knowledge base...")
    all_chunks = []
    chunk_counter = 0

    for filepath in sorted(KB_DIR.glob("*.md")):
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

        # Embed KB chunks
        print(f"  Embedding {len(kb_texts)} KB chunks...")
        for i in tqdm(range(0, len(kb_texts), BATCH_SIZE), desc="  Embedding KB"):
            batch_texts = kb_texts[i:i + BATCH_SIZE]
            batch_ids = kb_ids[i:i + BATCH_SIZE]
            batch_meta = kb_metas[i:i + BATCH_SIZE]

            embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()

            kb_collection.add(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_meta,
            )

        print(f"  ✅ {kb_collection.count()} KB chunks indexed in ChromaDB")

        # BM25 for KB
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

        # Save chunk index
        chunks_df = pd.DataFrame(all_chunks)
        chunks_df.to_csv(PROCESSED_DIR / "kb_chunks_index.csv", index=False)

    # ── Verification ──
    print("\n[5/5] Verification - test queries...")
    test_queries = [
        "VPN connection keeps dropping",
        "Outlook not syncing emails",
        "Laptop blue screen error",
        "Account locked out need password reset",
        "Software installation permission denied",
    ]

    print("\n  --- Ticket Search Test ---")
    for query in test_queries:
        q_embedding = model.encode(query).tolist()
        results = ticket_collection.query(
            query_embeddings=[q_embedding],
            n_results=3,
        )
        top_titles = [m.get("title", "N/A")[:60] for m in results["metadatas"][0]]
        print(f"  Q: '{query}'")
        for j, title in enumerate(top_titles):
            print(f"     [{j+1}] {title}")
        print()

    print("  --- KB Search Test ---")
    for query in test_queries[:3]:
        q_embedding = model.encode(query).tolist()
        results = kb_collection.query(
            query_embeddings=[q_embedding],
            n_results=2,
        )
        for j, doc in enumerate(results["documents"][0]):
            snippet = doc[:80].replace("\n", " ")
            print(f"  Q: '{query}' → [{j+1}] {snippet}...")
        print()

    print("=" * 60)
    print("✅ Vector store build complete!")
    print(f"  Tickets: {ticket_collection.count()} vectors in '{TICKET_COLLECTION}'")
    print(f"  KB:      {kb_collection.count()} vectors in '{KB_COLLECTION}'")
    print(f"  ChromaDB: {CHROMA_PERSIST_DIR}")
    print("\nNext step: python 04_hybrid_retrieval.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild collections")
    args = parser.parse_args()
    build_vector_store(reset=args.reset)
