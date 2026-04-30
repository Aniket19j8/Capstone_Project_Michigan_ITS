"""
ITS RAG - Step 4: Hybrid Retrieval Engine
Implements the complete retrieval pipeline:
  1. Dense vector search (ChromaDB / cosine similarity)
  2. BM25 keyword search (exact matches for error codes, IDs)
  3. Reciprocal Rank Fusion (RRF) to merge results
  4. Cross-encoder reranking for final precision

Usage:
  python 04_hybrid_retrieval.py                    # Interactive mode
  python 04_hybrid_retrieval.py --query "VPN drops" # Single query
  python 04_hybrid_retrieval.py --evaluate          # Run eval suite
"""

import json
import argparse
import requests
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb
import re


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
EMBEDDING_MODEL = "qwen3:0.6b"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHROMA_PERSIST_DIR = "./data/chroma_db"
PROCESSED_DIR = Path("./data/processed")

TICKET_COLLECTION = "its_tickets"
KB_COLLECTION = "its_knowledge_base"

# Retrieval params
DENSE_TOP_K = 20        # Candidates from vector search
BM25_TOP_K = 20         # Candidates from BM25
RRF_K = 60              # RRF constant (standard default)
BM25_WEIGHT = 0.4       # Weight for BM25 in hybrid
DENSE_WEIGHT = 0.6      # Weight for dense in hybrid
RERANK_TOP_K = 5        # Final results after reranking


# ──────────────────────────────────────────────
# BM25 Tokenizer (same as build step)
# ──────────────────────────────────────────────
def tokenize_for_bm25(text):
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


# ──────────────────────────────────────────────
# Ollama Embedder (replaces SentenceTransformer)
# ──────────────────────────────────────────────
class OllamaEmbedder:
    """Wraps Ollama /api/embed to match the SentenceTransformer .encode() interface."""

    def __init__(self, model: str = "qwen3:0.6b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()  # reuse TCP connections across calls
        probe = self._embed_batch(["ping"])
        self._dim = len(probe[0])

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, texts, batch_size: int = 128, show_progress_bar: bool = False) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        all_embs: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            all_embs.extend(self._embed_batch(texts[i : i + batch_size]))
        return np.array(all_embs)

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        resp = self._session.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]


# ──────────────────────────────────────────────
# Hybrid Retriever Class
# ──────────────────────────────────────────────
class HybridRetriever:
    """
    Complete hybrid retrieval pipeline:
    Dense (ChromaDB) + BM25 → RRF Fusion → Cross-Encoder Reranking
    """

    def __init__(self, collection_name: str = TICKET_COLLECTION, load_reranker: bool = True):
        print("Initializing Hybrid Retriever...")

        # Load embedding model
        print("  Loading embedding model...")
        self.embedder = OllamaEmbedder(EMBEDDING_MODEL)

        # Load cross-encoder reranker
        if load_reranker:
            print("  Loading cross-encoder reranker...")
            self.reranker = CrossEncoder(RERANKER_MODEL)
        else:
            self.reranker = None

        # Connect to ChromaDB
        print("  Connecting to ChromaDB...")
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.client.get_collection(collection_name)
        print(f"  Collection '{collection_name}': {self.collection.count()} documents")

        # Load BM25 corpus
        bm25_suffix = "tickets" if "ticket" in collection_name else "kb"
        bm25_path = PROCESSED_DIR / f"bm25_corpus_{bm25_suffix}.json"
        print(f"  Loading BM25 corpus from {bm25_path}...")
        with open(bm25_path, "r") as f:
            bm25_data = json.load(f)

        self.bm25_corpus = bm25_data["corpus"]
        self.bm25_ids = bm25_data["ids"]
        self.bm25_documents = bm25_data["documents"]
        self.bm25 = BM25Okapi(self.bm25_corpus)

        print(f"  BM25 index: {len(self.bm25_corpus)} documents")
        print("  [OK] Retriever ready!\n")

    # ── Dense Search ──
    def dense_search(self, query: str, top_k: int = DENSE_TOP_K,
                     filters: Optional[Dict] = None) -> List[Dict]:
        """Vector similarity search via ChromaDB."""
        query_embedding = self.embedder.encode(query).tolist()

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }
        if filters:
            kwargs["where"] = filters

        results = self.collection.query(**kwargs)

        docs = []
        for i in range(len(results["ids"][0])):
            docs.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else 0,
                "score": 1 - results["distances"][0][i] if results["distances"] else 0,  # cosine sim
                "source": "dense",
            })
        return docs

    # ── BM25 Search ──
    def bm25_search(self, query: str, top_k: int = BM25_TOP_K) -> List[Dict]:
        """BM25 keyword search."""
        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        docs = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include matches
                docs.append({
                    "id": self.bm25_ids[idx],
                    "text": self.bm25_documents[idx],
                    "metadata": {},
                    "score": float(scores[idx]),
                    "source": "bm25",
                })
        return docs

    # ── Reciprocal Rank Fusion ──
    def reciprocal_rank_fusion(self, dense_results: List[Dict],
                                bm25_results: List[Dict],
                                k: int = RRF_K,
                                dense_weight: float = DENSE_WEIGHT,
                                bm25_weight: float = BM25_WEIGHT) -> List[Dict]:
        """
        Merge results from dense and BM25 using Reciprocal Rank Fusion.
        RRF score = Σ weight / (k + rank)
        """
        rrf_scores = {}
        doc_map = {}

        # Score dense results
        for rank, doc in enumerate(dense_results):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + dense_weight / (k + rank + 1)
            doc_map[doc_id] = doc

        # Score BM25 results
        for rank, doc in enumerate(bm25_results):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + bm25_weight / (k + rank + 1)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        fused = []
        for doc_id in sorted_ids:
            doc = doc_map[doc_id].copy()
            doc["rrf_score"] = rrf_scores[doc_id]
            doc["source"] = "hybrid"
            fused.append(doc)

        return fused

    # ── Cross-Encoder Reranking ──
    def rerank(self, query: str, candidates: List[Dict],
               top_k: int = RERANK_TOP_K) -> List[Dict]:
        """Rerank candidates using cross-encoder model."""
        if not self.reranker or not candidates:
            return candidates[:top_k]

        # Prepare pairs for cross-encoder
        pairs = [(query, doc["text"][:512]) for doc in candidates]  # Truncate for speed

        # Score all pairs
        scores = self.reranker.predict(pairs)

        # Attach scores and sort
        for i, doc in enumerate(candidates):
            doc["rerank_score"] = float(scores[i])

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    # ── Full Hybrid Pipeline ──
    def search(self, query: str, top_k: int = RERANK_TOP_K,
               filters: Optional[Dict] = None,
               use_reranking: bool = True,
               verbose: bool = False) -> List[Dict]:
        """
        Complete hybrid search pipeline:
        1. Dense search (ChromaDB)
        2. BM25 search
        3. RRF fusion
        4. Cross-encoder reranking
        """
        # Step 1: Dense search
        dense_results = self.dense_search(query, top_k=DENSE_TOP_K, filters=filters)
        if verbose:
            print(f"  Dense: {len(dense_results)} results")

        # Step 2: BM25 search
        bm25_results = self.bm25_search(query, top_k=BM25_TOP_K)
        if verbose:
            print(f"  BM25:  {len(bm25_results)} results")

        # Step 3: RRF fusion
        fused = self.reciprocal_rank_fusion(dense_results, bm25_results)
        if verbose:
            print(f"  RRF:   {len(fused)} unique results")

        # Step 4: Reranking
        if use_reranking:
            final = self.rerank(query, fused, top_k=top_k)
            if verbose:
                print(f"  Reranked: {len(final)} results")
        else:
            final = fused[:top_k]

        return final

    # ── Convenience: Search only dense ──
    def search_dense_only(self, query: str, top_k: int = 5) -> List[Dict]:
        """Dense-only search (for comparison/ablation)."""
        return self.dense_search(query, top_k=top_k)

    # ── Convenience: Search only BM25 ──
    def search_bm25_only(self, query: str, top_k: int = 5) -> List[Dict]:
        """BM25-only search (for comparison/ablation)."""
        return self.bm25_search(query, top_k=top_k)


# ──────────────────────────────────────────────
# Knowledge Base Retriever (same class, different collection)
# ──────────────────────────────────────────────
class KBRetriever(HybridRetriever):
    """Retriever specifically for Knowledge Base documents."""

    def __init__(self, load_reranker=True):
        super().__init__(collection_name=KB_COLLECTION, load_reranker=load_reranker)


# ──────────────────────────────────────────────
# Combined Retriever (tickets + KB)
# ──────────────────────────────────────────────
class CombinedRetriever:
    """Retrieves from both tickets and knowledge base, merges results."""

    def __init__(self):
        print("=" * 50)
        print("Initializing Combined Retriever (Tickets + KB)")
        print("=" * 50)
        self.ticket_retriever = HybridRetriever(TICKET_COLLECTION)
        self.kb_retriever = HybridRetriever(KB_COLLECTION)

    def search(self, query: str, ticket_top_k: int = 3,
               kb_top_k: int = 3, verbose: bool = False) -> Dict:
        """Search both tickets and KB, return combined results."""
        ticket_results = self.ticket_retriever.search(
            query, top_k=ticket_top_k, verbose=verbose
        )
        kb_results = self.kb_retriever.search(
            query, top_k=kb_top_k, verbose=verbose
        )

        return {
            "similar_tickets": ticket_results,
            "knowledge_base": kb_results,
            "query": query,
        }


# ──────────────────────────────────────────────
# Pretty Print Results
# ──────────────────────────────────────────────
def print_results(results: List[Dict], title: str = "Results"):
    """Pretty print search results."""
    print(f"\n{'─' * 60}")
    print(f"  {title} ({len(results)} results)")
    print(f"{'─' * 60}")
    for i, doc in enumerate(results):
        score_str = ""
        if "rerank_score" in doc:
            score_str = f"rerank={doc['rerank_score']:.4f}"
        elif "rrf_score" in doc:
            score_str = f"rrf={doc['rrf_score']:.6f}"
        elif "score" in doc:
            score_str = f"score={doc['score']:.4f}"

        title_text = doc.get("metadata", {}).get("title", "")[:60]
        if not title_text:
            title_text = doc["text"][:60].replace("\n", " ")

        source = doc.get("source", "?")
        print(f"  [{i+1}] {title_text}")
        print(f"      {score_str} | source: {source} | id: {doc['id']}")
        if doc.get("metadata", {}).get("severity"):
            print(f"      severity: {doc['metadata']['severity']} | "
                  f"category: {doc['metadata'].get('category', 'N/A')}")
        print()


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
def run_evaluation():
    """Compare retrieval methods: Dense vs BM25 vs Hybrid vs Hybrid+Rerank."""
    print("\n" + "=" * 60)
    print("Retrieval Method Comparison (Ablation Study)")
    print("=" * 60)

    retriever = HybridRetriever(TICKET_COLLECTION, load_reranker=True)

    test_queries = [
        # Semantic queries (dense should win)
        "My computer won't start up in the morning",
        "Email application is extremely slow",
        "Cannot access company resources remotely",

        # Keyword-heavy queries (BM25 should help)
        "error code 0x80070005",
        "BSOD IRQL_NOT_LESS_OR_EQUAL",
        "VPN client version 4.2",

        # Mixed queries (hybrid should win)
        "Outlook not syncing after Windows update",
        "printer offline error on third floor",
        "Teams consuming too much memory",
        "account locked need reset MFA",
    ]

    print(f"\nRunning {len(test_queries)} test queries across 4 methods...\n")

    for query in test_queries:
        print(f"Q: \"{query}\"")
        print("-" * 50)

        # Method 1: Dense only
        dense = retriever.search_dense_only(query, top_k=3)
        d_titles = [d.get("metadata", {}).get("title", d["text"][:40]) for d in dense]

        # Method 2: BM25 only
        bm25 = retriever.search_bm25_only(query, top_k=3)
        b_titles = [d.get("text", "")[:40] for d in bm25]

        # Method 3: Hybrid (no rerank)
        hybrid = retriever.search(query, top_k=3, use_reranking=False)
        h_titles = [d.get("metadata", {}).get("title", d["text"][:40]) for d in hybrid]

        # Method 4: Hybrid + Rerank
        hybrid_rr = retriever.search(query, top_k=3, use_reranking=True)
        hr_titles = [d.get("metadata", {}).get("title", d["text"][:40]) for d in hybrid_rr]

        print(f"  Dense:          {d_titles[0][:50] if d_titles else 'No results'}")
        print(f"  BM25:           {b_titles[0][:50] if b_titles else 'No results'}")
        print(f"  Hybrid:         {h_titles[0][:50] if h_titles else 'No results'}")
        print(f"  Hybrid+Rerank:  {hr_titles[0][:50] if hr_titles else 'No results'}")
        print()

    print("✅ Evaluation complete! Use these results in your Status 1 presentation.")


# ──────────────────────────────────────────────
# Interactive Mode
# ──────────────────────────────────────────────
def interactive_mode():
    """Interactive search REPL."""
    print("\n" + "=" * 60)
    print("ITS Hybrid Retrieval - Interactive Mode")
    print("Commands: 'quit', 'kb' (switch to KB), 'tickets' (switch to tickets)")
    print("=" * 60)

    retriever = HybridRetriever(TICKET_COLLECTION)
    kb_retriever = None
    active = "tickets"

    while True:
        query = input(f"\n[{active}] Search: ").strip()
        if not query:
            continue
        if query.lower() == "quit":
            break
        if query.lower() == "kb":
            if kb_retriever is None:
                kb_retriever = HybridRetriever(KB_COLLECTION)
            active = "kb"
            print("Switched to Knowledge Base search.")
            continue
        if query.lower() == "tickets":
            active = "tickets"
            print("Switched to Ticket search.")
            continue

        r = retriever if active == "tickets" else kb_retriever
        results = r.search(query, top_k=5, verbose=True)
        print_results(results, f"Hybrid Search Results ({active})")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ITS Hybrid Retrieval Engine")
    parser.add_argument("--query", type=str, help="Single query to search")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation suite")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--collection", choices=["tickets", "kb"], default="tickets")
    args = parser.parse_args()

    if args.evaluate:
        run_evaluation()
    elif args.query:
        collection = TICKET_COLLECTION if args.collection == "tickets" else KB_COLLECTION
        retriever = HybridRetriever(collection)
        results = retriever.search(args.query, top_k=args.top_k, verbose=True)
        print_results(results)
    else:
        interactive_mode()
