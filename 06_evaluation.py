"""
ITS RAG - Step 6: Evaluation & Benchmarking
Metrics: Recall@k, Precision@k, MRR, nDCG, Dedup F1, Threshold Optimization

Usage:
  python 06_evaluation.py --all
  python 06_evaluation.py --retrieval
  python 06_evaluation.py --dedup
  python 06_evaluation.py --ablation
"""

import json
import argparse
import importlib.util
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROCESSED_DIR = Path("./data/processed")
EVAL_DIR = Path("./evaluation")
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def _configure_stdout_utf8():
    """Avoid UnicodeEncodeError on Windows terminals with legacy encodings."""
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _load_hybrid_retriever_class():
    """Load HybridRetriever from 04_hybrid_retrieval.py (numeric filename)."""
    retriever_path = Path(__file__).parent / "04_hybrid_retrieval.py"
    spec = importlib.util.spec_from_file_location("hybrid_retrieval_engine", retriever_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec from {retriever_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HybridRetriever


# ── Retrieval Metrics ──
def recall_at_k(relevant, retrieved, k):
    if not relevant: return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)

def precision_at_k(relevant, retrieved, k):
    if k == 0: return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / k

def mrr(relevant, retrieved):
    for i, doc_id in enumerate(retrieved):
        if doc_id in set(relevant):
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(relevant, retrieved, k):
    def dcg(ids, rel_set, k):
        score = 0.0
        for i, doc_id in enumerate(ids[:k]):
            if doc_id in rel_set:
                score += 1.0 / np.log2(i + 2)
        return score
    rel_set = set(relevant)
    actual = dcg(retrieved, rel_set, k)
    ideal = dcg(relevant, rel_set, k)
    return actual / ideal if ideal > 0 else 0.0


def evaluate_retrieval(retriever, test_queries: List[Dict], ks=[1, 3, 5, 10]):
    """
    Evaluate retrieval quality.
    test_queries: [{"query": "...", "relevant_ids": ["id1", "id2"]}]
    """
    results = {f"recall@{k}": [] for k in ks}
    results.update({f"precision@{k}": [] for k in ks})
    results["mrr"] = []
    results.update({f"ndcg@{k}": [] for k in ks})

    for tq in test_queries:
        query = tq["query"]
        relevant = tq["relevant_ids"]
        retrieved_docs = retriever.search(query, top_k=max(ks))
        retrieved_ids = [d["id"] for d in retrieved_docs]

        for k in ks:
            results[f"recall@{k}"].append(recall_at_k(relevant, retrieved_ids, k))
            results[f"precision@{k}"].append(precision_at_k(relevant, retrieved_ids, k))
            results[f"ndcg@{k}"].append(ndcg_at_k(relevant, retrieved_ids, k))
        results["mrr"].append(mrr(relevant, retrieved_ids))

    # Average
    avg = {metric: round(np.mean(vals), 4) for metric, vals in results.items()}
    return avg


# ── Duplicate Detection Evaluation ──
def evaluate_dedup_threshold(embedder, ticket_pairs: pd.DataFrame, thresholds=None):
    """
    Evaluate duplicate detection at various thresholds.
    ticket_pairs: DataFrame with columns [text1, text2, is_duplicate]
    """
    from sentence_transformers import util

    if thresholds is None:
        thresholds = np.arange(0.5, 0.98, 0.02)

    # Compute similarities
    texts1 = ticket_pairs["text1"].tolist()
    texts2 = ticket_pairs["text2"].tolist()
    labels = ticket_pairs["is_duplicate"].values

    emb1 = embedder.encode(texts1, show_progress_bar=True)
    emb2 = embedder.encode(texts2, show_progress_bar=True)
    similarities = util.cos_sim(emb1, emb2).diagonal().numpy()

    # Evaluate at each threshold
    results = []
    for tau in thresholds:
        preds = (similarities >= tau).astype(int)
        p = precision_score(labels, preds, zero_division=0)
        r = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        results.append({
            "threshold": round(float(tau), 3),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        })

    df_results = pd.DataFrame(results)

    # Find optimal threshold (max F1)
    best_idx = df_results["f1"].idxmax()
    best = df_results.iloc[best_idx]
    print(f"\n  Optimal threshold: τ = {best['threshold']}")
    print(f"  Precision: {best['precision']:.4f}")
    print(f"  Recall: {best['recall']:.4f}")
    print(f"  F1: {best['f1']:.4f}")

    # Cost-sensitive optimization
    # c_FP = cost of false positive (user annoyed by wrong duplicate flag)
    # c_FN = cost of false negative (missed duplicate wastes engineering time)
    c_fp, c_fn = 1.0, 2.0  # False negatives cost more
    df_results["cost"] = c_fp * df_results["false_positives"] + c_fn * df_results["false_negatives"]
    cost_best_idx = df_results["cost"].idxmin()
    cost_best = df_results.iloc[cost_best_idx]
    print(f"\n  Cost-optimal threshold (c_FP={c_fp}, c_FN={c_fn}): τ = {cost_best['threshold']}")
    print(f"  Cost: {cost_best['cost']:.1f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1, ax2 = axes

    ax1.plot(df_results["threshold"], df_results["precision"], "b-", label="Precision")
    ax1.plot(df_results["threshold"], df_results["recall"], "r-", label="Recall")
    ax1.plot(df_results["threshold"], df_results["f1"], "g--", label="F1", linewidth=2)
    ax1.axvline(x=best["threshold"], color="gray", linestyle=":", label=f"Best τ={best['threshold']}")
    ax1.set_xlabel("Threshold (τ)")
    ax1.set_ylabel("Score")
    ax1.set_title("Duplicate Detection: Precision/Recall/F1 vs Threshold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(df_results["threshold"], df_results["cost"], "m-", linewidth=2)
    ax2.axvline(x=cost_best["threshold"], color="gray", linestyle=":")
    ax2.set_xlabel("Threshold (τ)")
    ax2.set_ylabel(f"Cost (c_FP={c_fp}, c_FN={c_fn})")
    ax2.set_title("Cost-Sensitive Threshold Optimization")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = EVAL_DIR / "dedup_threshold_analysis.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\n  Plot saved: {plot_path}")
    plt.close()

    # Save results
    df_results.to_csv(EVAL_DIR / "dedup_threshold_results.csv", index=False)
    return df_results, best, cost_best


# ── Ablation Study ──
def run_ablation_study():
    """
    Compare retrieval methods:
    1. Dense only
    2. BM25 only
    3. Hybrid (Dense + BM25 + RRF)
    4. Hybrid + Cross-Encoder Reranking
    """
    HybridRetriever = _load_hybrid_retriever_class()

    print("\n" + "=" * 60)
    print("ABLATION STUDY: Retrieval Method Comparison")
    print("=" * 60)

    retriever = HybridRetriever("its_tickets", load_reranker=True)

    # Test queries with expected behavior notes
    test_queries = [
        {"query": "VPN keeps disconnecting every few minutes", "type": "semantic"},
        {"query": "error 0x80070005 permission denied", "type": "keyword"},
        {"query": "laptop overheating and fan is loud", "type": "semantic"},
        {"query": "BSOD KERNEL_DATA_INPAGE_ERROR", "type": "keyword"},
        {"query": "Outlook not syncing emails after update", "type": "mixed"},
        {"query": "account locked out MFA not working", "type": "mixed"},
        {"query": "Teams slow high memory usage", "type": "semantic"},
        {"query": "printer queue stuck cannot print", "type": "mixed"},
        {"query": "new hire account setup Active Directory", "type": "keyword"},
        {"query": "WiFi intermittent disconnections building 3", "type": "mixed"},
    ]

    methods = {
        "Dense Only": lambda q: retriever.search_dense_only(q, top_k=5),
        "BM25 Only": lambda q: retriever.search_bm25_only(q, top_k=5),
        "Hybrid (no rerank)": lambda q: retriever.search(q, top_k=5, use_reranking=False),
        "Hybrid + Rerank": lambda q: retriever.search(q, top_k=5, use_reranking=True),
    }

    results_table = []
    for tq in test_queries:
        query = tq["query"]
        row = {"query": query[:40], "type": tq["type"]}

        for method_name, search_fn in methods.items():
            results = search_fn(query)
            if results:
                top_title = results[0].get("metadata", {}).get("title", results[0]["text"][:30])
                top_score = results[0].get("rerank_score", results[0].get("rrf_score", results[0].get("score", 0)))
                row[f"{method_name}_top1"] = top_title[:35]
                row[f"{method_name}_score"] = round(float(top_score), 4)
            else:
                row[f"{method_name}_top1"] = "No results"
                row[f"{method_name}_score"] = 0

        results_table.append(row)

    df_ablation = pd.DataFrame(results_table)
    df_ablation.to_csv(EVAL_DIR / "ablation_results.csv", index=False)

    # Print summary
    print("\nTop-1 Results by Method:")
    print("-" * 100)
    for _, row in df_ablation.iterrows():
        print(f"\nQ: {row['query']} [{row['type']}]")
        for method in methods.keys():
            print(f"  {method:25s}: {row.get(f'{method}_top1', 'N/A')}")

    print(f"\n✅ Ablation results saved: {EVAL_DIR / 'ablation_results.csv'}")
    return df_ablation


# ── Main ──
if __name__ == "__main__":
    _configure_stdout_utf8()

    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--retrieval", action="store_true")
    parser.add_argument("--dedup", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()

    if args.ablation or args.all:
        run_ablation_study()

    if args.dedup or args.all:
        print("\n[Dedup Evaluation]")
        pairs_path = PROCESSED_DIR / "synthetic_duplicate_pairs.csv"
        if pairs_path.exists():
            # Use the same OllamaEmbedder as the rest of the pipeline
            import importlib.util as _ilu
            _spec = _ilu.spec_from_file_location(
                "hybrid_retrieval_engine",
                Path(__file__).parent / "04_hybrid_retrieval.py",
            )
            _mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            embedder = _mod.OllamaEmbedder("qwen3:0.6b")

            pairs = pd.read_csv(pairs_path)
            tickets = pd.read_csv(PROCESSED_DIR / "all_tickets.csv").fillna("")
            id_to_text = dict(zip(tickets["unified_id"], tickets["embedding_text"]))
            pairs["text1"] = pairs["id1"].map(id_to_text).fillna("")
            pairs["text2"] = pairs["id2"].map(id_to_text).fillna("")
            pairs = pairs[(pairs["text1"].str.len() > 5) & (pairs["text2"].str.len() > 5)]
            if len(pairs) > 0:
                evaluate_dedup_threshold(embedder, pairs)
            else:
                print("  ⚠ Pairs file exists but no matching texts found in all_tickets.csv")
        else:
            print("  No duplicate pairs found at", pairs_path)

    print("\n✅ Evaluation complete! Results in ./evaluation/")
