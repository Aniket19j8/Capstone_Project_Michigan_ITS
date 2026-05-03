"""
07 — self-retrieval torture test: each ticket queries itself, we print recall@k / MRR.

Writes evaluation/500_query_results.csv (+ summary json + chart when matplotlib happy).

Usage:
  python 07_run_500_eval.py
"""

import json
import time
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──
PROCESSED_DIR = Path("./data/processed")
EVAL_DIR = Path("./evaluation")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ── Load retriever module (numeric filename requires importlib) ──
_retriever_path = Path(__file__).parent / "04_hybrid_retrieval.py"
_spec = importlib.util.spec_from_file_location("hybrid_retrieval_engine", _retriever_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
HybridRetriever = _mod.HybridRetriever


# ── Metric Functions ──
def recall_at_k(target_id: str, retrieved_ids: list, k: int) -> float:
    return 1.0 if target_id in retrieved_ids[:k] else 0.0


def mrr_score(target_id: str, retrieved_ids: list) -> float:
    for i, rid in enumerate(retrieved_ids):
        if rid == target_id:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(target_id: str, retrieved_ids: list, k: int) -> float:
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid == target_id:
            return 1.0 / np.log2(i + 2)
    return 0.0


def run_evaluation():
    print("=" * 60)
    print("ITS RAG — 500-Query Self-Retrieval Evaluation")
    print("=" * 60)

    # ── Load tickets ──
    tickets_path = PROCESSED_DIR / "all_tickets.csv"
    if not tickets_path.exists():
        print(f"❌ {tickets_path} not found. Run 02_preprocess_data.py first.")
        return

    df = pd.read_csv(tickets_path)
    print(f"Loaded {len(df)} tickets\n")

    # ── Initialize retriever ──
    print("Loading retriever (takes 10-20s on first run)...")
    retriever = HybridRetriever("its_tickets", load_reranker=True)

    # ── Define methods ──
    methods = {
        "Dense_Only": lambda q: retriever.search_dense_only(q, top_k=10),
        "BM25_Only": lambda q: retriever.search_bm25_only(q, top_k=10),
        "Hybrid_NoRerank": lambda q: retriever.search(q, top_k=10, use_reranking=False),
        "Hybrid_Rerank": lambda q: retriever.search(q, top_k=10, use_reranking=True),
    }

    all_results = []
    total = len(df)
    print(f"\nRunning {total} queries × {len(methods)} methods...")
    print("Estimated time: 10-30 minutes.\n")

    for idx, row in df.iterrows():
        # Use the same text that was embedded
        query = str(row.get("embedding_text",
                    row.get("description_clean",
                    row.get("title_clean", ""))))
        if not query.strip():
            continue

        target_id = str(row["unified_id"])

        if idx % 50 == 0:
            print(f"  Progress: {idx}/{total} ({idx / total * 100:.0f}%)")

        rr = {
            "query_idx": idx,
            "unified_id": target_id,
            "ticket_id": str(row.get("ticket_id", "")),
            "category": str(row.get("category", "")),
            "severity": str(row.get("severity", "")),
            "query_length": len(query),
        }

        for method_name, search_fn in methods.items():
            try:
                t0 = time.time()
                results = search_fn(query)
                latency = time.time() - t0
                retrieved_ids = [r["id"] for r in results]

                for k in [1, 3, 5, 10]:
                    rr[f"{method_name}_recall@{k}"] = recall_at_k(target_id, retrieved_ids, k)
                rr[f"{method_name}_mrr"] = mrr_score(target_id, retrieved_ids)
                for k in [5, 10]:
                    rr[f"{method_name}_ndcg@{k}"] = ndcg_at_k(target_id, retrieved_ids, k)
                rr[f"{method_name}_latency_ms"] = round(latency * 1000, 1)

            except Exception as e:
                print(f"  ⚠ Error query {idx} ({method_name}): {e}")
                for k in [1, 3, 5, 10]:
                    rr[f"{method_name}_recall@{k}"] = 0
                rr[f"{method_name}_mrr"] = 0
                rr[f"{method_name}_ndcg@5"] = 0
                rr[f"{method_name}_ndcg@10"] = 0
                rr[f"{method_name}_latency_ms"] = 0

        all_results.append(rr)

    # ── Save raw results ──
    rdf = pd.DataFrame(all_results)
    rdf.to_csv(EVAL_DIR / "500_query_results.csv", index=False)

    # ── Compute summary per method ──
    summary = {}
    for mn in methods:
        s = {}
        for met in ["recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg@5", "ndcg@10"]:
            col = f"{mn}_{met}"
            s[met] = round(rdf[col].mean(), 4) if col in rdf.columns else 0
        col_lat = f"{mn}_latency_ms"
        s["avg_latency_ms"] = round(rdf[col_lat].mean(), 1) if col_lat in rdf.columns else 0
        summary[mn] = s

    # ── Per-category breakdown (best method) ──
    best = "Hybrid_Rerank"
    cat_summary = {}
    for cat in sorted(rdf["category"].dropna().unique()):
        cd = rdf[rdf["category"] == cat]
        cat_summary[cat] = {
            "recall@5": round(cd[f"{best}_recall@5"].mean(), 3),
            "mrr": round(cd[f"{best}_mrr"].mean(), 3),
            "n": int(len(cd)),
        }

    # ── Per-severity breakdown ──
    sev_summary = {}
    for sev in ["Critical", "High", "Medium", "Low"]:
        sd = rdf[rdf["severity"] == sev]
        if len(sd) > 0:
            sev_summary[sev] = {
                "recall@5": round(sd[f"{best}_recall@5"].mean(), 3),
                "mrr": round(sd[f"{best}_mrr"].mean(), 3),
                "n": int(len(sd)),
            }

    full_summary = {
        "overall": summary,
        "per_category": cat_summary,
        "per_severity": sev_summary,
        "total_queries": int(len(rdf)),
    }
    with open(EVAL_DIR / "metrics_summary.json", "w") as f:
        json.dump(full_summary, f, indent=2)

    # ── Generate comparison chart ──
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    colors = ["#5B8DEF", "#F0983A", "#A78BFA", "#3DD9B4"]
    method_names = list(methods.keys())

    for ax, met, col in zip(axes, ["recall@1", "recall@5", "mrr", "ndcg@5"], colors):
        vals = [summary[m].get(met, 0) for m in method_names]
        bars = ax.bar(range(len(method_names)), vals, color=col, alpha=0.85)
        ax.set_title(met.upper(), fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(method_names)))
        ax.set_xticklabels([m.replace("_", "\n") for m in method_names], fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                    f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")

    plt.suptitle("ITS RAG — 500-Query Retrieval Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "retrieval_comparison_chart.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Print tables for quad chart ──
    print(f"\n{'=' * 80}")
    print("COPY THESE NUMBERS INTO YOUR QUAD CHART")
    print(f"{'=' * 80}")
    print(f"{'Method':<20} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'nDCG@5':>7} {'Lat(ms)':>8}")
    print("-" * 70)
    for m in method_names:
        s = summary[m]
        print(f"{m:<20} {s['recall@1']:>6.3f} {s['recall@5']:>6.3f} "
              f"{s['recall@10']:>6.3f} {s['mrr']:>6.3f} "
              f"{s['ndcg@5']:>7.3f} {s['avg_latency_ms']:>7.1f}")

    print(f"\nPER-CATEGORY ({best}):")
    for c, v in cat_summary.items():
        print(f"  {c:12s}: R@5={v['recall@5']:.3f}  MRR={v['mrr']:.3f}  (n={v['n']})")

    print(f"\nPER-SEVERITY ({best}):")
    for sv, v in sev_summary.items():
        print(f"  {sv:10s}: R@5={v['recall@5']:.3f}  MRR={v['mrr']:.3f}  (n={v['n']})")

    print(f"\n✅ evaluation/500_query_results.csv")
    print(f"✅ evaluation/metrics_summary.json")
    print(f"✅ evaluation/retrieval_comparison_chart.png")


if __name__ == "__main__":
    run_evaluation()
