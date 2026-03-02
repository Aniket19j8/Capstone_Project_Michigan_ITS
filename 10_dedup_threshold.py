"""
ITS RAG - Script 10: Duplicate Detection Threshold Optimization
================================================================
Evaluates duplicate detection at various cosine similarity thresholds.
Uses the labeled duplicate pairs from synthetic data.

Outputs: evaluation/dedup_threshold_results.csv
         evaluation/dedup_threshold_chart.png
         evaluation/dedup_summary.json

Usage:  python 10_dedup_threshold.py
"""

import json, numpy as np, pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_score, recall_score, f1_score

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PROCESSED_DIR = Path("./data/processed")
EVAL_DIR = Path("./evaluation"); EVAL_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def run_dedup_eval():
    print("="*60+"\nITS RAG — Duplicate Detection Threshold Optimization\n"+"="*60)

    pairs_path = PROCESSED_DIR / "synthetic_duplicate_pairs.csv"
    tickets_path = PROCESSED_DIR / "all_tickets.csv"

    if not pairs_path.exists():
        print(f"❌ {pairs_path} not found. Run 01_download_data.py --generate-synthetic first.")
        return
    if not tickets_path.exists():
        print(f"❌ {tickets_path} not found. Run 02_preprocess_data.py first.")
        return

    pairs = pd.read_csv(pairs_path)
    tickets = pd.read_csv(tickets_path)
    print(f"Loaded {len(pairs)} duplicate pairs, {len(tickets)} tickets")

    # Build ticket text lookup
    ticket_text = {}
    for _, row in tickets.iterrows():
        tid = str(row.get("ticket_id", row.get("unified_id", "")))
        text = str(row.get("embedding_text", row.get("title_clean", "")))
        ticket_text[tid] = text

    # Build evaluation pairs
    texts1, texts2, labels = [], [], []
    for _, row in pairs.iterrows():
        t1_id = str(row.get("ticket_1", row.get("ticket_id_1", "")))
        t2_id = str(row.get("ticket_2", row.get("ticket_id_2", "")))
        is_dup = int(row.get("is_duplicate", row.get("label", 0)))

        t1_text = ticket_text.get(t1_id, "")
        t2_text = ticket_text.get(t2_id, "")

        if t1_text and t2_text:
            texts1.append(t1_text)
            texts2.append(t2_text)
            labels.append(is_dup)

    if not texts1:
        print("❌ Could not match any pairs to ticket texts. Check column names.")
        return

    labels = np.array(labels)
    print(f"Matched {len(labels)} pairs ({labels.sum()} duplicates, {(1-labels).sum()} non-duplicates)")

    # Embed
    print(f"\nEmbedding with {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    emb1 = model.encode(texts1, show_progress_bar=True)
    emb2 = model.encode(texts2, show_progress_bar=True)
    similarities = util.cos_sim(emb1, emb2).diagonal().numpy()

    # Sweep thresholds
    thresholds = np.arange(0.50, 0.98, 0.02)
    results = []
    for tau in thresholds:
        preds = (similarities >= tau).astype(int)
        p = precision_score(labels, preds, zero_division=0)
        r = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        fp = int(((preds==1)&(labels==0)).sum())
        fn = int(((preds==0)&(labels==1)).sum())
        # Cost-sensitive: FN costs 2x FP
        cost = 1.0 * fp + 2.0 * fn
        results.append({"threshold":round(float(tau),3),"precision":round(p,4),"recall":round(r,4),
                        "f1":round(f1,4),"false_positives":fp,"false_negatives":fn,"cost":round(cost,1)})

    rdf = pd.DataFrame(results)
    rdf.to_csv(EVAL_DIR / "dedup_threshold_results.csv", index=False)

    # Best F1
    best_idx = rdf["f1"].idxmax(); best = rdf.iloc[best_idx]
    cost_idx = rdf["cost"].idxmin(); cost_best = rdf.iloc[cost_idx]

    print(f"\n  Best F1 threshold: τ = {best['threshold']}")
    print(f"    P={best['precision']:.4f}  R={best['recall']:.4f}  F1={best['f1']:.4f}")
    print(f"\n  Cost-optimal threshold (c_FN=2×c_FP): τ = {cost_best['threshold']}")
    print(f"    Cost={cost_best['cost']:.1f}")

    summary = {
        "total_pairs": len(labels),
        "duplicate_pairs": int(labels.sum()),
        "best_f1_threshold": float(best["threshold"]),
        "best_f1": float(best["f1"]),
        "best_precision": float(best["precision"]),
        "best_recall": float(best["recall"]),
        "cost_optimal_threshold": float(cost_best["threshold"]),
    }
    with open(EVAL_DIR / "dedup_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(rdf["threshold"], rdf["precision"], "b-", label="Precision", linewidth=1.5)
    ax1.plot(rdf["threshold"], rdf["recall"], "r-", label="Recall", linewidth=1.5)
    ax1.plot(rdf["threshold"], rdf["f1"], "g--", label="F1", linewidth=2.5)
    ax1.axvline(best["threshold"], color="gray", linestyle=":", label=f"Best τ={best['threshold']}")
    ax1.set_xlabel("Threshold (τ)"); ax1.set_ylabel("Score")
    ax1.set_title("P / R / F1 vs Threshold", fontweight="bold"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(rdf["threshold"], rdf["cost"], "m-", linewidth=2)
    ax2.axvline(cost_best["threshold"], color="gray", linestyle=":")
    ax2.set_xlabel("Threshold (τ)"); ax2.set_ylabel("Cost (c_FP=1, c_FN=2)")
    ax2.set_title("Cost-Sensitive Optimization", fontweight="bold"); ax2.grid(alpha=0.3)

    plt.suptitle("ITS RAG — Duplicate Detection Threshold Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "dedup_threshold_chart.png", dpi=150, bbox_inches="tight"); plt.close()

    print(f"\n✅ evaluation/dedup_threshold_results.csv\n✅ evaluation/dedup_summary.json\n✅ evaluation/dedup_threshold_chart.png")


if __name__ == "__main__":
    run_dedup_eval()
