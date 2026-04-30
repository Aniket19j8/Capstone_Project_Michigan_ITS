"""
ITS RAG — Advanced Metrics (root research stack)

Beyond raw means in 06–10, this script computes:

  • Bootstrap 95% confidence intervals for Recall@1/5/10 and MRR per method.
  • Paired statistical tests (Wilcoxon signed-rank + paired t-test) for
      Hybrid_Rerank vs Dense_Only, Hybrid_Rerank vs BM25_Only,
      Hybrid_Rerank vs Hybrid_NoRerank.
  • Effect size (Cohen's d) for the same comparisons.
  • Per-category and per-severity breakdowns (Recall@5, MRR per method).
  • Latency P50 / P95 / P99 per method.
  • Cost-quality Pareto for dedup (best-F1 vs cost-optimal).
  • Hallucination metrics:
      - hallucination rate (Base, RAG) on the IT subset
      - KB grounding rate proxy (RAG answers with kb_docs_passed_filter > 0)
      - refusal precision on out-of-scope queries
      - LLM-judge head-to-head win rate.

Outputs:

  evaluation/advanced_metrics.json
  evaluation/per_category_breakdown.csv
  evaluation/per_severity_breakdown.csv
  evaluation/significance_tests.json
  evaluation/cost_pareto.csv
  evaluation/advanced_metrics_chart.png

Usage:
  python 16_advanced_metrics.py
  python 16_advanced_metrics.py --bootstrap 5000
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats as scistats  # type: ignore
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

warnings.filterwarnings("ignore", category=FutureWarning)

EVAL_DIR = Path("evaluation")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["Dense_Only", "BM25_Only", "Hybrid_NoRerank", "Hybrid_Rerank"]
PALETTE = {
    "Dense_Only": "#7f8fa6",
    "BM25_Only": "#f1c40f",
    "Hybrid_NoRerank": "#3498db",
    "Hybrid_Rerank": "#27ae60",
}


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap CI
# ─────────────────────────────────────────────────────────────────────────────


def bootstrap_ci(values: np.ndarray, n_boot: int, alpha: float = 0.05, rng: np.random.Generator | None = None):
    rng = rng or np.random.default_rng(42)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.empty(n_boot)
    n = len(values)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = values[idx].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(values.mean()), lo, hi


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_sd = np.sqrt(((a.var(ddof=1) + b.var(ddof=1)) / 2) or 1e-12)
    return float((a.mean() - b.mean()) / pooled_sd)


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval analysis (500-query CSV)
# ─────────────────────────────────────────────────────────────────────────────


def retrieval_analysis(n_boot: int) -> dict:
    csv = EVAL_DIR / "500_query_results.csv"
    if not csv.exists():
        print(f"[skip retrieval] {csv} not found")
        return {}

    df = pd.read_csv(csv)
    metrics = ["recall@1", "recall@5", "recall@10", "mrr"]
    rng = np.random.default_rng(42)

    # Per-method bootstrap CIs.
    ci_summary: dict = {}
    for m in METHODS:
        ci_summary[m] = {}
        for met in metrics:
            col = f"{m}_{met}"
            if col not in df.columns:
                continue
            mean, lo, hi = bootstrap_ci(df[col].astype(float).values, n_boot, rng=rng)
            ci_summary[m][met] = {"mean": mean, "ci95_low": lo, "ci95_high": hi}
        for col in [f"{m}_latency_ms"]:
            if col not in df.columns:
                continue
            v = df[col].astype(float).values
            ci_summary[m]["latency_ms"] = {
                "mean": float(np.mean(v)),
                "p50": float(np.percentile(v, 50)),
                "p95": float(np.percentile(v, 95)),
                "p99": float(np.percentile(v, 99)),
            }

    # Pairwise significance vs Hybrid_Rerank for Recall@5.
    sig_tests: dict = {}
    if "Hybrid_Rerank_recall@5" in df.columns:
        baseline = df["Hybrid_Rerank_recall@5"].astype(float).values
        for m in METHODS:
            if m == "Hybrid_Rerank":
                continue
            other = df[f"{m}_recall@5"].astype(float).values
            entry = {
                "n": int(len(baseline)),
                "mean_diff_hybrid_minus_other": float(baseline.mean() - other.mean()),
                "cohens_d": cohens_d(baseline, other),
            }
            if HAS_SCIPY:
                t = scistats.ttest_rel(baseline, other, nan_policy="omit")
                w = scistats.wilcoxon(baseline, other, zero_method="zsplit", nan_policy="omit")
                entry["paired_t_p_value"] = float(t.pvalue)
                entry["paired_t_statistic"] = float(t.statistic)
                entry["wilcoxon_p_value"] = float(w.pvalue)
            sig_tests[f"Hybrid_Rerank vs {m}"] = entry

    # Per-category & per-severity Recall@5 for each method.
    per_cat = {}
    per_sev = {}
    if "category" in df.columns:
        grouped = df.groupby("category")
        per_cat = {
            cat: {m: float(g[f"{m}_recall@5"].mean()) for m in METHODS if f"{m}_recall@5" in g.columns}
            for cat, g in grouped
        }
    if "severity" in df.columns:
        grouped = df.groupby("severity")
        per_sev = {
            sev: {m: float(g[f"{m}_recall@5"].mean()) for m in METHODS if f"{m}_recall@5" in g.columns}
            for sev, g in grouped
        }

    pd.DataFrame(per_cat).T.to_csv(EVAL_DIR / "per_category_breakdown.csv")
    pd.DataFrame(per_sev).T.to_csv(EVAL_DIR / "per_severity_breakdown.csv")

    return {
        "n_queries": int(len(df)),
        "bootstrap_ci": ci_summary,
        "significance_tests_vs_hybrid_rerank": sig_tests,
        "per_category_recall5": per_cat,
        "per_severity_recall5": per_sev,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Hallucination analysis (08 CSV)
# ─────────────────────────────────────────────────────────────────────────────


def hallucination_analysis() -> dict:
    csv = EVAL_DIR / "hallucination_comparison.csv"
    if not csv.exists():
        print(f"[skip hallucination] {csv} not found")
        return {}

    df = pd.read_csv(csv)
    it = df[df["scope"] == "it"]
    oos = df[df["scope"] == "oos"]

    base_h_rate = float(it["judge_hallucination_in_a"].dropna().mean())
    rag_h_rate = float(it["judge_hallucination_in_b"].dropna().mean())

    # KB grounding proxy: RAG answers that had at least one KB doc pass the filter.
    rag_grounded = float((it["kb_docs_passed_filter"].fillna(0) > 0).mean())

    # Refusal on OOS: RAG should ideally refuse (judge marks it not hallucinating
    # AND we expect the answer to be short / explicitly say "I don't have that info").
    oos_refusal_proxy = float((oos["judge_hallucination_in_b"].fillna(False) == False).mean())  # noqa: E712

    winner = it["judge_which_is_better"].dropna()
    a_wins = int((winner == "A").sum())
    b_wins = int((winner == "B").sum())

    # Latency.
    base_lat = it["base_latency_s"].dropna().astype(float)
    rag_lat = it["rag_latency_s"].dropna().astype(float)

    return {
        "n_total": int(len(df)),
        "n_it": int(len(it)),
        "n_oos": int(len(oos)),
        "base_hallucination_rate_it": base_h_rate,
        "rag_hallucination_rate_it": rag_h_rate,
        "rag_kb_grounding_rate_it": rag_grounded,
        "rag_refusal_rate_oos": oos_refusal_proxy,
        "head_to_head": {"base_wins": a_wins, "rag_wins": b_wins},
        "latency_seconds": {
            "base_mean": float(base_lat.mean() if len(base_lat) else float("nan")),
            "rag_mean": float(rag_lat.mean() if len(rag_lat) else float("nan")),
            "base_p95": float(np.percentile(base_lat, 95) if len(base_lat) else float("nan")),
            "rag_p95": float(np.percentile(rag_lat, 95) if len(rag_lat) else float("nan")),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stability analysis
# ─────────────────────────────────────────────────────────────────────────────


def stability_analysis() -> dict:
    csv = EVAL_DIR / "perturbation_stability_n500.csv"
    if not csv.exists():
        print(f"[skip stability] {csv} not found")
        return {}
    df = pd.read_csv(csv)
    methods = sorted(df["method"].unique())
    out = {}
    for m in methods:
        g = df[df["method"] == m]
        out[m] = {
            "n_tickets": int(len(g)),
            "jaccard_top10_mean": float(g["jaccard_top10_across_perturbs"].mean()),
            "jaccard_top10_median": float(g["jaccard_top10_across_perturbs"].median()),
            "jaccard_top10_p25": float(g["jaccard_top10_across_perturbs"].quantile(0.25)),
            "jaccard_top10_p75": float(g["jaccard_top10_across_perturbs"].quantile(0.75)),
            "latency_ms_mean": float(g["mean_latency_ms"].mean()),
            "latency_ms_p95": float(g["mean_latency_ms"].quantile(0.95)),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Dedup cost-quality Pareto
# ─────────────────────────────────────────────────────────────────────────────


def dedup_pareto() -> dict:
    csv = EVAL_DIR / "dedup_threshold_results.csv"
    if not csv.exists():
        print(f"[skip dedup] {csv} not found")
        return {}
    df = pd.read_csv(csv).sort_values("threshold").reset_index(drop=True)

    pareto = df[["threshold", "f1", "cost", "precision", "recall"]].copy()
    pareto.to_csv(EVAL_DIR / "cost_pareto.csv", index=False)

    f1_idx = df["f1"].idxmax()
    cost_idx = df["cost"].idxmin()

    return {
        "best_f1": {
            "threshold": float(df.loc[f1_idx, "threshold"]),
            "f1": float(df.loc[f1_idx, "f1"]),
            "precision": float(df.loc[f1_idx, "precision"]),
            "recall": float(df.loc[f1_idx, "recall"]),
            "cost": float(df.loc[f1_idx, "cost"]),
        },
        "cost_optimal": {
            "threshold": float(df.loc[cost_idx, "threshold"]),
            "cost": float(df.loc[cost_idx, "cost"]),
            "precision": float(df.loc[cost_idx, "precision"]),
            "recall": float(df.loc[cost_idx, "recall"]),
            "f1": float(df.loc[cost_idx, "f1"]),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Composite chart for the deck
# ─────────────────────────────────────────────────────────────────────────────


def composite_chart(retrieval: dict, hallucination: dict, stability: dict, dedup: dict, dpi: int) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Recall@5 per method with bootstrap CI error bars.
    ax = axes[0, 0]
    if retrieval:
        names, means, errs = [], [], []
        for m in METHODS:
            row = retrieval.get("bootstrap_ci", {}).get(m, {}).get("recall@5", {})
            if not row:
                continue
            names.append(m)
            means.append(row["mean"])
            errs.append([[row["mean"] - row["ci95_low"]], [row["ci95_high"] - row["mean"]]])
        if names:
            x = np.arange(len(names))
            for i, name in enumerate(names):
                ax.bar(x[i], means[i], color=PALETTE[name])
                ax.errorbar(x[i], means[i], yerr=errs[i], color="black", capsize=4, fmt="none")
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=15)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Recall@5 ± 95% CI (bootstrap)")
            ax.set_title("Retrieval — Recall@5 with 95% bootstrap CI")

    # 2. Hallucination Base vs RAG.
    ax = axes[0, 1]
    if hallucination:
        b = hallucination.get("base_hallucination_rate_it", 0) * 100
        r = hallucination.get("rag_hallucination_rate_it", 0) * 100
        ax.bar(["Base", "RAG"], [b, r], color=["#e74c3c", "#27ae60"])
        for i, v in enumerate([b, r]):
            ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom")
        ax.set_ylabel("hallucination rate (%)")
        ax.set_ylim(0, max(b, 5) * 1.2)
        ax.set_title(f"Hallucination — IT subset (n={hallucination.get('n_it', 0)})")

    # 3. Stability — Jaccard top-10 boxplot.
    ax = axes[1, 0]
    if stability:
        labels = list(stability.keys())
        means = [stability[m]["jaccard_top10_mean"] for m in labels]
        colors = [PALETTE.get(m, "#888") for m in labels]
        ax.bar(labels, means, color=colors)
        for i, v in enumerate(means):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("mean Jaccard top-10 across perturbations")
        ax.set_title("Stability under typos / noise / case / code-injection")
        ax.tick_params(axis="x", rotation=15)

    # 4. Dedup cost-quality.
    ax = axes[1, 1]
    if dedup:
        bf = dedup["best_f1"]
        co = dedup["cost_optimal"]
        ax.bar(["Best-F1 τ", "Cost-optimal τ"], [bf["f1"], co["f1"]], color=["#27ae60", "#c0392b"])
        for i, (lbl, v, t) in enumerate([("F1", bf["f1"], bf["threshold"]), ("F1", co["f1"], co["threshold"])]):
            ax.text(i, v, f"{lbl}={v:.2f}\nτ={t:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("F1")
        ax.set_title("Dedup — quality vs cost decision points")

    fig.suptitle("Advanced Metrics Snapshot — Root Research Stack", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = EVAL_DIR / "advanced_metrics_chart.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=2000, help="bootstrap iterations")
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    print(f"[16] scipy={'yes' if HAS_SCIPY else 'no - paired tests skipped'} bootstrap={args.bootstrap}")

    retrieval = retrieval_analysis(args.bootstrap)
    halluc = hallucination_analysis()
    stab = stability_analysis()
    dedup = dedup_pareto()

    out_payload = {
        "_about": "Advanced metrics derived from existing CSV artifacts in evaluation/.",
        "retrieval": retrieval,
        "hallucination": halluc,
        "stability": stab,
        "dedup": dedup,
    }
    (EVAL_DIR / "advanced_metrics.json").write_text(json.dumps(out_payload, indent=2))
    print(f"  [ok] wrote {EVAL_DIR / 'advanced_metrics.json'}")

    sig_only = retrieval.get("significance_tests_vs_hybrid_rerank", {}) if retrieval else {}
    (EVAL_DIR / "significance_tests.json").write_text(json.dumps(sig_only, indent=2))
    print(f"  [ok] wrote {EVAL_DIR / 'significance_tests.json'}")

    out = composite_chart(retrieval, halluc, stab, dedup, args.dpi)
    print(f"  [ok] wrote {out}")
    print("[16] Done.")


if __name__ == "__main__":
    main()
