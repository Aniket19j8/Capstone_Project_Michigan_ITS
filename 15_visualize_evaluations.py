"""
ITS RAG — Evaluation Visualizer (root research stack)

Reads existing CSV / JSON artifacts under `evaluation/` and `data/processed/`,
and writes the full set of presentation-ready PNG charts:

  evaluation/retrieval_comparison_chart.png     (from 500_query_results.csv)
  evaluation/hallucination_chart.png            (from hallucination_comparison.csv)
  evaluation/dedup_threshold_chart.png          (from dedup_threshold_results.csv)
  evaluation/perturbation_stability_chart.png   (from perturbation_stability_n500.csv)
  evaluation/ablation_top1_chart.png            (from ablation_results.csv)
  evaluation/embedding_ablation_chart.png       (from embedded Table 1 numbers)
  evaluation/dataset_distributions_chart.png    (from data/processed/data_stats.json)
  evaluation/kpi_dashboard_chart.png            (Status 1 / Status 2 / Targets)

Usage:
  python 15_visualize_evaluations.py
  python 15_visualize_evaluations.py --only retrieval hallucination dedup
  python 15_visualize_evaluations.py --dpi 150
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# Consistent palette and style. We avoid seaborn so the dep set stays small.
plt.rcParams.update({
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 110,
    "savefig.bbox": "tight",
})

PALETTE = {
    "Dense_Only": "#7f8fa6",
    "BM25_Only": "#f1c40f",
    "Hybrid_NoRerank": "#3498db",
    "Hybrid_Rerank": "#27ae60",
    "Base": "#e74c3c",
    "RAG": "#27ae60",
    "Status1": "#7f8fa6",
    "Status2": "#27ae60",
    "Target": "#34495e",
}

EVAL_DIR = Path("evaluation")
DATA_DIR = Path("data/processed")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _save(fig: plt.Figure, name: str, dpi: int) -> Path:
    out = EVAL_DIR / name
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def _bar_with_values(ax, x_labels, values, colors, fmt="{:.3f}"):
    bars = ax.bar(x_labels, values, color=colors)
    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v,
            fmt.format(v),
            ha="center",
            va="bottom",
            fontsize=9,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Retrieval comparison (500-query CSV)
# ─────────────────────────────────────────────────────────────────────────────


def chart_retrieval(dpi: int) -> Path | None:
    csv = EVAL_DIR / "500_query_results.csv"
    if not csv.exists():
        print(f"[skip] {csv} not found")
        return None

    df = pd.read_csv(csv)
    methods = ["Dense_Only", "BM25_Only", "Hybrid_NoRerank", "Hybrid_Rerank"]
    metrics = ["recall@1", "recall@5", "recall@10", "mrr"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, metric in zip(axes[:3], metrics[:3]):
        vals = [df[f"{m}_{metric}"].mean() for m in methods]
        _bar_with_values(ax, methods, vals, [PALETTE[m] for m in methods])
        ax.set_title(metric.upper())
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("score (mean over 500 queries)")
        ax.tick_params(axis="x", rotation=15)

    ax = axes[3]
    lat = [df[f"{m}_latency_ms"].mean() for m in methods]
    _bar_with_values(ax, methods, lat, [PALETTE[m] for m in methods], fmt="{:.0f} ms")
    ax.set_title("MEAN PER-QUERY LATENCY (ms)")
    ax.set_ylabel("milliseconds")
    ax.tick_params(axis="x", rotation=15)

    fig.suptitle(
        "Retrieval Comparison — 500-query self-retrieval (root research stack)\n"
        "Dense_Only and Hybrid_NoRerank reflect a diagnosed embedding/ID-space"
        " mismatch in this run; production uses Hybrid_Rerank.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, "retrieval_comparison_chart.png", dpi)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Hallucination comparison (Base vs RAG, LLM-as-judge)
# ─────────────────────────────────────────────────────────────────────────────


def chart_hallucination(dpi: int) -> Path | None:
    csv = EVAL_DIR / "hallucination_comparison.csv"
    if not csv.exists():
        print(f"[skip] {csv} not found")
        return None

    df = pd.read_csv(csv)
    it = df[df["scope"] == "it"]
    oos = df[df["scope"] == "oos"]

    base_h = it["judge_hallucination_in_a"].dropna().mean()
    rag_h = it["judge_hallucination_in_b"].dropna().mean()

    score_cols_a = [
        "judge_answer_a_groundedness",
        "judge_answer_a_specificity",
        "judge_answer_a_accuracy",
    ]
    score_cols_b = [
        "judge_answer_b_groundedness",
        "judge_answer_b_specificity",
        "judge_answer_b_accuracy",
    ]
    base_scores = [it[c].dropna().mean() for c in score_cols_a]
    rag_scores = [it[c].dropna().mean() for c in score_cols_b]

    winner = it["judge_which_is_better"].dropna()
    a_wins = (winner == "A").sum()
    b_wins = (winner == "B").sum()
    parsed = a_wins + b_wins

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    ax = axes[0]
    _bar_with_values(
        ax,
        ["Base LLM", "RAG"],
        [base_h * 100, rag_h * 100],
        [PALETTE["Base"], PALETTE["RAG"]],
        fmt="{:.1f}%",
    )
    ax.set_title(f"Hallucination rate — IT subset (n={len(it)})")
    ax.set_ylabel("hallucinated answers (%)")
    ax.set_ylim(0, max(70, base_h * 110))

    ax = axes[1]
    x = np.arange(len(score_cols_a))
    w = 0.35
    ax.bar(x - w / 2, base_scores, w, label="Base", color=PALETTE["Base"])
    ax.bar(x + w / 2, rag_scores, w, label="RAG", color=PALETTE["RAG"])
    ax.set_xticks(x)
    ax.set_xticklabels(["Groundedness", "Specificity", "Accuracy"])
    ax.set_ylim(0, 5.2)
    ax.set_ylabel("LLM-judge score (1–5)")
    ax.set_title("Quality scores — IT subset")
    ax.legend()

    ax = axes[2]
    _bar_with_values(
        ax,
        ["Base preferred", "RAG preferred"],
        [a_wins, b_wins],
        [PALETTE["Base"], PALETTE["RAG"]],
        fmt="{:.0f}",
    )
    ax.set_title(f"Head-to-head — RAG wins {b_wins}/{parsed} parsed")
    ax.set_ylabel("count")

    fig.suptitle(
        "RAG vs Base LLM (50 queries: 45 IT + 5 OOS) — LLM-as-judge",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    summary = {
        "n_total": int(len(df)),
        "n_it": int(len(it)),
        "n_oos": int(len(oos)),
        "base_hallucination_rate_it": float(base_h),
        "rag_hallucination_rate_it": float(rag_h),
        "base_quality_means": dict(zip(score_cols_a, [float(x) for x in base_scores])),
        "rag_quality_means": dict(zip(score_cols_b, [float(x) for x in rag_scores])),
        "head_to_head_a_wins": int(a_wins),
        "head_to_head_b_wins": int(b_wins),
    }
    (EVAL_DIR / "hallucination_summary.json").write_text(json.dumps(summary, indent=2))
    return _save(fig, "hallucination_chart.png", dpi)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dedup threshold sweep
# ─────────────────────────────────────────────────────────────────────────────


def chart_dedup(dpi: int) -> Path | None:
    csv = EVAL_DIR / "dedup_threshold_results.csv"
    if not csv.exists():
        print(f"[skip] {csv} not found")
        return None

    df = pd.read_csv(csv)
    df = df.sort_values("threshold").reset_index(drop=True)
    best_f1_idx = df["f1"].idxmax()
    best_cost_idx = df["cost"].idxmin()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(df["threshold"], df["precision"], label="Precision", color="#2980b9", marker="o", ms=4)
    ax.plot(df["threshold"], df["recall"], label="Recall", color="#e67e22", marker="o", ms=4)
    ax.plot(df["threshold"], df["f1"], label="F1", color="#27ae60", marker="o", ms=4, linewidth=2)
    ax.axvline(df.loc[best_f1_idx, "threshold"], color="#27ae60", linestyle=":", alpha=0.5)
    ax.set_xlabel("similarity threshold τ")
    ax.set_ylabel("score")
    ax.set_title(
        f"P / R / F1 curve — best F1={df.loc[best_f1_idx, 'f1']:.3f} at τ={df.loc[best_f1_idx, 'threshold']:.2f}"
    )
    ax.legend()
    ax.set_ylim(0, 1.05)

    ax = axes[1]
    ax.plot(df["threshold"], df["cost"], color="#c0392b", marker="o", ms=4)
    ax.axvline(df.loc[best_cost_idx, "threshold"], color="#c0392b", linestyle=":", alpha=0.5)
    ax.set_xlabel("similarity threshold τ")
    ax.set_ylabel("cost (FN counts 2× FP)")
    ax.set_title(
        f"Cost curve — min cost {df.loc[best_cost_idx, 'cost']:.0f} at τ={df.loc[best_cost_idx, 'threshold']:.2f}"
    )

    fig.suptitle(
        "Dedup threshold sweep — operational decision framework",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    summary = {
        "best_f1_threshold": float(df.loc[best_f1_idx, "threshold"]),
        "best_f1": float(df.loc[best_f1_idx, "f1"]),
        "cost_optimal_threshold": float(df.loc[best_cost_idx, "threshold"]),
        "min_cost": float(df.loc[best_cost_idx, "cost"]),
    }
    (EVAL_DIR / "dedup_summary.json").write_text(json.dumps(summary, indent=2))
    return _save(fig, "dedup_threshold_chart.png", dpi)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Perturbation stability (Jaccard top-10 across perturbations)
# ─────────────────────────────────────────────────────────────────────────────


def chart_stability(dpi: int) -> Path | None:
    csv = EVAL_DIR / "perturbation_stability_n500.csv"
    if not csv.exists():
        print(f"[skip] {csv} not found")
        return None

    df = pd.read_csv(csv)
    methods = ["BM25_Only", "Dense_Only", "Hybrid_NoRerank"]
    colors = [PALETTE[m] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    data = [df[df["method"] == m]["jaccard_top10_across_perturbs"].dropna() for m in methods]
    # tick_labels= is the matplotlib >= 3.9 spelling; older versions used labels=.
    try:
        parts = ax.boxplot(data, tick_labels=methods, patch_artist=True, showmeans=True, widths=0.55)
    except TypeError:
        parts = ax.boxplot(data, labels=methods, patch_artist=True, showmeans=True, widths=0.55)
    for patch, c in zip(parts["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel("Jaccard top-10 across 5 perturbations")
    ax.set_title("Stability under typos / noise / case / code-injection")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=15)

    ax = axes[1]
    data = [df[df["method"] == m]["mean_latency_ms"].dropna() for m in methods]
    try:
        parts = ax.boxplot(data, tick_labels=methods, patch_artist=True, showmeans=True, widths=0.55)
    except TypeError:
        parts = ax.boxplot(data, labels=methods, patch_artist=True, showmeans=True, widths=0.55)
    for patch, c in zip(parts["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel("mean latency per query (ms, log)")
    ax.set_title("Per-method latency under perturbations")
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=15)

    fig.suptitle(
        "Perturbation stability — n=500 tickets × 5 query variants per ticket",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    summary = {
        m: {
            "jaccard_top10_mean": float(df[df["method"] == m]["jaccard_top10_across_perturbs"].mean()),
            "jaccard_top10_median": float(df[df["method"] == m]["jaccard_top10_across_perturbs"].median()),
            "latency_ms_mean": float(df[df["method"] == m]["mean_latency_ms"].mean()),
            "latency_ms_p95": float(df[df["method"] == m]["mean_latency_ms"].quantile(0.95)),
        }
        for m in methods
    }
    (EVAL_DIR / "perturbation_stability_summary.json").write_text(json.dumps(summary, indent=2))
    return _save(fig, "perturbation_stability_chart.png", dpi)


# ─────────────────────────────────────────────────────────────────────────────
# 5. 10-query manual ablation top-1
# ─────────────────────────────────────────────────────────────────────────────


def chart_ablation_top1(dpi: int) -> Path | None:
    csv = EVAL_DIR / "ablation_results.csv"
    if not csv.exists():
        print(f"[skip] {csv} not found")
        return None

    df = pd.read_csv(csv)
    score_cols = [
        ("Dense Only_score", "Dense_Only"),
        ("BM25 Only_score", "BM25_Only"),
        ("Hybrid (no rerank)_score", "Hybrid_NoRerank"),
        ("Hybrid + Rerank_score", "Hybrid_Rerank"),
    ]

    fig, ax = plt.subplots(figsize=(11, 4.2))
    queries = df["query"].str.slice(0, 32).tolist()
    n_q = len(queries)
    width = 0.2
    x = np.arange(n_q)

    for i, (col, label) in enumerate(score_cols):
        if col not in df.columns:
            continue
        vals = df[col].astype(float).fillna(0).values
        # Normalize so semi-comparable across very different scoring scales (cosine, BM25, RRF, rerank).
        vmax = max(abs(vals.max()), abs(vals.min()), 1e-6)
        norm = vals / vmax
        ax.bar(x + (i - 1.5) * width, norm, width, label=label, color=PALETTE[label])

    ax.set_xticks(x)
    ax.set_xticklabels(queries, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("top-1 score (per-method min-max normalized)")
    ax.set_title("10-query manual ablation — top-1 score per method per query")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.30))
    fig.tight_layout()
    return _save(fig, "ablation_top1_chart.png", dpi)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Embedding ablation Table 1
# ─────────────────────────────────────────────────────────────────────────────


def chart_embedding_ablation(dpi: int) -> Path:
    table1 = pd.DataFrame(
        [
            ("Qwen3-0.6B", 1024, 0.887, 0.849, 0.933, 0.953, 0.966, 0.987),
            ("Gemma-300M", 768, 0.800, 0.744, 0.873, 0.904, 0.930, 0.971),
            ("Nomic-v2 MoE", 768, 0.795, 0.735, 0.868, 0.903, 0.927, 0.969),
            ("BGE-Large v1.5", 1024, 0.735, 0.670, 0.811, 0.893, 0.893, 0.956),
            ("MiniLM-L6 v2", 384, 0.697, 0.625, 0.782, 0.879, 0.879, 0.949),
        ],
        columns=["model", "dim", "MRR", "R@1", "R@5", "R@10", "R@20", "R@100"],
    )
    table1.to_csv(EVAL_DIR / "embedding_ablation_table1.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    colors = ["#27ae60", "#3498db", "#9b59b6", "#e67e22", "#7f8fa6"]

    ax = axes[0]
    metrics = ["MRR", "R@1", "R@5", "R@10", "R@20", "R@100"]
    x = np.arange(len(metrics))
    width = 0.16
    for i, row in table1.iterrows():
        ax.bar(
            x + (i - 2) * width,
            [row[m] for m in metrics],
            width,
            label=f"{row['model']} ({row['dim']}-d)",
            color=colors[i],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.6, 1.02)
    ax.set_ylabel("score")
    ax.set_title("Embedding model comparison — Table 1")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    _bar_with_values(
        ax,
        table1["model"].tolist(),
        table1["MRR"].tolist(),
        colors,
        fmt="{:.3f}",
    )
    ax.set_title("MRR — single-axis ranking")
    ax.set_ylim(0.6, 1.0)
    ax.tick_params(axis="x", rotation=20)

    fig.suptitle(
        "Embedding ablation (3,045 queries / 35,597-chunk FAISS index) — Qwen3-0.6B is the production embedder",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "embedding_ablation_chart.png", dpi)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Dataset distributions (final 80K corpus)
# ─────────────────────────────────────────────────────────────────────────────


def chart_dataset_distributions(dpi: int) -> Path | None:
    stats_path = DATA_DIR / "data_stats.json"
    if not stats_path.exists():
        print(f"[skip] {stats_path} not found")
        return None
    stats = json.loads(stats_path.read_text())

    def _topn(d: dict, n: int = 10):
        items = sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:n]
        return [k for k, _ in items], [v for _, v in items]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    panels = [
        ("categories", "Categories"),
        ("departments", "Departments"),
        ("source_systems", "Source systems"),
        ("languages", "Languages"),
    ]
    palette = ["#27ae60", "#3498db", "#9b59b6", "#e67e22"]
    for ax, (key, title), color in zip(axes.flatten(), panels, palette):
        labels, values = _topn(stats.get(key, {}))
        ax.barh(labels[::-1], values[::-1], color=color)
        ax.set_title(f"{title} (top {len(labels)})")
        ax.set_xlabel("ticket count")
        for i, v in enumerate(values[::-1]):
            ax.text(v, i, f" {v:,}", va="center", fontsize=8)

    fig.suptitle(
        f"Final 80K corpus — distributions (avg quality {stats.get('avg_quality_score', 0):.3f},"
        f" {stats.get('tickets_with_resolution', 0):,} with resolutions)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, "dataset_distributions_chart.png", dpi)


# ─────────────────────────────────────────────────────────────────────────────
# 8. KPI dashboard (Status 1 vs Status 2 vs Targets)
# ─────────────────────────────────────────────────────────────────────────────


def chart_kpi_dashboard(dpi: int) -> Path:
    rows = [
        ("MRR", 0.887, 0.988, 0.85),
        ("Recall@1", 0.849, 0.92, 0.80),
        ("Recall@5", 0.933, 0.996, 0.85),
        ("RAG hallucination (1-rate)", 0.90, 0.955, 0.95),
        ("Dedup F1", 0.78, 0.93, 0.85),
    ]
    labels = [r[0] for r in rows]
    s1 = [r[1] for r in rows]
    s2 = [r[2] for r in rows]
    tg = [r[3] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    x = np.arange(len(labels))
    w = 0.27
    ax.bar(x - w, s1, w, label="Status 1", color=PALETTE["Status1"])
    ax.bar(x, s2, w, label="Status 2", color=PALETTE["Status2"])
    ax.bar(x + w, tg, w, label="Target", color=PALETTE["Target"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("score")
    ax.set_title("KPI progression — quality")
    ax.legend()

    ax = axes[1]
    lat_labels = ["Status 1\nlocal Ollama Qwen3-4B", "Status 2\nQwen3-0.6B emb + vLLM", "Final\ncloud (OpenAI+Pinecone)"]
    lat_values = [6500, 596, 350]
    _bar_with_values(
        ax,
        lat_labels,
        lat_values,
        [PALETTE["Status1"], PALETTE["Status2"], "#27ae60"],
        fmt="{:.0f} ms",
    )
    ax.set_title("Per-query latency progression (mean)")
    ax.set_ylabel("ms (lower is better)")
    ax.set_yscale("log")

    fig.suptitle(
        "KPI dashboard — Status 1 → Status 2 → Final",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "kpi_dashboard_chart.png", dpi)


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────


CHARTS = {
    "retrieval": chart_retrieval,
    "hallucination": chart_hallucination,
    "dedup": chart_dedup,
    "stability": chart_stability,
    "ablation": chart_ablation_top1,
    "embedding": chart_embedding_ablation,
    "dataset": chart_dataset_distributions,
    "kpi": chart_kpi_dashboard,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="*",
        choices=list(CHARTS.keys()),
        help="Generate only the named charts (default: all)",
    )
    parser.add_argument("--dpi", type=int, default=120, help="PNG DPI")
    args = parser.parse_args()

    selected = args.only or list(CHARTS.keys())
    print(f"[15] Generating {len(selected)} chart(s) -> {EVAL_DIR.resolve()}")
    for key in selected:
        try:
            out = CHARTS[key](args.dpi)
            if out:
                print(f"  [ok]   {key:14s} -> {out}")
            else:
                print(f"  [skip] {key:14s} (input missing)")
        except Exception as exc:  # noqa: BLE001
            print(f"  [fail] {key:14s} {exc}")
    print("[15] Done.")


if __name__ == "__main__":
    main()
