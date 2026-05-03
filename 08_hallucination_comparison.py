"""
08 — RAG vs "naked" LLM on the same prompts; judge model scores hallucination-ish answers.

We only stuff context into RAG when rerank scores look sane, otherwise the model
lies confidently. Separate IT vs OOS buckets so aggregates aren't nonsense.
Requires Ollama (qwen3:8b or qwen3:4b). Outputs: evaluation/hallucination_*.{csv,json,png}

Usage:
  python 08_hallucination_comparison.py
  python 08_hallucination_comparison.py --model qwen3:4b --num-queries 20
"""

import json
import time
import re
import argparse
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──
EVAL_DIR = Path("./evaluation")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ── Load retriever ──
_rp = Path(__file__).parent / "04_hybrid_retrieval.py"
_sp = importlib.util.spec_from_file_location("hybrid_retrieval_engine", _rp)
_mod = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(_mod)
HybridRetriever = _mod.HybridRetriever

# ── Default config ──
DEFAULT_MODEL = "qwen3:8b"
OLLAMA_URL = "http://localhost:11434"

# Cross-encoder ms-marco-MiniLM outputs raw logits.
# Scores below this threshold indicate the document is off-topic for the query.
# Positive scores = relevant; < -3 = clearly irrelevant (e.g. VPN runbook for pasta query).
MIN_RERANK_SCORE = -3.0


def llm_generate(prompt: str, system: str = "", model: str = DEFAULT_MODEL,
                 temperature: float = 0.1, max_tokens: int = 1024) -> str:
    """Call Ollama /api/chat endpoint."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system or "You are a helpful IT support assistant."},
                    {"role": "user", "content": prompt},
                ],
                "options": {"temperature": temperature},
            },
            timeout=180,
        )
        data = resp.json()
        if "message" in data and isinstance(data["message"], dict):
            return data["message"].get("content", "").strip()
        return data.get("response", "").strip()
    except Exception as e:
        return f"[LLM_ERROR] {e}"


def strip_thinking_tokens(text: str) -> str:
    """Remove <think>...</think> blocks that qwen3 and similar models emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_judge_json(judge_raw: str) -> dict:
    """
    Parse the JSON score block from the judge response.
    Handles qwen3 thinking tokens and nested braces.
    """
    cleaned = strip_thinking_tokens(judge_raw)
    # Try to find the outermost JSON object
    try:
        # Greedy match that handles nested braces one level deep
        json_match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass
    # Fallback: try the raw text in case thinking strip changed nothing useful
    try:
        json_match = re.search(r'\{[^{}]*\}', judge_raw, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass
    return {"parse_error": True}


def format_context(ticket_results: list, kb_results: list) -> str:
    """
    Format retrieved results into a context string for the RAG prompt.

    Applies a relevance threshold using the cross-encoder rerank_score:
      - rerank_score is the raw logit from ms-marco-MiniLM-L-6-v2.
      - Scores below MIN_RERANK_SCORE mean the document is off-topic.
      - Filtering prevents irrelevant IT documents from being fed to the LLM
        when the query is out-of-scope (e.g. cooking, weather), which would
        cause the LLM to generate a confused, context-anchored hallucination.

    If all retrieved documents are below the threshold the function returns
    "No relevant context found." and the LLM is instructed to say so.
    """
    parts = []

    for doc in kb_results:
        score = doc.get("rerank_score", None)
        # Only apply threshold when cross-encoder was used (rerank_score present)
        if score is not None and float(score) < MIN_RERANK_SCORE:
            continue
        meta = doc.get("metadata", {})
        # Increased from 500 → 800 chars so runbook steps aren't cut mid-procedure
        text = doc.get("text", "")[:800]
        parts.append(
            f"[KB Source: {meta.get('source_file', meta.get('doc_title', 'KB'))}]\n{text}"
        )

    for doc in ticket_results:
        score = doc.get("rerank_score", None)
        if score is not None and float(score) < MIN_RERANK_SCORE:
            continue
        meta = doc.get("metadata", {})
        # Increased from 300 → 500 chars
        text = doc.get("text", "")[:500]
        tid = meta.get("ticket_id", doc["id"])
        parts.append(f"[Ticket: {tid} | {meta.get('title', 'N/A')}]\n{text}")

    return "\n\n".join(parts) if parts else "No relevant context found."


# ── Test queries ──
# Each entry: {"query": "...", "scope": "it" | "oos"}
#   "it"  = in-scope IT support question (RAG should help)
#   "oos" = out-of-scope question (RAG should correctly refuse)
ALL_TEST_QUERIES = [
    # ── VPN (KB has runbook) ──
    {"query": "How do I fix VPN connection drops?", "scope": "it"},
    {"query": "My VPN keeps disconnecting every 10 minutes on Windows 11", "scope": "it"},
    {"query": "VPN client version 4.1 is not connecting", "scope": "it"},
    {"query": "What MTU setting should I use for VPN?", "scope": "it"},
    {"query": "How to collect VPN logs for escalation?", "scope": "it"},
    # ── Password / Account (KB has runbook) ──
    {"query": "My account is locked out, how do I unlock it?", "scope": "it"},
    {"query": "How to reset MFA enrollment?", "scope": "it"},
    {"query": "What causes account lockout in Active Directory?", "scope": "it"},
    {"query": "How to verify user identity for password reset?", "scope": "it"},
    {"query": "How do I enable self-service password reset?", "scope": "it"},
    # ── Email (KB has runbook) ──
    {"query": "Outlook is not syncing my emails", "scope": "it"},
    {"query": "How to clear Outlook cache?", "scope": "it"},
    {"query": "Shared mailbox not showing in Outlook", "scope": "it"},
    {"query": "What is the mailbox size quota?", "scope": "it"},
    {"query": "How to check Microsoft 365 service health?", "scope": "it"},
    # ── Hardware (KB has runbook) ──
    {"query": "Laptop showing blue screen IRQL_NOT_LESS_OR_EQUAL", "scope": "it"},
    {"query": "My laptop won't turn on after hard reset", "scope": "it"},
    {"query": "How to fix overheating laptop?", "scope": "it"},
    {"query": "What does KERNEL_DATA_INPAGE_ERROR mean?", "scope": "it"},
    {"query": "How to run Windows Memory Diagnostic?", "scope": "it"},
    # ── Software (KB has runbook) ──
    {"query": "Software installation failed with permission error", "scope": "it"},
    {"query": "Product activation failed for Office 365", "scope": "it"},
    {"query": "How to get Adobe Creative Cloud license?", "scope": "it"},
    {"query": "Error: disk space insufficient for installation", "scope": "it"},
    {"query": "How to fix license limit reached error?", "scope": "it"},
    # ── Teams (KB has article) ──
    {"query": "Microsoft Teams is running very slow and using too much memory", "scope": "it"},
    {"query": "How to clear Teams cache?", "scope": "it"},
    {"query": "Should I upgrade to New Teams?", "scope": "it"},
    {"query": "Teams GPU hardware acceleration setting", "scope": "it"},
    {"query": "How to reduce Teams memory usage?", "scope": "it"},
    # ── Printer (KB has article) ──
    {"query": "Print jobs stuck in queue, nothing printing", "scope": "it"},
    {"query": "How to set up a network printer?", "scope": "it"},
    {"query": "Printer showing offline status", "scope": "it"},
    {"query": "How does secure print work?", "scope": "it"},
    {"query": "Poor print quality from network printer", "scope": "it"},
    # ── General IT (no specific KB) ──
    {"query": "How do I set up dual monitors?", "scope": "it"},
    {"query": "My mouse cursor is jumping around", "scope": "it"},
    {"query": "WiFi keeps disconnecting in building 3", "scope": "it"},
    {"query": "How to request a new laptop?", "scope": "it"},
    {"query": "Can I install personal software on work computer?", "scope": "it"},
    # ── IT Policy / SLA ──
    {"query": "What is the SLA for critical tickets?", "scope": "it"},
    {"query": "How to escalate a ticket to Tier 2?", "scope": "it"},
    {"query": "What are the working hours for help desk?", "scope": "it"},
    {"query": "How to submit a facilities request?", "scope": "it"},
    {"query": "What antivirus software do we use?", "scope": "it"},
    # ── Out-of-scope (RAG should refuse; base LLM may answer) ──
    {"query": "How to make pasta carbonara?", "scope": "oos"},
    {"query": "What's the weather today?", "scope": "oos"},
    {"query": "Tell me about quantum computing", "scope": "oos"},
    {"query": "How to file taxes in Arizona?", "scope": "oos"},
    {"query": "What's the capital of France?", "scope": "oos"},
]


def build_judge_prompt(query: str, scope: str, base_answer: str, rag_answer: str) -> str:
    """
    Build a scope-aware judge prompt that correctly defines hallucination
    in the RAG context and handles out-of-scope queries fairly.

    Key design decisions:
    - Explicitly tells the judge that for an IT support RAG system, refusing
      to answer out-of-scope queries is CORRECT, not hallucination.
    - Defines hallucination precisely: fabricating specific steps, URLs,
      numbers, policies, or error codes not present in the retrieved context.
    - For OOS queries, the judge is told that Answer B saying it has no
      information should be scored as high-groundedness, correct behavior.
    """
    if scope == "oos":
        scope_note = (
            "NOTE: This question is NOT an IT support question. "
            "Answer B comes from an IT support RAG system. "
            "For an IT support system, refusing to answer non-IT questions "
            "is the CORRECT behavior (not hallucination). "
            "If Answer B says 'I don't have this information in our knowledge base' "
            "or similar, that is ideal behavior — mark hallucination_in_b as false "
            "and give it high groundedness. "
            "Score Answer A on whether it answers the question correctly."
        )
    else:
        scope_note = (
            "This is an IT support question. "
            "Answer B comes from a RAG system with retrieved IT knowledge base context. "
            "HALLUCINATION definition: fabricating specific steps, error codes, URLs, "
            "policy numbers, or procedures that are NOT in the retrieved context. "
            "Saying 'I don't have enough information' when context is missing is NOT hallucination. "
            "A grounded answer that cites sources and stays within the context is ideal."
        )

    return (
        f"You are evaluating an IT support AI system. Score these two answers.\n\n"
        f"{scope_note}\n\n"
        f"Question: {query}\n\n"
        f"Answer A (Base LLM, no retrieved context): {base_answer[:600]}\n\n"
        f"Answer B (RAG system, with retrieved KB context): {rag_answer[:600]}\n\n"
        f"Score each answer 1-5 on:\n"
        f"1. Groundedness: Does it stay within what it actually knows/retrieved? "
        f"   (5=fully grounded, 1=makes up specific facts)\n"
        f"2. Specificity: Does it give org-specific, actionable steps? "
        f"   (5=very specific, 1=vague generic advice)\n"
        f"3. Accuracy: Is the answer correct and useful for IT support? "
        f"   (5=fully correct, 1=wrong or harmful)\n\n"
        f"Respond in JSON only, no markdown fences:\n"
        f'{{"answer_a_groundedness":N,"answer_a_specificity":N,"answer_a_accuracy":N,'
        f'"answer_b_groundedness":N,"answer_b_specificity":N,"answer_b_accuracy":N,'
        f'"which_is_better":"A or B or TIE",'
        f'"hallucination_in_a":true_or_false,"hallucination_in_b":true_or_false,'
        f'"reasoning":"one sentence"}}'
    )


def run_comparison(model: str = DEFAULT_MODEL, num_queries: int = 50):
    print("=" * 60)
    print("ITS RAG — Hallucination Comparison: RAG vs Base LLM")
    print(f"Model: {model} | Queries: {num_queries}")
    print(f"Context relevance threshold: rerank_score >= {MIN_RERANK_SCORE}")
    print("=" * 60)

    test_queries = ALL_TEST_QUERIES[:num_queries]

    # Load retrievers
    print("\nLoading retrievers...")
    ticket_retriever = HybridRetriever("its_tickets", load_reranker=True)
    kb_retriever = HybridRetriever("its_knowledge_base", load_reranker=True)

    results = []
    print(f"\nRunning {len(test_queries)} queries through both systems...\n")

    for i, item in enumerate(test_queries):
        query = item["query"]
        scope = item["scope"]
        scope_label = "[IT]" if scope == "it" else "[OOS]"
        print(f"  [{i + 1}/{len(test_queries)}] {scope_label} {query[:50]}...")

        # ── Method 1: Base LLM (no context) ──
        t0 = time.time()
        base_answer = llm_generate(
            query,
            system="You are an IT support assistant. Answer the user's question concisely.",
            model=model,
        )
        base_time = time.time() - t0

        # ── Method 2: RAG (with relevance-filtered retrieved context) ──
        t1 = time.time()
        ticket_results = ticket_retriever.search(query, top_k=3, use_reranking=True)
        kb_results = kb_retriever.search(query, top_k=3, use_reranking=True)
        # format_context applies MIN_RERANK_SCORE threshold — off-topic docs are dropped
        context = format_context(ticket_results, kb_results)

        rag_answer = llm_generate(
            f"Answer based ONLY on the provided context. "
            f"If the context does not contain the answer or is not relevant to the question, "
            f"say exactly: 'I don't have this information in our knowledge base.'\n\n"
            f"Context:\n{context}\n\nQuestion: {query}",
            system=(
                "You are an IT support assistant. "
                "Answer ONLY from the provided context. Cite sources when available. "
                "Never invent steps, commands, URLs, or policy details not in the context."
            ),
            model=model,
        )
        rag_time = time.time() - t1

        # ── Score with LLM-as-Judge ──
        judge_prompt = build_judge_prompt(query, scope, base_answer, rag_answer)
        judge_raw = llm_generate(judge_prompt, model=model, temperature=0.0, max_tokens=350)

        scores = parse_judge_json(judge_raw)

        # Log how many docs passed the relevance filter
        passed_kb = sum(
            1 for d in kb_results
            if d.get("rerank_score") is None or float(d.get("rerank_score", 0)) >= MIN_RERANK_SCORE
        )
        passed_tickets = sum(
            1 for d in ticket_results
            if d.get("rerank_score") is None or float(d.get("rerank_score", 0)) >= MIN_RERANK_SCORE
        )

        results.append({
            "query": query,
            "scope": scope,
            "query_idx": i,
            "base_answer": base_answer[:300],
            "rag_answer": rag_answer[:300],
            "context_snippet": context[:200],
            "base_latency_s": round(base_time, 2),
            "rag_latency_s": round(rag_time, 2),
            "kb_docs_retrieved": len(kb_results),
            "ticket_docs_retrieved": len(ticket_results),
            "kb_docs_passed_filter": passed_kb,
            "ticket_docs_passed_filter": passed_tickets,
            **{f"judge_{k}": v for k, v in scores.items()},
        })

    # ── Save raw results ──
    rdf = pd.DataFrame(results)
    rdf.to_csv(EVAL_DIR / "hallucination_comparison.csv", index=False)

    # ── Summary reporting ──
    print(f"\n{'=' * 60}")
    print("HALLUCINATION COMPARISON RESULTS")
    print(f"{'=' * 60}")

    def report_subset(df, label):
        total = len(df)
        if total == 0:
            return {}

        base_h_col = df.get("judge_hallucination_in_a", pd.Series(dtype=object))
        rag_h_col = df.get("judge_hallucination_in_b", pd.Series(dtype=object))

        # Handle both bool and string "true"/"false" from the model
        def coerce_bool(series):
            def to_bool(v):
                if isinstance(v, bool):
                    return v
                if isinstance(v, str):
                    return v.strip().lower() == "true"
                return bool(v) if pd.notna(v) else False
            return series.apply(to_bool)

        base_hall = int(coerce_bool(base_h_col).sum())
        rag_hall = int(coerce_bool(rag_h_col).sum())

        wins = df.get("judge_which_is_better", pd.Series(dtype=str))
        b_wins = int((wins.str.upper() == "B").sum())
        a_wins = int((wins.str.upper() == "A").sum())
        ties = total - b_wins - a_wins

        print(f"\n  [{label}] ({total} queries)")
        print(f"  Base LLM hallucination: {base_hall}/{total} ({base_hall / total * 100:.0f}%)")
        print(f"  RAG hallucination:      {rag_hall}/{total} ({rag_hall / total * 100:.0f}%)")
        print(f"  RAG wins: {b_wins}/{total}  Base wins: {a_wins}/{total}  Ties: {ties}/{total}")

        for prefix, lbl in [("judge_answer_a", "Base LLM"), ("judge_answer_b", "RAG System")]:
            g = pd.to_numeric(df.get(f"{prefix}_groundedness", pd.Series(dtype=float)), errors="coerce").mean()
            s = pd.to_numeric(df.get(f"{prefix}_specificity", pd.Series(dtype=float)), errors="coerce").mean()
            a = pd.to_numeric(df.get(f"{prefix}_accuracy", pd.Series(dtype=float)), errors="coerce").mean()
            if not np.isnan(g):
                print(f"    {lbl}: Ground={g:.2f}/5  Specific={s:.2f}/5  Accuracy={a:.2f}/5")

        return {
            "total": total,
            "base_hallucination_count": base_hall,
            "base_hallucination_rate": f"{base_hall / total * 100:.0f}%",
            "rag_hallucination_count": rag_hall,
            "rag_hallucination_rate": f"{rag_hall / total * 100:.0f}%",
            "rag_win_count": b_wins,
            "rag_win_rate": f"{b_wins / total * 100:.0f}%",
        }

    it_df = rdf[rdf["scope"] == "it"]
    oos_df = rdf[rdf["scope"] == "oos"]

    it_stats = report_subset(it_df, "IT Support Queries (in-scope)")
    oos_stats = report_subset(oos_df, "Out-of-Scope Queries")
    all_stats = report_subset(rdf, "ALL Queries Combined")

    print(f"\n  Avg latency — Base: {rdf['base_latency_s'].mean():.1f}s  "
          f"RAG: {rdf['rag_latency_s'].mean():.1f}s")
    avg_kb_filtered = rdf["kb_docs_passed_filter"].mean()
    avg_ticket_filtered = rdf["ticket_docs_passed_filter"].mean()
    print(f"  Avg context docs passed relevance filter: "
          f"KB={avg_kb_filtered:.1f}/3  Tickets={avg_ticket_filtered:.1f}/3")

    # Save summary
    summary = {
        "it_queries": it_stats,
        "oos_queries": oos_stats,
        "all_queries": all_stats,
        "avg_base_latency_s": round(float(rdf["base_latency_s"].mean()), 2),
        "avg_rag_latency_s": round(float(rdf["rag_latency_s"].mean()), 2),
        "min_rerank_score_threshold": MIN_RERANK_SCORE,
    }
    with open(EVAL_DIR / "hallucination_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Generate chart (IT queries only — this is the meaningful comparison) ──
    _generate_chart(it_df, all_stats, rdf)

    print(f"\n  evaluation/hallucination_comparison.csv")
    print(f"  evaluation/hallucination_summary.json")
    print(f"  evaluation/hallucination_chart.png")


def _generate_chart(it_df: pd.DataFrame, all_stats: dict, rdf: pd.DataFrame):
    """Generate a 3-panel chart showing IT-scope results, quality scores, and win rate."""

    def coerce_bool(series):
        def to_bool(v):
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.strip().lower() == "true"
            return bool(v) if pd.notna(v) else False
        return series.apply(to_bool)

    total_it = max(len(it_df), 1)
    base_hall_it = int(coerce_bool(
        it_df.get("judge_hallucination_in_a", pd.Series(dtype=object))
    ).sum())
    rag_hall_it = int(coerce_bool(
        it_df.get("judge_hallucination_in_b", pd.Series(dtype=object))
    ).sum())

    wins = rdf.get("judge_which_is_better", pd.Series(dtype=str))
    b_wins = int((wins.str.upper() == "B").sum())
    a_wins = int((wins.str.upper() == "A").sum())
    ties = len(rdf) - b_wins - a_wins

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Chart 1: Hallucination rates (IT queries only)
    ax = axes[0]
    rates = [base_hall_it / total_it * 100, rag_hall_it / total_it * 100]
    bars = ax.bar(["Base LLM", "RAG System"], rates,
                  color=["#EF5350", "#4ADE80"], alpha=0.85)
    ax.set_title("Hallucination Rate — IT Queries (%)", fontweight="bold")
    ax.set_ylim(0, 100)
    for b, v in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                f"{v:.0f}%", ha="center", fontweight="bold")
    ax.set_ylabel("Hallucination Rate (%)")
    ax.text(0.5, -0.15, "(Lower is better)", ha="center", transform=ax.transAxes,
            fontsize=9, color="gray")

    # Chart 2: Quality scores (IT queries)
    ax = axes[1]
    metrics = ["groundedness", "specificity", "accuracy"]
    base_scores = [
        pd.to_numeric(it_df.get(f"judge_answer_a_{m}", pd.Series(dtype=float)),
                      errors="coerce").mean()
        for m in metrics
    ]
    rag_scores = [
        pd.to_numeric(it_df.get(f"judge_answer_b_{m}", pd.Series(dtype=float)),
                      errors="coerce").mean()
        for m in metrics
    ]
    x = np.arange(len(metrics))
    ax.bar(x - 0.18, base_scores, 0.35, label="Base LLM", color="#EF5350", alpha=0.8)
    ax.bar(x + 0.18, rag_scores, 0.35, label="RAG System", color="#4ADE80", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.title() for m in metrics])
    ax.set_ylim(0, 5.5)
    ax.set_title("Quality Scores — IT Queries (1-5)", fontweight="bold")
    ax.legend()
    ax.set_ylabel("Score (higher is better)")

    # Chart 3: Win rate (all queries)
    ax = axes[2]
    ax.pie(
        [b_wins, a_wins, ties],
        labels=["RAG Wins", "Base Wins", "Tie"],
        colors=["#4ADE80", "#EF5350", "#888888"],
        autopct="%1.0f%%",
        startangle=90,
    )
    ax.set_title("Head-to-Head — All Queries", fontweight="bold")

    plt.suptitle(
        "ITS RAG — Hallucination Comparison\n"
        "(Charts 1 & 2 use IT-scope queries only; Chart 3 uses all queries)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "hallucination_chart.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--num-queries", type=int, default=50, help="Number of test queries (max 50)")
    args = parser.parse_args()
    run_comparison(model=args.model, num_queries=args.num_queries)
