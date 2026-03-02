"""
ITS RAG - Script 08: Hallucination Comparison (RAG vs Base LLM)
================================================================
Compatible with: 04_hybrid_retrieval.py, 05_rag_pipeline.py
Place in project root alongside scripts 01-06.

Sends 50 queries through both:
  1. Base LLM (no RAG context) — answers from training data only
  2. RAG system (with retrieved context) — grounded answers

Uses LLM-as-judge to score groundedness, specificity, accuracy.

Requires: Ollama running with your model pulled (qwen3:8b or qwen3:4b)

Usage:
  python 08_hallucination_comparison.py
  python 08_hallucination_comparison.py --model qwen3:4b
  python 08_hallucination_comparison.py --num-queries 20   (quick test)

Outputs:
  evaluation/hallucination_comparison.csv
  evaluation/hallucination_summary.json
  evaluation/hallucination_chart.png
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


def llm_generate(prompt: str, system: str = "", model: str = DEFAULT_MODEL,
                 temperature: float = 0.1, max_tokens: int = 1024) -> str:
    """Call Ollama /api/chat endpoint (same API as your llm_ollama.py OllamaClient)."""
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
        # Handle both /api/chat and /api/generate response shapes
        if "message" in data and isinstance(data["message"], dict):
            return data["message"].get("content", "").strip()
        return data.get("response", "").strip()
    except Exception as e:
        return f"[LLM_ERROR] {e}"


def format_context(ticket_results: list, kb_results: list) -> str:
    """Format retrieved results into context string for RAG prompt."""
    parts = []
    for i, doc in enumerate(kb_results):
        meta = doc.get("metadata", {})
        text = doc.get("text", "")[:500]
        parts.append(f"[KB Source: {meta.get('source_file', meta.get('doc_title', 'KB'))}]\n{text}")
    for i, doc in enumerate(ticket_results):
        meta = doc.get("metadata", {})
        text = doc.get("text", "")[:300]
        tid = meta.get("ticket_id", doc["id"])
        parts.append(f"[Ticket: {tid} | {meta.get('title', 'N/A')}]\n{text}")
    return "\n\n".join(parts) if parts else "No relevant context found."


# ── Test queries covering different categories ──
ALL_TEST_QUERIES = [
    # VPN (KB has runbook)
    "How do I fix VPN connection drops?",
    "My VPN keeps disconnecting every 10 minutes on Windows 11",
    "VPN client version 4.1 is not connecting",
    "What MTU setting should I use for VPN?",
    "How to collect VPN logs for escalation?",
    # Password / Account (KB has runbook)
    "My account is locked out, how do I unlock it?",
    "How to reset MFA enrollment?",
    "What causes account lockout in Active Directory?",
    "How to verify user identity for password reset?",
    "How do I enable self-service password reset?",
    # Email (KB has runbook)
    "Outlook is not syncing my emails",
    "How to clear Outlook cache?",
    "Shared mailbox not showing in Outlook",
    "What is the mailbox size quota?",
    "How to check Microsoft 365 service health?",
    # Hardware (KB has runbook)
    "Laptop showing blue screen IRQL_NOT_LESS_OR_EQUAL",
    "My laptop won't turn on after hard reset",
    "How to fix overheating laptop?",
    "What does KERNEL_DATA_INPAGE_ERROR mean?",
    "How to run Windows Memory Diagnostic?",
    # Software (KB has runbook)
    "Software installation failed with permission error",
    "Product activation failed for Office 365",
    "How to get Adobe Creative Cloud license?",
    "Error: disk space insufficient for installation",
    "How to fix license limit reached error?",
    # Teams (KB has article)
    "Microsoft Teams is running very slow and using too much memory",
    "How to clear Teams cache?",
    "Should I upgrade to New Teams?",
    "Teams GPU hardware acceleration setting",
    "How to reduce Teams memory usage?",
    # Printer (KB has article)
    "Print jobs stuck in queue, nothing printing",
    "How to set up a network printer?",
    "Printer showing offline status",
    "How does secure print work?",
    "Poor print quality from network printer",
    # General IT (no specific KB)
    "How do I set up dual monitors?",
    "My mouse cursor is jumping around",
    "WiFi keeps disconnecting in building 3",
    "How to request a new laptop?",
    "Can I install personal software on work computer?",
    # Edge cases
    "What is the SLA for critical tickets?",
    "How to escalate a ticket to Tier 2?",
    "What are the working hours for help desk?",
    "How to submit a facilities request?",
    "What antivirus software do we use?",
    # Out of scope (should NOT hallucinate)
    "How to make pasta carbonara?",
    "What's the weather today?",
    "Tell me about quantum computing",
    "How to file taxes in Arizona?",
    "What's the capital of France?",
]


def run_comparison(model: str = DEFAULT_MODEL, num_queries: int = 50):
    print("=" * 60)
    print("ITS RAG — Hallucination Comparison: RAG vs Base LLM")
    print(f"Model: {model} | Queries: {num_queries}")
    print("=" * 60)

    # Use only requested number of queries
    test_queries = ALL_TEST_QUERIES[:num_queries]

    # Load retrievers
    print("\nLoading retrievers...")
    ticket_retriever = HybridRetriever("its_tickets", load_reranker=True)
    kb_retriever = HybridRetriever("its_knowledge_base", load_reranker=True)

    results = []
    print(f"\nRunning {len(test_queries)} queries through both systems...\n")

    for i, query in enumerate(test_queries):
        print(f"  [{i + 1}/{len(test_queries)}] {query[:50]}...")

        # ── Method 1: Base LLM (no context) ──
        t0 = time.time()
        base_answer = llm_generate(
            query,
            system="You are an IT support assistant. Answer the user's question concisely.",
            model=model,
        )
        base_time = time.time() - t0

        # ── Method 2: RAG (with retrieved context) ──
        t1 = time.time()
        ticket_results = ticket_retriever.search(query, top_k=3, use_reranking=True)
        kb_results = kb_retriever.search(query, top_k=3, use_reranking=True)
        context = format_context(ticket_results, kb_results)

        rag_answer = llm_generate(
            f"Answer based ONLY on the provided context. If the context doesn't "
            f"contain the answer, say 'I don't have this information in our knowledge base.'\n\n"
            f"Context:\n{context}\n\nQuestion: {query}",
            system="You are an IT support assistant. Answer ONLY from the provided context. Cite sources.",
            model=model,
        )
        rag_time = time.time() - t1

        # ── Score with LLM-as-Judge ──
        judge_prompt = (
            f"Score these two answers to the same IT support question.\n\n"
            f"Question: {query}\n\n"
            f"Answer A (no context): {base_answer[:500]}\n\n"
            f"Answer B (with retrieved KB): {rag_answer[:500]}\n\n"
            f"Score each answer 1-5 on:\n"
            f"1. Groundedness (cites real sources vs makes things up?)\n"
            f"2. Specificity (org-specific procedures vs generic advice?)\n"
            f"3. Accuracy (correct and actionable?)\n\n"
            f"Respond in JSON only, no markdown:\n"
            f'{{"answer_a_groundedness":N,"answer_a_specificity":N,"answer_a_accuracy":N,'
            f'"answer_b_groundedness":N,"answer_b_specificity":N,"answer_b_accuracy":N,'
            f'"which_is_better":"A or B or TIE",'
            f'"hallucination_in_a":true/false,"hallucination_in_b":true/false}}'
        )
        judge_raw = llm_generate(judge_prompt, model=model, temperature=0.0, max_tokens=300)

        # Parse judge result
        scores = {}
        try:
            json_match = re.search(r'\{.*\}', judge_raw, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
        except Exception:
            scores = {"parse_error": True}

        results.append({
            "query": query,
            "query_idx": i,
            "base_answer": base_answer[:300],
            "rag_answer": rag_answer[:300],
            "base_latency_s": round(base_time, 2),
            "rag_latency_s": round(rag_time, 2),
            "kb_sources_found": len(kb_results),
            "ticket_sources_found": len(ticket_results),
            **{f"judge_{k}": v for k, v in scores.items()},
        })

    # ── Save & summarize ──
    rdf = pd.DataFrame(results)
    rdf.to_csv(EVAL_DIR / "hallucination_comparison.csv", index=False)

    print(f"\n{'=' * 60}")
    print("HALLUCINATION COMPARISON RESULTS")
    print(f"{'=' * 60}")

    total = len(rdf)

    # Count hallucinations
    base_h = rdf.get("judge_hallucination_in_a", pd.Series(dtype=float))
    rag_h = rdf.get("judge_hallucination_in_b", pd.Series(dtype=float))
    base_hall = int(base_h.sum()) if len(base_h) > 0 else 0
    rag_hall = int(rag_h.sum()) if len(rag_h) > 0 else 0

    print(f"\n  Base LLM hallucination: {base_hall}/{total} ({base_hall / max(total, 1) * 100:.0f}%)")
    print(f"  RAG hallucination:      {rag_hall}/{total} ({rag_hall / max(total, 1) * 100:.0f}%)")

    # Average scores
    for prefix, label in [("judge_answer_a", "Base LLM"), ("judge_answer_b", "RAG System")]:
        g = rdf.get(f"{prefix}_groundedness", pd.Series(dtype=float)).mean()
        s = rdf.get(f"{prefix}_specificity", pd.Series(dtype=float)).mean()
        a = rdf.get(f"{prefix}_accuracy", pd.Series(dtype=float)).mean()
        if not np.isnan(g):
            print(f"\n  {label}: Groundedness={g:.2f}/5  Specificity={s:.2f}/5  Accuracy={a:.2f}/5")

    # Win rate
    wins = rdf.get("judge_which_is_better", pd.Series(dtype=str))
    b_wins = int((wins.str.upper() == "B").sum())
    a_wins = int((wins.str.upper() == "A").sum())
    ties = total - b_wins - a_wins
    print(f"\n  RAG wins: {b_wins}/{total} ({b_wins / max(total, 1) * 100:.0f}%)")
    print(f"  Base wins: {a_wins}/{total}")
    print(f"  Ties: {ties}/{total}")

    # Latency
    print(f"\n  Avg latency — Base: {rdf['base_latency_s'].mean():.1f}s  RAG: {rdf['rag_latency_s'].mean():.1f}s")

    # Save summary
    hall_summary = {
        "total_queries": total,
        "base_hallucination_count": base_hall,
        "base_hallucination_rate": f"{base_hall / max(total, 1) * 100:.0f}%",
        "rag_hallucination_count": rag_hall,
        "rag_hallucination_rate": f"{rag_hall / max(total, 1) * 100:.0f}%",
        "rag_win_rate": f"{b_wins / max(total, 1) * 100:.0f}%",
        "avg_base_latency_s": round(rdf["base_latency_s"].mean(), 2),
        "avg_rag_latency_s": round(rdf["rag_latency_s"].mean(), 2),
    }
    with open(EVAL_DIR / "hallucination_summary.json", "w") as f:
        json.dump(hall_summary, f, indent=2)

    # ── Generate chart ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Chart 1: Hallucination rates
    ax = axes[0]
    rates = [base_hall / max(total, 1) * 100, rag_hall / max(total, 1) * 100]
    bars = ax.bar(["Base LLM", "RAG System"], rates, color=["#EF5350", "#4ADE80"], alpha=0.85)
    ax.set_title("Hallucination Rate (%)", fontweight="bold")
    ax.set_ylim(0, 100)
    for b, v in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{v:.0f}%",
                ha="center", fontweight="bold")

    # Chart 2: Quality scores
    ax = axes[1]
    metrics = ["groundedness", "specificity", "accuracy"]
    base_scores = [rdf.get(f"judge_answer_a_{m}", pd.Series(dtype=float)).mean() for m in metrics]
    rag_scores = [rdf.get(f"judge_answer_b_{m}", pd.Series(dtype=float)).mean() for m in metrics]
    x = np.arange(len(metrics))
    ax.bar(x - 0.18, base_scores, 0.35, label="Base LLM", color="#EF5350", alpha=0.8)
    ax.bar(x + 0.18, rag_scores, 0.35, label="RAG System", color="#4ADE80", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.title() for m in metrics])
    ax.set_ylim(0, 5.5)
    ax.set_title("Quality Scores (1-5)", fontweight="bold")
    ax.legend()

    # Chart 3: Win rate
    ax = axes[2]
    ax.pie([b_wins, a_wins, ties], labels=["RAG Wins", "Base Wins", "Tie"],
           colors=["#4ADE80", "#EF5350", "#888888"], autopct="%1.0f%%", startangle=90)
    ax.set_title("Head-to-Head Comparison", fontweight="bold")

    plt.suptitle("ITS RAG — Hallucination Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "hallucination_chart.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n✅ evaluation/hallucination_comparison.csv")
    print(f"✅ evaluation/hallucination_summary.json")
    print(f"✅ evaluation/hallucination_chart.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--num-queries", type=int, default=50, help="Number of test queries")
    args = parser.parse_args()
    run_comparison(model=args.model, num_queries=args.num_queries)
