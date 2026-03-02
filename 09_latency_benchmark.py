"""
ITS RAG - Script 09: Latency Benchmark
========================================
Measures per-stage latency of the RAG pipeline.

Stages: Embedding, Dense Search, BM25 Search, RRF Fusion, Reranking, LLM Generation
Runs 20 queries, reports mean/p50/p95 per stage.

Usage:  python 09_latency_benchmark.py
        python 09_latency_benchmark.py --model qwen3:4b --num-queries 10
Output: evaluation/latency_results.json, evaluation/latency_chart.png
"""

import json, time, argparse, importlib.util, numpy as np, pandas as pd, requests
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

EVAL_DIR = Path("./evaluation"); EVAL_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = Path("./data/processed")

_rp = Path(__file__).parent / "04_hybrid_retrieval.py"
_sp = importlib.util.spec_from_file_location("hre", _rp)
_m = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_m)
HybridRetriever = _m.HybridRetriever

DEFAULT_MODEL = "qwen3:8b"
OLLAMA_URL = "http://localhost:11434"

TEST_QUERIES = [
    "VPN keeps disconnecting every few minutes",
    "error 0x80070005 permission denied",
    "laptop overheating and fan is loud",
    "Outlook not syncing emails after update",
    "account locked need reset MFA",
    "Teams consuming too much memory",
    "printer queue stuck cannot print",
    "BSOD KERNEL_DATA_INPAGE_ERROR",
    "How do I reset my password?",
    "Software installation failed permission error",
    "WiFi intermittent disconnections building 3",
    "new hire account setup Active Directory",
    "Shared mailbox not showing in Outlook",
    "laptop won't turn on after hard reset",
    "How to clear Teams cache?",
    "VPN MTU setting for remote work",
    "printer offline error network printer",
    "Office 365 activation failed",
    "Blue screen after Windows update",
    "Email attachment size limit exceeded",
]


def llm_call(prompt, model, system="You are an IT support assistant."):
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json={
            "model": model, "stream": False,
            "messages": [{"role":"system","content":system},{"role":"user","content":prompt}],
            "options": {"temperature": 0.1},
        }, timeout=180)
        d = resp.json()
        return d.get("message",{}).get("content","") if "message" in d else d.get("response","")
    except Exception as e:
        return f"[ERROR] {e}"


def run_benchmark(model=DEFAULT_MODEL, num_queries=20):
    print("="*60+f"\nITS RAG — Latency Benchmark ({num_queries} queries)\n"+"="*60)
    queries = TEST_QUERIES[:num_queries]

    retriever = HybridRetriever("its_tickets", load_reranker=True)
    kb_retriever = HybridRetriever("its_knowledge_base", load_reranker=True)

    records = []
    for i, q in enumerate(queries):
        print(f"  [{i+1}/{len(queries)}] {q[:45]}...")
        r = {"query": q}

        # Embedding
        t=time.time(); emb=retriever.embedder.encode(q); r["embed_ms"]=round((time.time()-t)*1000,1)

        # Dense search
        t=time.time(); dense=retriever.dense_search(q, top_k=20); r["dense_ms"]=round((time.time()-t)*1000,1)

        # BM25 search
        t=time.time(); bm25=retriever.bm25_search(q, top_k=20); r["bm25_ms"]=round((time.time()-t)*1000,1)

        # RRF fusion
        t=time.time(); fused=retriever.reciprocal_rank_fusion(dense, bm25); r["rrf_ms"]=round((time.time()-t)*1000,1)

        # Reranking
        t=time.time(); reranked=retriever.rerank(q, fused, top_k=5); r["rerank_ms"]=round((time.time()-t)*1000,1)

        # KB retrieval
        t=time.time(); kb=kb_retriever.search(q, top_k=3); r["kb_retrieval_ms"]=round((time.time()-t)*1000,1)

        # LLM generation
        context = "\n".join([d.get("text","")[:300] for d in reranked[:3]+kb[:2]])
        prompt = f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer concisely."
        t=time.time(); ans=llm_call(prompt, model); r["llm_ms"]=round((time.time()-t)*1000,1)

        r["total_ms"] = round(sum(r[k] for k in r if k.endswith("_ms")), 1)
        records.append(r)

    df = pd.DataFrame(records)

    # Summary stats
    stages = ["embed_ms","dense_ms","bm25_ms","rrf_ms","rerank_ms","kb_retrieval_ms","llm_ms","total_ms"]
    summary = {}
    print(f"\n{'='*60}\nLATENCY SUMMARY\n{'='*60}")
    print(f"{'Stage':<20} {'Mean':>8} {'P50':>8} {'P95':>8}")
    print("-"*48)
    for s in stages:
        if s in df.columns:
            vals = df[s].values
            summary[s] = {"mean":round(np.mean(vals),1),"p50":round(np.percentile(vals,50),1),"p95":round(np.percentile(vals,95),1)}
            print(f"  {s:<18} {summary[s]['mean']:>7.1f} {summary[s]['p50']:>7.1f} {summary[s]['p95']:>7.1f}")

    with open(EVAL_DIR/"latency_results.json","w") as f:
        json.dump({"summary":summary,"per_query":records},f,indent=2)

    # Chart: stacked bar showing stage breakdown
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: mean latency per stage
    stage_labels = [s.replace("_ms","") for s in stages if s != "total_ms"]
    stage_means = [summary[s]["mean"] for s in stages if s != "total_ms"]
    colors = ["#5B8DEF","#3DD9B4","#F0983A","#A78BFA","#EF5350","#FACC15","#4ADE80"]
    bars = ax1.barh(stage_labels, stage_means, color=colors[:len(stage_labels)])
    ax1.set_xlabel("Latency (ms)"); ax1.set_title("Mean Latency per Stage", fontweight="bold")
    for b,v in zip(bars, stage_means):
        ax1.text(b.get_width()+5, b.get_y()+b.get_height()/2, f"{v:.0f}ms", va="center", fontsize=9)

    # Right: total pipeline latency distribution
    ax2.hist(df["total_ms"].values, bins=10, color="#5B8DEF", alpha=0.8, edgecolor="white")
    ax2.axvline(summary["total_ms"]["mean"], color="#EF5350", linestyle="--", label=f"Mean: {summary['total_ms']['mean']:.0f}ms")
    ax2.axvline(summary["total_ms"]["p95"], color="#F0983A", linestyle="--", label=f"P95: {summary['total_ms']['p95']:.0f}ms")
    ax2.set_xlabel("Total Latency (ms)"); ax2.set_title("Pipeline Latency Distribution", fontweight="bold")
    ax2.legend()

    plt.suptitle("ITS RAG — Latency Benchmark", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(EVAL_DIR/"latency_chart.png", dpi=150, bbox_inches="tight"); plt.close()

    print(f"\n✅ evaluation/latency_results.json\n✅ evaluation/latency_chart.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--num-queries", type=int, default=20)
    a = p.parse_args()
    run_benchmark(model=a.model, num_queries=a.num_queries)
