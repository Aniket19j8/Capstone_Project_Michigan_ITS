"""
ITS RAG - Phase 5 Evaluation Suite

This script creates meaningful project-level evaluation artifacts:
  - data readiness / dataset quality
  - 4-department Model 2 triage quality
  - retrieval self-checks (optional, requires vector store)
  - Model 1 chatbot contract checks (optional, requires backend/Ollama/RAG)
  - consolidated phase5_summary.json and phase5_report.md

Usage examples:
  # Cheap checks only; safe before vector DB build
  python 11_phase5_evaluation.py --data-readiness --triage

  # AWS run after vector DB is built
  python 11_phase5_evaluation.py --all --sample-size 1000

  # Heavy chatbot contract check with a small case count
  python 11_phase5_evaluation.py --chatbot --max-chatbot-cases 12
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path
from typing import Any

import pandas as pd

from department_mapping import DEPARTMENTS, classify_department


PROCESSED_DIR = Path("./data/processed")
EVAL_DIR = Path("./evaluation")
EVAL_DIR.mkdir(parents=True, exist_ok=True)


PHASE5_QUERIES = [
    {
        "query": "VPN disconnects every 10 minutes after Windows update",
        "expected_department": "Network & Security",
    },
    {
        "query": "Laptop screen flickers and battery drains quickly",
        "expected_department": "End-User & Desktop Support",
    },
    {
        "query": "Teams calendar not syncing with Outlook meetings",
        "expected_department": "Applications & Data Services",
    },
    {
        "query": "Active Directory group policy not applying to users",
        "expected_department": "IT Infrastructure & Platform",
    },
    {
        "query": "Printer queue stuck and users cannot print",
        "expected_department": "End-User & Desktop Support",
    },
    {
        "query": "Firewall blocking SFTP connection from branch office",
        "expected_department": "Network & Security",
    },
    {
        "query": "Database login error MSSQL 18456 in production app",
        "expected_department": "Applications & Data Services",
    },
    {
        "query": "Cloud backup job failed for storage volume",
        "expected_department": "IT Infrastructure & Platform",
    },
]


CHATBOT_CASES = [
    {
        "issue": "VPN keeps disconnecting every few minutes on my Windows laptop.",
        "expected_escalation": False,
    },
    {
        "issue": "Outlook is not syncing emails after an update.",
        "expected_escalation": False,
    },
    {
        "issue": "The firewall change for production network segmentation is failing.",
        "expected_escalation": True,
    },
    {
        "issue": "My laptop is overheating and the fan is very loud.",
        "expected_escalation": False,
    },
]


def _default_tickets_path() -> Path:
    all_tickets = PROCESSED_DIR / "all_tickets.csv"
    major = PROCESSED_DIR / "its_tickets_80k.csv"
    return all_tickets if all_tickets.exists() else major


def _safe_read_csv(path: Path, sample_size: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Ticket dataset not found: {path}")
    if sample_size and sample_size > 0:
        return pd.read_csv(path, nrows=sample_size, low_memory=False).fillna("")
    return pd.read_csv(path, low_memory=False).fillna("")


def _pct(value: float) -> float:
    return round(float(value) * 100, 2)


def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def evaluate_data_readiness(df: pd.DataFrame, source_path: Path) -> dict[str, Any]:
    required = ["ticket_id", "title", "description", "category", "component", "status"]
    useful = ["resolution", "severity", "source", "source_system", "external_source", "language", "tags"]

    missing_required_cols = [c for c in required if c not in df.columns]
    missing_useful_cols = [c for c in useful if c not in df.columns]

    completeness = {}
    for col in required + [c for c in useful if c in df.columns]:
        if col in df.columns:
            completeness[col] = _pct((df[col].astype(str).str.strip() != "").mean())

    duplicate_ticket_ids = int(df["ticket_id"].duplicated().sum()) if "ticket_id" in df.columns else None
    duplicate_embedding_text = (
        int(df["embedding_text"].duplicated().sum()) if "embedding_text" in df.columns else None
    )

    resolution_col = "resolution_clean" if "resolution_clean" in df.columns else "resolution"
    resolution_coverage = (
        _pct((df[resolution_col].astype(str).str.strip() != "").mean())
        if resolution_col in df.columns else 0.0
    )

    quality_summary = {}
    if "quality_score" in df.columns:
        q = pd.to_numeric(df["quality_score"], errors="coerce").dropna()
        if len(q) > 0:
            quality_summary = {
                "mean": round(float(q.mean()), 4),
                "median": round(float(q.median()), 4),
                "p10": round(float(q.quantile(0.10)), 4),
                "p90": round(float(q.quantile(0.90)), 4),
            }

    report = {
        "source_path": str(source_path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_required_columns": missing_required_cols,
        "missing_useful_columns": missing_useful_cols,
        "field_completeness_percent": completeness,
        "resolution_coverage_percent": resolution_coverage,
        "duplicate_ticket_ids": duplicate_ticket_ids,
        "duplicate_embedding_text": duplicate_embedding_text,
        "quality_score": quality_summary,
        "category_distribution": df["category"].value_counts().head(30).to_dict()
        if "category" in df.columns else {},
        "component_distribution": df["component"].value_counts().head(30).to_dict()
        if "component" in df.columns else {},
        "status_distribution": df["status"].value_counts().to_dict()
        if "status" in df.columns else {},
        "language_distribution": df["language"].value_counts().to_dict()
        if "language" in df.columns else {},
        "source_system_distribution": (
            df["source_system"].value_counts().head(30).to_dict()
            if "source_system" in df.columns else
            df["external_source"].value_counts().head(30).to_dict()
            if "external_source" in df.columns else
            df["source"].value_counts().head(30).to_dict()
            if "source" in df.columns else {}
        ),
    }
    _write_json(EVAL_DIR / "phase5_data_readiness.json", report)
    return report


def evaluate_triage(df: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for _, r in df.iterrows():
        result = classify_department(
            r.get("department", ""),
            r.get("mapped_department", ""),
            r.get("category", ""),
            r.get("component", ""),
            r.get("tags", ""),
            r.get("title", ""),
            r.get("description", ""),
        )
        existing = str(r.get("department") or r.get("mapped_department") or "").strip()
        rows.append({
            "ticket_id": r.get("ticket_id", ""),
            "category": r.get("category", ""),
            "component": r.get("component", ""),
            "existing_department": existing,
            "predicted_department": result["department"],
            "confidence": result["confidence"],
            "source": result["source"],
            "reason": result["reason"],
            "matches_existing_department": bool(existing and existing == result["department"]),
        })

    out = pd.DataFrame(rows)
    out.to_csv(EVAL_DIR / "phase5_triage_results.csv", index=False)

    label_counts = out["predicted_department"].value_counts().to_dict()
    avg_conf = float(out["confidence"].mean()) if len(out) else 0.0
    low_conf = out[out["confidence"] < 0.6]

    case_results = []
    for case in PHASE5_QUERIES:
        pred = classify_department(case["query"])
        case_results.append({
            **case,
            "predicted_department": pred["department"],
            "confidence": pred["confidence"],
            "pass": pred["department"] == case["expected_department"],
            "reason": pred["reason"],
        })

    case_accuracy = (
        sum(1 for r in case_results if r["pass"]) / len(case_results)
        if case_results else 0.0
    )

    metrics = {
        "rows_evaluated": int(len(out)),
        "predicted_department_distribution": label_counts,
        "avg_confidence": round(avg_conf, 4),
        "low_confidence_count": int(len(low_conf)),
        "low_confidence_percent": _pct(len(low_conf) / len(out)) if len(out) else 0.0,
        "agreement_with_existing_department": (
            round(float(out["matches_existing_department"].mean()), 4)
            if out["existing_department"].astype(bool).any() else None
        ),
        "golden_case_accuracy": round(case_accuracy, 4),
        "golden_cases": case_results,
    }
    _write_json(EVAL_DIR / "phase5_triage_metrics.json", metrics)
    return metrics


def _load_hybrid_retriever_class():
    retriever_path = Path(__file__).parent / "04_hybrid_retrieval.py"
    spec = importlib.util.spec_from_file_location("hybrid_retrieval_engine", retriever_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec from {retriever_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HybridRetriever


def evaluate_retrieval_smoke(df: pd.DataFrame, sample_size: int = 100) -> dict[str, Any]:
    HybridRetriever = _load_hybrid_retriever_class()
    retriever = HybridRetriever("its_tickets", load_reranker=True)

    sample = df[df.get("embedding_text", "").astype(str).str.len() > 20].head(sample_size)
    records = []
    for _, r in sample.iterrows():
        query = str(r.get("embedding_text", ""))
        target_id = str(r.get("unified_id", ""))
        t0 = time.time()
        results = retriever.search(query, top_k=10, use_reranking=True)
        latency_ms = round((time.time() - t0) * 1000, 1)
        ids = [str(x.get("id", "")) for x in results]
        records.append({
            "unified_id": target_id,
            "ticket_id": str(r.get("ticket_id", "")),
            "category": str(r.get("category", "")),
            "department": str(r.get("department", "")),
            "hit_at_1": target_id in ids[:1],
            "hit_at_5": target_id in ids[:5],
            "hit_at_10": target_id in ids[:10],
            "latency_ms": latency_ms,
        })

    out = pd.DataFrame(records)
    out.to_csv(EVAL_DIR / "phase5_retrieval_smoke_results.csv", index=False)
    metrics = {
        "rows_evaluated": int(len(out)),
        "hit_at_1": round(float(out["hit_at_1"].mean()), 4) if len(out) else 0.0,
        "hit_at_5": round(float(out["hit_at_5"].mean()), 4) if len(out) else 0.0,
        "hit_at_10": round(float(out["hit_at_10"].mean()), 4) if len(out) else 0.0,
        "avg_latency_ms": round(float(out["latency_ms"].mean()), 1) if len(out) else 0.0,
        "p95_latency_ms": round(float(out["latency_ms"].quantile(0.95)), 1) if len(out) else 0.0,
    }
    _write_json(EVAL_DIR / "phase5_retrieval_smoke_metrics.json", metrics)
    return metrics


def evaluate_chatbot_contract(max_cases: int = 4) -> dict[str, Any]:
    from its_brain import MAX_AI_ATTEMPTS, run_model1

    records = []
    for idx, case in enumerate(CHATBOT_CASES[:max_cases]):
        t0 = time.time()
        result = run_model1(case["issue"], attempt_number=1)
        latency_ms = round((time.time() - t0) * 1000, 1)
        steps = result.get("recommended_steps", [])
        records.append({
            "case_id": idx + 1,
            "issue": case["issue"],
            "ok": bool(result.get("ok")),
            "confidence": result.get("confidence"),
            "has_response": bool(str(result.get("response", "")).strip()),
            "recommended_step_count": len(steps) if isinstance(steps, list) else 0,
            "escalation_recommended": bool(result.get("escalation_recommended")),
            "expected_escalation": case["expected_escalation"],
            "latency_ms": latency_ms,
        })

    out = pd.DataFrame(records)
    out.to_csv(EVAL_DIR / "phase5_chatbot_contract_results.csv", index=False)
    metrics = {
        "cases_evaluated": int(len(out)),
        "max_ai_attempts": MAX_AI_ATTEMPTS,
        "schema_pass_rate": round(float((
            out["has_response"] & (out["recommended_step_count"] >= 0)
        ).mean()), 4) if len(out) else 0.0,
        "avg_latency_ms": round(float(out["latency_ms"].mean()), 1) if len(out) else 0.0,
        "avg_confidence": round(float(out["confidence"].mean()), 4) if len(out) else 0.0,
    }
    _write_json(EVAL_DIR / "phase5_chatbot_contract_metrics.json", metrics)
    return metrics


def write_markdown_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Phase 5 Evaluation Report",
        "",
        "## Purpose",
        "Evaluate data readiness, triage quality, retrieval behavior, latency, and chatbot contract quality before LoRA fine-tuning or production deployment.",
        "",
    ]

    if "data_readiness" in summary:
        dr = summary["data_readiness"]
        lines += [
            "## Data Readiness",
            f"- Rows evaluated: {dr.get('rows')}",
            f"- Resolution coverage: {dr.get('resolution_coverage_percent')}%",
            f"- Duplicate ticket IDs: {dr.get('duplicate_ticket_ids')}",
            f"- Missing required columns: {dr.get('missing_required_columns')}",
            "",
        ]

    if "triage" in summary:
        tr = summary["triage"]
        lines += [
            "## Model 2 Triage",
            f"- Rows evaluated: {tr.get('rows_evaluated')}",
            f"- Avg confidence: {tr.get('avg_confidence')}",
            f"- Low-confidence percent: {tr.get('low_confidence_percent')}%",
            f"- Golden case accuracy: {tr.get('golden_case_accuracy')}",
            "",
        ]

    if "retrieval_smoke" in summary:
        rt = summary["retrieval_smoke"]
        lines += [
            "## Retrieval Smoke Test",
            f"- Rows evaluated: {rt.get('rows_evaluated')}",
            f"- Hit@1: {rt.get('hit_at_1')}",
            f"- Hit@5: {rt.get('hit_at_5')}",
            f"- P95 latency: {rt.get('p95_latency_ms')} ms",
            "",
        ]

    if "chatbot_contract" in summary:
        cb = summary["chatbot_contract"]
        lines += [
            "## Model 1 Chatbot Contract",
            f"- Cases evaluated: {cb.get('cases_evaluated')}",
            f"- Schema pass rate: {cb.get('schema_pass_rate')}",
            f"- Avg latency: {cb.get('avg_latency_ms')} ms",
            "",
        ]

    lines += [
        "## Phase 6 Guidance",
        "Run LoRA fine-tuning only after these metrics identify a clear weakness in the base LLM, such as poor conversational tone, missing recommended steps, or weak escalation phrasing.",
        "Do not use LoRA to memorize tickets; keep ticket knowledge in RAG.",
        "",
    ]
    (EVAL_DIR / "phase5_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="ITS RAG Phase 5 Evaluation Suite")
    parser.add_argument("--tickets-path", default=str(_default_tickets_path()))
    parser.add_argument("--sample-size", type=int, default=0, help="Limit rows for cheap checks; 0 = all rows")
    parser.add_argument("--data-readiness", action="store_true")
    parser.add_argument("--triage", action="store_true")
    parser.add_argument("--retrieval-smoke", action="store_true")
    parser.add_argument("--chatbot", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-chatbot-cases", type=int, default=4)
    parser.add_argument("--retrieval-sample-size", type=int, default=100)
    args = parser.parse_args()

    if not any([args.data_readiness, args.triage, args.retrieval_smoke, args.chatbot, args.all]):
        args.data_readiness = True
        args.triage = True

    tickets_path = Path(args.tickets_path)
    df = _safe_read_csv(tickets_path, sample_size=args.sample_size or None)
    summary: dict[str, Any] = {
        "tickets_path": str(tickets_path),
        "rows_loaded": int(len(df)),
        "sample_size": args.sample_size or None,
        "generated_at_unix": int(time.time()),
    }

    if args.data_readiness or args.all:
        summary["data_readiness"] = evaluate_data_readiness(df, tickets_path)
    if args.triage or args.all:
        summary["triage"] = evaluate_triage(df)
    if args.retrieval_smoke or args.all:
        summary["retrieval_smoke"] = evaluate_retrieval_smoke(df, sample_size=args.retrieval_sample_size)
    if args.chatbot or args.all:
        summary["chatbot_contract"] = evaluate_chatbot_contract(max_cases=args.max_chatbot_cases)

    _write_json(EVAL_DIR / "phase5_summary.json", summary)
    write_markdown_report(summary)
    print("Phase 5 evaluation artifacts written to ./evaluation/")


if __name__ == "__main__":
    main()
