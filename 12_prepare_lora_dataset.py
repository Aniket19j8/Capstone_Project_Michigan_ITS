"""
ITS RAG - Phase 6: Prepare LoRA Fine-Tuning Dataset

Creates supervised chat examples for refining the assistant's tone, step
formatting, confidence calibration, and escalation phrasing.

Important:
  - This does NOT train a model.
  - This does NOT replace RAG.
  - It creates examples from historical tickets that already have resolutions.

Usage:
  python 12_prepare_lora_dataset.py \
    --tickets-path data/processed/all_tickets.csv \
    --max-examples 12000

Outputs:
  data/lora/its_lora_train.jsonl
  data/lora/its_lora_val.jsonl
  data/lora/its_lora_test.jsonl
  data/lora/lora_dataset_manifest.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd


PROCESSED_DIR = Path("./data/processed")
LORA_DIR = Path("./data/lora")
LORA_DIR.mkdir(parents=True, exist_ok=True)


SYSTEM_PROMPT = (
    "You are a Tier-1 IT helpdesk assistant inside a RAG system. "
    "Use the ticket context to write a natural support response. "
    "Return JSON ONLY with this schema: "
    "{\"reply\": string, \"recommended_steps\": string[], "
    "\"confidence\": number, \"escalation_recommended\": boolean, "
    "\"escalation_reason\": string}. "
    "Keep recommended_steps short, safe, and user-actionable."
)


def _default_tickets_path() -> Path:
    all_tickets = PROCESSED_DIR / "all_tickets.csv"
    major = PROCESSED_DIR / "its_tickets_80k.csv"
    return all_tickets if all_tickets.exists() else major


def _clean(text: object, limit: int = 2000) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    return value[:limit]


def _split_resolution(resolution: str) -> list[str]:
    """Turn resolution prose into 2-5 user-facing actions."""
    text = _clean(resolution, 1200)
    if not text:
        return []

    chunks = re.split(r"(?:\.\s+|;\s+|\n+|\s+-\s+)", text)
    steps = []
    for chunk in chunks:
        item = _clean(chunk, 260)
        if len(item) < 8:
            continue
        if not item.endswith("."):
            item += "."
        steps.append(item)
        if len(steps) >= 5:
            break

    if len(steps) == 1:
        steps.append("Confirm whether the issue is resolved after completing the step.")
    return steps[:5]


def _confidence_from_row(row: pd.Series) -> float:
    status = str(row.get("status", "")).lower()
    quality = pd.to_numeric(row.get("quality_score", 0.7), errors="coerce")
    quality_f = float(quality) if pd.notna(quality) else 0.7
    base = 0.72 if status in {"resolved", "closed"} else 0.58
    return round(max(0.35, min(0.94, base + 0.18 * quality_f)), 3)


def _make_example(row: pd.Series) -> dict[str, Any] | None:
    title = _clean(row.get("title_clean") or row.get("title"), 300)
    description = _clean(row.get("description_clean") or row.get("description"), 1500)
    resolution = _clean(row.get("resolution_clean") or row.get("resolution"), 1500)
    if len(title) < 5 or len(description) < 15 or len(resolution) < 8:
        return None

    steps = _split_resolution(resolution)
    if not steps:
        return None

    category = _clean(row.get("category"), 100)
    component = _clean(row.get("component"), 100)
    department = _clean(row.get("department") or row.get("mapped_department"), 120)
    severity = _clean(row.get("severity"), 80)

    user_prompt = (
        f"User issue: {title}\n\n"
        f"Details: {description}\n\n"
        f"Ticket context: category={category}; component={component}; "
        f"department={department}; severity={severity}."
    )

    confidence = _confidence_from_row(row)
    escalation = confidence < 0.55 or str(row.get("severity", "")).lower() == "critical"
    reply = (
        "I found a similar historical support case. "
        "Let's try the safest documented fix first and then confirm whether it worked."
    )
    if escalation:
        reply = (
            "This may need specialist review, but we can try the safest documented checks first."
        )

    assistant = {
        "reply": reply,
        "recommended_steps": steps,
        "confidence": confidence,
        "escalation_recommended": escalation,
        "escalation_reason": (
            "Critical or lower-confidence issue; route to the specialist team if the steps do not resolve it."
            if escalation else
            "Historical resolved ticket provides a relevant Tier-1 fix."
        ),
    }

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=True)},
        ],
        "metadata": {
            "ticket_id": str(row.get("ticket_id", "")),
            "unified_id": str(row.get("unified_id", "")),
            "category": category,
            "component": component,
            "department": department,
            "source": str(row.get("source", "")),
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ITS LoRA chat dataset")
    parser.add_argument("--tickets-path", default=str(_default_tickets_path()))
    parser.add_argument("--output-dir", default=str(LORA_DIR))
    parser.add_argument("--max-examples", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.08)
    parser.add_argument("--test-ratio", type=float, default=0.08)
    parser.add_argument("--english-only", action="store_true", default=True)
    args = parser.parse_args()

    tickets_path = Path(args.tickets_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(tickets_path, low_memory=False).fillna("")
    if args.english_only and "language" in df.columns:
        df = df[df["language"].astype(str).str.lower().isin(["", "en"])]

    if "resolution" in df.columns:
        df = df[df["resolution"].astype(str).str.strip().str.len() > 8]
    elif "resolution_clean" in df.columns:
        df = df[df["resolution_clean"].astype(str).str.strip().str.len() > 8]

    examples = []
    for _, row in df.iterrows():
        ex = _make_example(row)
        if ex:
            examples.append(ex)

    random.seed(args.seed)
    random.shuffle(examples)
    if args.max_examples > 0:
        examples = examples[: args.max_examples]

    n = len(examples)
    n_test = int(n * args.test_ratio)
    n_val = int(n * args.val_ratio)
    test = examples[:n_test]
    val = examples[n_test:n_test + n_val]
    train = examples[n_test + n_val:]

    _write_jsonl(out_dir / "its_lora_train.jsonl", train)
    _write_jsonl(out_dir / "its_lora_val.jsonl", val)
    _write_jsonl(out_dir / "its_lora_test.jsonl", test)

    manifest = {
        "tickets_path": str(tickets_path),
        "total_source_rows_after_filters": int(len(df)),
        "examples_total": n,
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "system_prompt": SYSTEM_PROMPT,
        "purpose": "Refine support conversation tone, step formatting, confidence, and escalation phrasing.",
        "do_not_use_for": "Memorizing ticket facts. Ticket knowledge remains in RAG.",
    }
    (out_dir / "lora_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
