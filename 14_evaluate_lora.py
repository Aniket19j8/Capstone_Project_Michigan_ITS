"""
ITS RAG - Phase 6: Evaluate Base Model vs LoRA Adapter

Run after:
  1. Phase 5 evaluation
  2. 12_prepare_lora_dataset.py
  3. 13_train_lora.py

Example:
  python 14_evaluate_lora.py \
    --model-name Qwen/Qwen3-4B-Instruct \
    --adapter-dir models/lora/its-qwen3-helpdesk-lora \
    --test-jsonl data/lora/its_lora_test.jsonl \
    --max-cases 100 \
    --load-in-4bit

Outputs:
  evaluation/phase6_lora_comparison.json
  evaluation/phase6_lora_comparison.csv
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd


EVAL_DIR = Path("./evaluation")
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def _load_jsonl(path: Path, max_cases: int) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_cases > 0 and len(rows) >= max_cases:
                break
    return rows


def _strip_json(text: str) -> dict[str, Any] | None:
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.I)
    m = re.search(r"\{.*\}", clean, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _score_response(parsed: dict[str, Any] | None) -> dict[str, Any]:
    if not parsed:
        return {
            "json_valid": False,
            "has_reply": False,
            "step_count": 0,
            "confidence_valid": False,
            "escalation_valid": False,
            "contract_score": 0.0,
        }

    steps = parsed.get("recommended_steps")
    conf = parsed.get("confidence")
    esc = parsed.get("escalation_recommended")
    json_valid = True
    has_reply = isinstance(parsed.get("reply"), str) and len(parsed.get("reply", "").strip()) > 10
    step_count = len(steps) if isinstance(steps, list) else 0
    confidence_valid = isinstance(conf, (int, float)) and 0 <= float(conf) <= 1
    escalation_valid = isinstance(esc, bool)
    checks = [
        json_valid,
        has_reply,
        2 <= step_count <= 5,
        confidence_valid,
        escalation_valid,
        isinstance(parsed.get("escalation_reason"), str),
    ]
    return {
        "json_valid": json_valid,
        "has_reply": has_reply,
        "step_count": step_count,
        "confidence_valid": confidence_valid,
        "escalation_valid": escalation_valid,
        "contract_score": round(sum(bool(x) for x in checks) / len(checks), 4),
    }


def _prompt_from_example(tokenizer, example: dict[str, Any]) -> str:
    messages = example["messages"][:-1]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\nASSISTANT:"


def _generate(model, tokenizer, prompt: str, max_new_tokens: int) -> tuple[str, float]:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = round((time.time() - t0) * 1000, 1)
    generated = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True), latency_ms


def _load_model(model_name: str, adapter_dir: str | None, load_in_4bit: bool):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir or model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="bfloat16",
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quantization_config,
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base model vs ITS LoRA adapter")
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B-Instruct")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--test-jsonl", default="data/lora/its_lora_test.jsonl")
    parser.add_argument("--max-cases", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--skip-base", action="store_true", help="Evaluate only the LoRA adapter.")
    args = parser.parse_args()

    examples = _load_jsonl(Path(args.test_jsonl), args.max_cases)
    if not examples:
        raise ValueError(f"No test examples found at {args.test_jsonl}")

    rows = []
    models_to_run = []
    if not args.skip_base:
        models_to_run.append(("base", None))
    models_to_run.append(("lora", args.adapter_dir))

    for label, adapter in models_to_run:
        model, tokenizer = _load_model(args.model_name, adapter, args.load_in_4bit)
        for i, ex in enumerate(examples, start=1):
            prompt = _prompt_from_example(tokenizer, ex)
            text, latency_ms = _generate(model, tokenizer, prompt, args.max_new_tokens)
            parsed = _strip_json(text)
            scores = _score_response(parsed)
            rows.append({
                "model": label,
                "case_id": i,
                "latency_ms": latency_ms,
                "raw_output": text[:3000],
                "parsed_output": json.dumps(parsed or {}, ensure_ascii=True),
                **scores,
            })
        del model

    df = pd.DataFrame(rows)
    df.to_csv(EVAL_DIR / "phase6_lora_comparison.csv", index=False)

    summary = {}
    for model_name, group in df.groupby("model"):
        summary[model_name] = {
            "cases": int(len(group)),
            "json_valid_rate": round(float(group["json_valid"].mean()), 4),
            "avg_contract_score": round(float(group["contract_score"].mean()), 4),
            "avg_step_count": round(float(group["step_count"].mean()), 2),
            "avg_latency_ms": round(float(group["latency_ms"].mean()), 1),
            "p95_latency_ms": round(float(group["latency_ms"].quantile(0.95)), 1),
        }

    payload = {
        "model_name": args.model_name,
        "adapter_dir": args.adapter_dir,
        "test_jsonl": args.test_jsonl,
        "max_cases": args.max_cases,
        "summary": summary,
        "recommendation": (
            "Use the LoRA adapter only if it improves avg_contract_score and JSON validity without unacceptable latency."
        ),
    }
    (EVAL_DIR / "phase6_lora_comparison.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
