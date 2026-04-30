# Phase 6 LoRA Runbook

Use this phase only after Phase 5 evaluation identifies a clear LLM behavior gap.
LoRA is for improving conversation style, JSON structure, recommended-step formatting, and escalation phrasing. It is not for memorizing historical tickets. Ticket knowledge stays in RAG.

## 1. Install Optional Training Dependencies

Run on the AWS GPU training instance:

```bash
source venv/bin/activate
pip install -r requirements-lora.txt
```

## 2. Prepare Training Data

After preprocessing the 80k dataset:

```bash
python 12_prepare_lora_dataset.py \
  --tickets-path data/processed/all_tickets.csv \
  --max-examples 12000
```

If `all_tickets.csv` is not built yet, use the major dataset directly:

```bash
python 12_prepare_lora_dataset.py \
  --tickets-path data/processed/its_tickets_80k.csv \
  --max-examples 12000
```

Outputs:

```text
data/lora/its_lora_train.jsonl
data/lora/its_lora_val.jsonl
data/lora/its_lora_test.jsonl
data/lora/lora_dataset_manifest.json
```

## 3. Train LoRA Adapter

Recommended first run:

```bash
python 13_train_lora.py \
  --model-name Qwen/Qwen3-4B-Instruct \
  --train-jsonl data/lora/its_lora_train.jsonl \
  --val-jsonl data/lora/its_lora_val.jsonl \
  --output-dir models/lora/its-qwen3-helpdesk-lora \
  --load-in-4bit \
  --epochs 2 \
  --batch-size 2 \
  --grad-accum 8
```

The script saves only the adapter:

```text
models/lora/its-qwen3-helpdesk-lora/
```

## 4. Evaluate Base vs LoRA

```bash
python 14_evaluate_lora.py \
  --model-name Qwen/Qwen3-4B-Instruct \
  --adapter-dir models/lora/its-qwen3-helpdesk-lora \
  --test-jsonl data/lora/its_lora_test.jsonl \
  --max-cases 100 \
  --load-in-4bit
```

Outputs:

```text
evaluation/phase6_lora_comparison.json
evaluation/phase6_lora_comparison.csv
```

## 5. Decision Rule

Use the LoRA adapter only if it improves:

- JSON validity rate
- average contract score
- recommended-step consistency
- escalation phrasing

Do not use it if latency increases too much or if it becomes less grounded than the base model.

## 6. Production Note

Do not switch the live backend to LoRA automatically. First compare:

```text
base Qwen3:4B + RAG
vs
LoRA Qwen3:4B + RAG
```

Only promote the adapter after Phase 6 evaluation shows a measurable improvement.
