"""
ITS RAG - Phase 6: Train LoRA Adapter

Run only after Phase 5 evaluation and after creating JSONL data with:
  python 12_prepare_lora_dataset.py

Example AWS GPU run:
  python 13_train_lora.py \
    --model-name Qwen/Qwen3-4B-Instruct \
    --train-jsonl data/lora/its_lora_train.jsonl \
    --val-jsonl data/lora/its_lora_val.jsonl \
    --output-dir models/lora/its-qwen3-4b-helpdesk-lora \
    --load-in-4bit

Notes:
  - This script requires optional dependencies from requirements-lora.txt.
  - It saves only the LoRA adapter, not a full merged model.
  - RAG remains the source of ticket knowledge; LoRA refines behavior/tone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _format_messages(tokenizer, example: dict) -> str:
    messages = example["messages"]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    rendered = []
    for msg in messages:
        rendered.append(f"{msg['role'].upper()}: {msg['content']}")
    return "\n".join(rendered)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ITS helpdesk LoRA adapter")
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B-Instruct")
    parser.add_argument("--train-jsonl", default="data/lora/its_lora_train.jsonl")
    parser.add_argument("--val-jsonl", default="data/lora/its_lora_val.jsonl")
    parser.add_argument("--output-dir", default="models/lora/its-qwen3-helpdesk-lora")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=True)
    args = parser.parse_args()

    # Optional heavy imports are inside main so normal project usage does not
    # require training dependencies.
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="bfloat16",
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quantization_config,
    )
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    data_files = {"train": args.train_jsonl, "validation": args.val_jsonl}
    dataset = load_dataset("json", data_files=data_files)

    def to_text(example: dict) -> dict:
        return {"text": _format_messages(tokenizer, example)}

    dataset = dataset.map(to_text, remove_columns=dataset["train"].column_names)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        bf16=args.bf16,
        fp16=not args.bf16,
        optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        peft_config=peft_config,
        args=train_args,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    manifest = {
        "model_name": args.model_name,
        "adapter_dir": str(output_dir),
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "load_in_4bit": args.load_in_4bit,
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
