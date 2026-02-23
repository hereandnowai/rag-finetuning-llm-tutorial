"""Activity 2: Fine-tuned model (No RAG).

Trains a small local model on QA pairs and answers questions without retrieval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


PROMPT_TEMPLATE = "<start_of_turn>user\nQuestion: {question}<end_of_turn>\n<start_of_turn>model\nAnswer: {answer}<end_of_turn>"


def load_qa_jsonl(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_dataset(rows: List[Dict[str, str]]) -> Dataset:
    texts = [PROMPT_TEMPLATE.format(question=r["question"], answer=r["answer"]) for r in rows]
    return Dataset.from_dict({"text": texts})


def train_model(data_path: Path, output_dir: Path, base_model: str, epochs: int, batch_size: int) -> None:
    rows = load_qa_jsonl(data_path)
    dataset = build_dataset(rows)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Gemma tokenizers often need the pad_token set to something specific or eos
    tokenizer.padding_side = "right" 

    def tokenize(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        return tokenizer(batch["text"], truncation=True, max_length=512)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16 if torch.backends.mps.is_available() else torch.float32)
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_total_limit=1,
        logging_steps=5,
        learning_rate=2e-4, # Gemma often likes slightly higher LR than GPT2
        use_mps_device=torch.backends.mps.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def ask_model(model_dir: Path, question: str, max_new_tokens: int = 80) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("Answer:", 1)[-1].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a small LLM without retrieval.")
    parser.add_argument("--data", default="data/qa_train.jsonl")
    parser.add_argument("--output", default="models/ft_no_rag")
    parser.add_argument("--base-model", default="google/gemma-3-270m")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ask", default="", help="Optional single question after training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    train_model(data_path, output_dir, args.base_model, args.epochs, args.batch_size)
    print(f"Fine-tuning complete. Model saved at: {output_dir}")

    if args.ask:
        answer = ask_model(output_dir, args.ask)
        print("\nSample answer:")
        print(answer)


if __name__ == "__main__":
    main()
