"""Activity 6 (Use Case 2): Fine-tune on selected PDF pages only (No RAG).

Goal: model answers factual questions about selected section/pages of the PDF
without retrieval, tools, or internet search.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from pypdf import PdfReader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_pdf_text(pdf_path: Path, start_page: int, end_page: int) -> str:
    reader = PdfReader(str(pdf_path))
    page_count = len(reader.pages)

    start_idx = max(0, start_page - 1)
    end_idx = min(page_count, end_page)

    text_parts: List[str] = []
    for i in range(start_idx, end_idx):
        text = reader.pages[i].extract_text() or ""
        text_parts.append(normalize(text))

    return "\n".join(text_parts)


def build_qa_from_text(text: str, max_pairs: int = 80) -> List[Dict[str, str]]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 50]
    rows: List[Dict[str, str]] = []

    for i, sent in enumerate(sentences[:max_pairs]):
        q = f"State one factual point from selected Economic Survey passage {i + 1}."
        rows.append({"question": q, "answer": sent})

    return rows


def train(rows: List[Dict[str, str]], out_dir: Path, base_model: str, epochs: int) -> None:
    train_texts = [f"Question: {r['question']}\nAnswer: {r['answer']}" for r in rows]
    dataset = Dataset.from_dict({"text": train_texts})

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        save_total_limit=1,
        logging_steps=5,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))


def answer_question(model_dir: Path, question: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=90,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return decoded.split("Answer:", 1)[-1].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use Case 2: Fine-tune on selected PDF pages, no RAG.")
    parser.add_argument("--pdf", default="economic survey of India 2025-26.pdf")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=20)
    parser.add_argument("--base-model", default="distilgpt2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output", default="models/usecase2_ft_only")
    parser.add_argument("--ask", default="")
    parser.add_argument("--dump-train", default="data/usecase2_train.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    text = extract_pdf_text(pdf_path, args.start_page, args.end_page)
    rows = build_qa_from_text(text)

    if not rows:
        raise RuntimeError("No training rows created. Try a different page range.")

    dump_path = Path(args.dump_train)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    with dump_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    train(rows, out_dir, args.base_model, args.epochs)

    print(f"Use Case 2 fine-tuning complete. Model: {out_dir}")
    print(f"Training rows dumped to: {dump_path}")

    if args.ask:
        print("\nQuestion:", args.ask)
        print("Answer:", answer_question(out_dir, args.ask))


if __name__ == "__main__":
    main()
