"""Activity 1: Prepare domain dataset from Economic Survey PDF.

Creates:
- data/domain_corpus.txt
- data/corpus_chunks.json
- data/qa_train.jsonl
- data/eval_questions.json

This script is intentionally simple and beginner-friendly.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader


@dataclass
class QAPair:
    question: str
    answer: str


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_text_from_pdf(pdf_path: Path, start_page: int, end_page: int) -> str:
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    # Convert user-friendly page numbers to zero-based indices.
    start_idx = max(0, start_page - 1)
    end_idx = min(total_pages, end_page)

    page_texts: List[str] = []
    for i in range(start_idx, end_idx):
        text = reader.pages[i].extract_text() or ""
        page_texts.append(normalize_whitespace(text))

    return "\n".join(page_texts)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 120) -> List[str]:
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(text_len, start + chunk_size)
        chunk = normalize_whitespace(text[start:end])
        if chunk:
            chunks.append(chunk)
        start = max(end - overlap, end)

    return chunks


def make_qa_pairs(chunks: List[str], max_pairs: int = 120) -> List[QAPair]:
    qa_pairs: List[QAPair] = []
    year_pattern = re.compile(r"(20\d{2}|19\d{2})")

    for idx, chunk in enumerate(chunks):
        if len(qa_pairs) >= max_pairs:
            break

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk) if len(s.strip()) > 40]
        if not sentences:
            continue

        answer = sentences[0]
        year_match = year_pattern.search(answer)

        if year_match:
            question = (
                f"In the Economic Survey section chunk {idx + 1}, what fact is stated around the year "
                f"{year_match.group(1)}?"
            )
        else:
            question = f"What is one key factual statement from Economic Survey chunk {idx + 1}?"

        qa_pairs.append(QAPair(question=question, answer=answer))

    return qa_pairs


def write_outputs(output_dir: Path, corpus: str, chunks: List[str], qa_pairs: List[QAPair]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "domain_corpus.txt").write_text(corpus, encoding="utf-8")
    (output_dir / "corpus_chunks.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")

    with (output_dir / "qa_train.jsonl").open("w", encoding="utf-8") as f:
        for row in qa_pairs:
            f.write(json.dumps({"question": row.question, "answer": row.answer}, ensure_ascii=True) + "\n")

    eval_rows = [{"question": row.question, "gold_answer": row.answer} for row in qa_pairs[:20]]
    (output_dir / "eval_questions.json").write_text(json.dumps(eval_rows, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare dataset from PDF for fine-tuning and RAG demos.")
    parser.add_argument("--pdf", default="economic survey of India 2025-26.pdf", help="Path to PDF file")
    parser.add_argument("--start-page", type=int, default=1, help="Start page number (1-based)")
    parser.add_argument("--end-page", type=int, default=30, help="End page number (1-based, inclusive)")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in characters")
    parser.add_argument("--max-pairs", type=int, default=120, help="Maximum QA pairs to generate")
    parser.add_argument("--output-dir", default="data", help="Directory to store generated artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    corpus = extract_text_from_pdf(pdf_path, args.start_page, args.end_page)
    chunks = chunk_text(corpus, chunk_size=args.chunk_size)
    qa_pairs = make_qa_pairs(chunks, max_pairs=args.max_pairs)

    if not qa_pairs:
        raise RuntimeError("No QA pairs generated. Try different pages or larger range.")

    write_outputs(Path(args.output_dir), corpus, chunks, qa_pairs)

    print("Dataset preparation complete.")
    print(f"Corpus chars: {len(corpus)}")
    print(f"Chunks: {len(chunks)}")
    print(f"QA pairs: {len(qa_pairs)}")


if __name__ == "__main__":
    main()
