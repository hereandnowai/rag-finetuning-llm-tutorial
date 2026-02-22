"""Activity 5: Structured comparison of three systems.

Compares:
1) Fine-tuned only (no RAG)
2) RAG only (no fine-tuning)
3) Hybrid (fine-tuned + retrieval)

Metrics:
- accuracy (exact match)
- hallucination rate (heuristic)
- latency
- cost estimate
- maintainability score
- scalability score
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Dict, List
import numpy as np
from ollama import Client

import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text)


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize(pred) == normalize(gold) else 0.0


def simple_hallucination(pred: str, corpus: str) -> float:
    pred_tokens = set(normalize(pred).split())
    corpus_tokens = set(normalize(corpus).split())
    if not pred_tokens:
        return 1.0
    overlap_ratio = len(pred_tokens & corpus_tokens) / max(1, len(pred_tokens))
    return 1.0 if overlap_ratio < 0.25 else 0.0


def load_eval(path: Path) -> List[Dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_chunks(path: Path) -> List[str]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_ft_model(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    return tokenizer, model


def answer_ft_only(tokenizer, model, question: str) -> str:
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=90,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("Answer:", 1)[-1].strip()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def embed_texts(client: Client, texts: List[str]) -> np.ndarray:
    res = client.embed(model="embeddinggemma", input=texts)
    return np.array(res["embeddings"], dtype=np.float32)


def answer_rag_only(client: Client, chunks: List[str], chunk_vecs: np.ndarray, question: str) -> str:
    q_vec = embed_texts(client, [question])[0]
    scores = [cosine(v, q_vec) for v in chunk_vecs]
    idx = np.argsort(scores)[::-1][:4]
    context = "\n\n".join(chunks[int(i)] for i in idx)
    prompt = (
        "Use only this context. If missing, say: I do not know from the selected pages.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}"
    )
    res = client.chat(model="gpt-oss:20b", messages=[{"role": "user", "content": prompt}])
    return res["message"]["content"]


def answer_hybrid(client: Client, tokenizer, model, chunks: List[str], chunk_vecs: np.ndarray, question: str) -> str:
    # Use same Ollama embeddings as RAG
    q_vec = embed_texts(client, [question])[0]
    scores = [cosine(v, q_vec) for v in chunk_vecs]
    idx = np.argsort(scores)[::-1][:4]
    context = "\n\n".join(chunks[int(i)] for i in idx)

    prompt = (
        "Answer using only the context facts.\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=90,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("Answer:", 1)[-1].strip()


def evaluate_system(
    name: str,
    answer_fn: Callable[[str], str],
    eval_rows: List[Dict[str, str]],
    corpus_text: str,
    cost_per_question: float,
    maintainability_score: int,
    scalability_score: int,
) -> Dict[str, float]:
    em_scores: List[float] = []
    hall_scores: List[float] = []
    latencies: List[float] = []

    for row in eval_rows:
        question = row["question"]
        gold = row["gold_answer"]

        t0 = time.perf_counter()
        pred = answer_fn(question)
        dt = time.perf_counter() - t0

        latencies.append(dt)
        em_scores.append(exact_match(pred, gold))
        hall_scores.append(simple_hallucination(pred, corpus_text))

    n = len(eval_rows)
    return {
        "system": name,
        "accuracy_exact_match": round(sum(em_scores) / n, 4),
        "hallucination_rate": round(sum(hall_scores) / n, 4),
        "avg_latency_sec": round(sum(latencies) / n, 3),
        "estimated_cost_usd": round(cost_per_question * n, 4),
        "maintainability_score_1_to_5": maintainability_score,
        "scalability_score_1_to_5": scalability_score,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare fine-tuned, RAG, and hybrid systems.")
    parser.add_argument("--eval", default="data/eval_questions.json")
    parser.add_argument("--chunks", default="data/corpus_chunks.json")
    parser.add_argument("--corpus", default="data/domain_corpus.txt")
    parser.add_argument("--model", default="models/ft_no_rag")
    parser.add_argument("--out", default="outputs/comparison_metrics.csv")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    eval_rows = load_eval(Path(args.eval))
    chunks = load_chunks(Path(args.chunks))
    corpus_text = Path(args.corpus).read_text(encoding="utf-8")
    client = Client(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    chunk_vecs = embed_texts(client, chunks)

    if not Path(args.model).exists():
        raise FileNotFoundError(
            f"Fine-tuned model not found at '{args.model}'. Run activity2_finetune_no_rag.py first."
        )

    tokenizer, model = load_ft_model(Path(args.model))

    ft_result = evaluate_system(
        name="Fine-Tuned (No RAG)",
        answer_fn=lambda q: answer_ft_only(tokenizer, model, q),
        eval_rows=eval_rows,
        corpus_text=corpus_text,
        cost_per_question=0.0005,
        maintainability_score=4,
        scalability_score=3,
    )

    hybrid_result = evaluate_system(
        name="Hybrid (Fine-Tuned + RAG)",
        answer_fn=lambda q: answer_hybrid(client, tokenizer, model, chunks, chunk_vecs, q),
        eval_rows=eval_rows,
        corpus_text=corpus_text,
        cost_per_question=0.0008,
        maintainability_score=3,
        scalability_score=4,
    )

    rag_result = evaluate_system(
        name="RAG (No Fine-Tune)",
        answer_fn=lambda q: answer_rag_only(client, chunks, chunk_vecs, q),
        eval_rows=eval_rows,
        corpus_text=corpus_text,
        cost_per_question=0.0002,
        maintainability_score=4,
        scalability_score=5,
    )

    all_results = [ft_result, rag_result, hybrid_result]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(out_path, index=False)

    md_out = out_path.with_suffix(".md")
    md_out.write_text(df.to_markdown(index=False), encoding="utf-8")

    print(f"Comparison saved: {out_path}")
    print(f"Comparison table: {md_out}")


if __name__ == "__main__":
    main()
