"""Activity 3: Simple RAG (No fine-tuning) with Ollama only.

LLM: gpt-oss:20b
Embeddings: embeddinggemma
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List
import numpy as np
from ollama import Client

from dotenv import load_dotenv


def load_chunks(path: Path) -> List[str]:
    return json.loads(path.read_text(encoding="utf-8"))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def embed_texts(client: Client, model: str, texts: List[str]) -> np.ndarray:
    res = client.embed(model=model, input=texts)
    return np.array(res["embeddings"], dtype=np.float32)


def top_k_chunks(chunks: List[str], chunk_vecs: np.ndarray, q_vec: np.ndarray, k: int) -> List[str]:
    scores = [cosine(v, q_vec) for v in chunk_vecs]
    idx = np.argsort(scores)[::-1][:k]
    return [chunks[int(i)] for i in idx]


def answer_question(client: Client, model_name: str, question: str, context: str) -> str:
    prompt = (
        "You are a factual assistant for India's Economic Survey. "
        "Use only the context. If missing, say: I do not know from the selected pages.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}"
    )
    res = client.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    return res["message"]["content"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG (no fine-tuning) on prepared corpus chunks.")
    parser.add_argument("--chunks", default="data/corpus_chunks.json")
    parser.add_argument("--question", required=True)
    parser.add_argument("--k", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    chunks = load_chunks(chunks_path)
    client = Client(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    embed_model = "embeddinggemma"
    chat_model = "gpt-oss:20b"

    chunk_vecs = embed_texts(client, embed_model, chunks)
    q_vec = embed_texts(client, embed_model, [args.question])[0]
    context_chunks = top_k_chunks(chunks, chunk_vecs, q_vec, args.k)
    context = "\n\n".join(context_chunks)
    answer = answer_question(client, chat_model, args.question, context)

    print("Question:", args.question)
    print("Answer:", answer)


if __name__ == "__main__":
    main()
