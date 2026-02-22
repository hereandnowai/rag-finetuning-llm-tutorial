"""Activity 4: Hybrid system (Fine-Tuned + RAG).

Retrieves top chunks using TF-IDF and feeds them to a fine-tuned local model.
No external API is required for this activity.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from ollama import Client
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_chunks(chunks_path: Path) -> List[str]:
    return json.loads(chunks_path.read_text(encoding="utf-8"))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def retrieve(client: Client, chunks: List[str], question: str, k: int = 4) -> List[str]:
    # Use Ollama for embeddings
    res = client.embed(model="embeddinggemma", input=[question])
    q_vec = np.array(res["embeddings"][0], dtype=np.float32)

    # In a real app we'd pre-calculate these, but for simplicity here we do it on flight
    # or just use the first few if we want to be ultra fast. 
    # Let's do it properly.
    res_corpus = client.embed(model="embeddinggemma", input=chunks)
    corpus_vecs = np.array(res_corpus["embeddings"], dtype=np.float32)

    scores = [cosine(v, q_vec) for v in corpus_vecs]
    top_indices = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top_indices]


def generate_with_context(model_dir: Path, question: str, context_chunks: List[str], max_new_tokens: int = 120) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    context = "\n\n".join(context_chunks)
    prompt = (
        "You are answering from India's Economic Survey. Use context facts only.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)

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
    parser = argparse.ArgumentParser(description="Run Hybrid (fine-tuned + retrieval) system.")
    parser.add_argument("--model", default="models/ft_no_rag")
    parser.add_argument("--chunks", default="data/corpus_chunks.json")
    parser.add_argument("--question", required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_dir = Path(args.model)
    chunks_path = Path(args.chunks)

    if not model_dir.exists():
        raise FileNotFoundError(f"Fine-tuned model not found: {model_dir}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    client = Client(host=args.ollama_url)
    chunks = load_chunks(chunks_path)
    
    print(f"Retrieving using Ollama embeddinggemma...")
    top_chunks = retrieve(client, chunks, args.question, k=args.k)
    
    print(f"Generating using fine-tuned model at {model_dir}...")
    answer = generate_with_context(model_dir, args.question, top_chunks)

    print("\n" + "="*50)
    print("Question:", args.question)
    print("Answer:", answer)
    print("="*50)


if __name__ == "__main__":
    main()
