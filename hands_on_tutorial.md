# Hands-On Tutorial: Fine-Tuning, RAG, and Hybrid on Economic Survey PDF

## 1. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ollama setup for RAG:

```bash
cp .env.example .env
ollama pull gpt-oss:20b
ollama pull embeddinggemma
```

## 2. Build Shared Dataset (Use Case 1 foundation)

```bash
python activity1_prepare_domain_dataset.py --start-page 1 --end-page 30
```

This generates:
- `data/domain_corpus.txt`
- `data/corpus_chunks.json`
- `data/qa_train.jsonl`
- `data/eval_questions.json`

## 3. System 1: Fine-Tuned LLM (No RAG)

```bash
export KMP_DUPLICATE_LIB_OK=TRUE && python activity2_finetune_no_rag.py --epochs 3
python activity2_finetune_no_rag.py --ask "What is one key factual statement from Economic Survey chunk 2?"
```

## 4. System 2: RAG (No Fine-Tuning)

```bash
python activity3_rag_no_finetune.py --question "What does the survey mention about inflation trends?"
```

## 5. System 3: Hybrid (Fine-Tuned + RAG)

```bash
python activity4_hybrid_finetune_rag.py --question "What does the survey mention about GDP growth outlook?"
```

## 6. Structured Comparison

```bash
python activity5_compare_systems.py
```

Generated report files:
- `outputs/comparison_metrics.csv`
- `outputs/comparison_metrics.md`

## 7. Use Case 2: Fine-Tune on PDF Only (No RAG, No Search)

```bash
python activity6_usecase2_pdf_finetune_only.py --start-page 1 --end-page 20 --epochs 3
python activity6_usecase2_pdf_finetune_only.py --ask "State one factual point from selected Economic Survey passage 5."
```

## 8. Classroom Demo Tips
- Keep page range small (for faster demo).
- Run 1 question across all systems and compare side by side.
- Explain trade-offs using generated metrics file.
