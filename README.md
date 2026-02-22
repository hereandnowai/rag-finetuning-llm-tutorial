# RAG vs Fine-tuning vs Hybrid: A Hands-on Tutorial

> **AI is Good**
> Developed by [HERE AND NOW AI](https://hereandnowai.com)

This tutorial provides a practical comparison between three common LLM architectures using India's Economic Survey as the domain knowledge.

## Prerequisites
- **Python 3.11**
- **Ollama** installed and running.
- Pull the required models:
  ```bash
  ollama pull gpt-oss:20b
  ollama pull embeddinggemma
  ```

## Setup
1. Create and activate virtual environment:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Tutorial Activities

### Activity 1: Data Preparation
Extracts text from the PDF, chunks it for retrieval, and generates Synthetic QA pairs for fine-tuning.
```bash
python activity1_prepare_domain_dataset.py --pdf data/economic_survey_sample.pdf
```

### Activity 2: Fine-tuning (Baseline)
Trains a small local model (`distilgpt2`) on the generated QA pairs. This model tries to "memorize" the facts.
```bash
python activity2_finetune_no_rag.py --epochs 3
```

### Activity 3: RAG (Retrieval-Augmented Generation)
Uses a large open-source model (`gpt-oss:20b`) and retrieves context dynamically from the PDF using `embeddinggemma`.
```bash
python activity3_rag_no_finetune.py --question "What is the projected GDP growth for FY25?"
```

### Activity 4: Hybrid System
Combines the Fine-tuned model from Activity 2 with the Retrieval system from Activity 3.
```bash
python activity4_hybrid_finetune_rag.py --question "Mention three key infrastructure indicators."
```

### Activity 5: Systematic Comparison
Runs an evaluation across all three systems and generates a performance report.
```bash
python activity5_compare_systems.py
```

## Results
After running Activity 5, check `outputs/comparison_metrics.md` for a detailed table comparing Accuracy, Hallucination, Latency, and Cost.
