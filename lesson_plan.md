# Lesson Plan: Fine-Tuning vs RAG vs Hybrid (Beginner Hands-On)

## Target Audience
- Beginners in LLM engineering
- Basic Python familiarity

## Total Duration
- 1 day intensive (5.5 to 6 hours)

## Learning Outcomes
By the end, learners will be able to:
- Build a fine-tuned LLM without RAG.
- Build a RAG pipeline without fine-tuning.
- Build a hybrid (fine-tuned + retrieval) system.
- Compare systems on accuracy, hallucination, latency, cost, maintainability, and scalability.
- Fine-tune on selected pages of a real-world PDF and answer factual questions without retrieval.

## Prerequisites
- Python 3.10+
- Terminal basics
- Ollama installed locally for RAG activity
- Pulled models: `gpt-oss:20b` and `embeddinggemma`

## Setup (20 mins)
1. Create virtual environment:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Optional API setup for RAG:
   - Copy `.env.example` to `.env`
   - Fill `OPENAI_API_KEY`

## Activity Flow

### Activity 1 (45 mins): Dataset Preparation from PDF
- File: `activity1_prepare_domain_dataset.py`
- Goal: Convert selected pages into corpus, chunks, QA train set, and evaluation set.
- Run:
  - `python activity1_prepare_domain_dataset.py --start-page 1 --end-page 30`

### Activity 2 (60 mins): Fine-Tuned LLM (No RAG)
- File: `activity2_finetune_no_rag.py`
- Goal: Train a small local model (`google/gemma-3-270m`) on generated QA pairs.
- Run:
  - `export KMP_DUPLICATE_LIB_OK=TRUE && python activity2_finetune_no_rag.py --epochs 3`
- Sample query:
  - `python activity2_finetune_no_rag.py --ask "What is one key factual statement from Economic Survey chunk 2?"`

### Activity 3 (45 mins): RAG (No Fine-Tuning)
- File: `activity3_rag_no_finetune.py`
- Goal: Retrieve context from chunks and answer with an LLM.
- Run:
  - `python activity3_rag_no_finetune.py --question "What does the survey say about inflation trends?"`

### Activity 4 (45 mins): Hybrid (Fine-Tuned + Retrieval)
- File: `activity4_hybrid_finetune_rag.py`
- Goal: Add retrieval context into the fine-tuned model prompt.
- Run:
  - `python activity4_hybrid_finetune_rag.py --question "What does the survey mention about GDP growth outlook?"`

### Activity 5 (50 mins): Structured Comparison
- File: `activity5_compare_systems.py`
- Goal: Evaluate all systems with consistent metrics.
- Run:
  - `python activity5_compare_systems.py`
- Output:
  - `outputs/comparison_metrics.csv`
  - `outputs/comparison_metrics.md`

### Activity 6 (Use Case 2) (50 mins): PDF-Only Fine-Tuning (No RAG)
- File: `activity6_usecase2_pdf_finetune_only.py`
- Goal: Fine-tune on selected pages and answer factual questions without retrieval.
- Run:
  - `python activity6_usecase2_pdf_finetune_only.py --start-page 1 --end-page 20 --epochs 3`
- Ask question:
  - `python activity6_usecase2_pdf_finetune_only.py --ask "State one factual point from selected Economic Survey passage 5."`

## Teaching Notes
- Explain the conceptual difference first:
  - Fine-tuning changes model weights.
  - RAG keeps model fixed and adds retrieval-time context.
  - Hybrid combines both.
- Keep expectations realistic for tiny local models.
- Emphasize that evaluation must use the same question set across systems.

## Assessment Rubric for Students
- Correct setup and execution: 20%
- Working outputs from all 3 systems: 30%
- Proper comparison and interpretation: 30%
- Explanation of trade-offs and limitations: 20%

## Demo Storyline for IIT Gandhinagar Teaching Selection
1. Start with a simple real-world question from Economic Survey.
2. Show each system's answer side by side.
3. Present metrics table and discuss practical trade-offs.
4. Conclude with when to choose each approach in production.

## Expected Challenges and Workarounds
- Slow fine-tuning on CPU:
  - Use fewer pages, fewer epochs, or small subset.
- Weak answer quality:
  - Improve QA generation quality and chunking.
- Missing API key for RAG:
  - Skip Activity 3 and compare Fine-Tuned vs Hybrid local setup.
