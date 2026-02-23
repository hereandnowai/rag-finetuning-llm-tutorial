# Comprehensive Professor's Study Guide: RAG vs. Fine-Tuning

This guide is designed for anyone who is going to be inteviewed on RAG and Fine-tuning. It covers conceptual, technical, and implementation-specific details based on the scripts in this project.

---

## Part 1: High-Level AI Architecture

### Q1: What is the "knowledge cutoff" problem, and how does this project address it?
**Answer:** Standard LLMs are frozen in time based on their training data. If you ask a base model about the "Economic Survey 2025", it will fail. This project demonstrates the two industry-standard solutions:
1. **Fine-Tuning:** Updating the model's weights with the 2025 data (Activities 2 & 6).
2. **RAG:** Providing the model with a "searchable" library of the 2025 document (Activity 3).

### Q2: Compare the "In-context Learning" (RAG) vs. "Weight-based Learning" (Fine-tuning).
**Answer:** 
- **RAG** uses **In-context Learning**: The knowledge is provided temporarily in the prompt. It’s like a person reading a sticky note.
- **Fine-tuning** uses **Weight-based Learning**: The knowledge is baked into the neural network's parameters. It's like a person actually learning a new language.

---

## Part 2: Data Engineering ([activity1_prepare_domain_dataset.py](activity1_prepare_domain_dataset.py))

### Q3: Explain the logic behind your `chunk_text` function. Why 1000 characters with 100 overlap?
**Answer:** Smaller chunks (1000 chars) prevent exceeding the LLM's context window and ensure we only retrieve relevant facts. The **overlap (100 chars)** is critical for "Semantic Continuity"—it prevents a sentence from being cut in half, which would destroy the embedding's meaning.

### Q4: How did you generate the "Synthetic QA" pairs?
**Answer:** In `activity1`, we use a **heuristic-based template approach**. We extract factual sentences using Regex (searching for years like '2025') and wrap them in a static question template. This keeps the demo fast and offline.

In a **production setting**, we would use a high-reasoning **"Teacher Model"** (such as GPT-5+, Gemini 3+, or Claude Opus 4+) to perform **LLM-based Synthetic Data Generation**. This allows for much more complex, multi-hop questions and logically challenging answers that avoid simple keywords.

---

## Part 3: Deep-Dive into Fine-Tuning ([activity2_finetune_no_rag.py](activity2_finetune_no_rag.py))

### Q5: Why `distilgpt2` and not a BERT-style model?
**Answer:** BERT is an **"Encoder-only"** model, great for classification but bad at speaking. `distilgpt2` is a **"Decoder-only"** (Causal) model. Since our goal is to generate *answers* to questions, we need a Causal LLM.

### Q6: What happens during `trainer.train()`? 
**Answer:** The model performs "Backpropagation." It compares its predicted answer to the `gold_answer` in our dataset, calculates the **Cross-Entropy Loss**, and uses an **Optimizer (AdamW)** to adjust the model's weights to minimize that loss.

### Q7: Explain the `DataCollatorForLanguageModeling`.
**Answer:** In causal modeling, we aren't just predicting a label. We are predicting the *next token*. The collator handles "masking" and ensures the input sequences are padded correctly so the GPU can process them in batches.

---

## Part 4: Retrieval-Augmented Generation ([activity3_rag_no_finetune.py](activity3_rag_no_finetune.py))

### Q8: Walk me through the math of `cosine_similarity`.
**Answer:** 
$$ \text{similarity} = \frac{A \cdot B}{\|A\| \|B\|} $$
It measures the cosine of the angle between two vectors. If the angle is 0, the vectors are identical. In our code, we use `np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))` to find which PDF chunk is most "mathematically similar" to the user's question.

### Q9: Why use `embeddinggemma`?
**Answer:** Gemma embeddings are modern, lightweight, and specifically tuned for retrieval tasks. Using a different model for embeddings vs. chat (gpt-oss) shows a **"Modular RAG Strategy"** where you pick the best tool for each sub-task.

---

## Part 5: The Hybrid System ([activity4_hybrid_finetune_rag.py](activity4_hybrid_finetune_rag.py))

### Q10: Why combine the two? Isn't that redundant?
**Answer:** No. 
1. **Fine-tuning** gives the model "Domain Fluency" (understanding the language and format of Economic Surveys).
2. **RAG** gives "Factual Grounding" (ensuring the specific numbers in the answer are correct).
A hybrid model is less likely to hallucinate and more likely to sound like a professional economist.

---

## Part 6: Evaluation & Mastery ([activity5_compare_systems.py](activity5_compare_systems.py))

### Q11: How do you programmatically detect a Hallucination?
**Answer:** In Activity 5, we use a **Lexical Overlap Heuristic**. If the LLM's answer uses 75% words that do NOT appear in the source corpus, we flag it as a potential hallucination. In a research paper, we would use **"NLI (Natural Language Inference)"** models to check if the answer is "entailed" by the context.

### Q12: Explain the "Catastrophic Forgetting" risk in fine-tuning.
**Answer:** If we fine-tuned `distilgpt2` too aggressively (too many epochs or high learning rate) on only Economic data, it might "forget" how to speak English generally or perform basic logic. This is why we keep the learning rate low (5e-5) in `activity2`.

---

## Part 7: Teacher/Professor Tips for IIT Gandhinagar

*   **Mention "PEFT/LoRA":** If they ask about scaling, say: *"For larger models like Llama-3, I would use LoRA (Low-Rank Adaptation) to only train 1% of the weights, which is much faster."*
*   **Mention "Vector Databases":** Say: *"In this demo, I use a simple NumPy array for research clarity. In production, I would use ChromaDB or FAISS to handle millions of documents."*
*   **Mention "System Prompts":** Point out how you used role-play in Activity 3: *"You are a factual assistant for India's Economic Survey."* This is called **"Persona Grounding."**

---
## Part 8: Infrastructure & Deployment (The "How-to-Run" Questions)

### Q18: Why did you choose Ollama for this project instead of running models directly via PyTorch/HuggingFace?
**Answer:** "Ollama provides **Inference Optimization** (like 4-bit quantization and GGUF formatting) out of the box. Running a 20B parameter model (`gpt-oss`) raw in PyTorch would require ~40GB of VRAM. Ollama manages memory much more efficiently, allowing a high-tier model to run on consumer hardware, which is vital for student accessibility."

### Q19: Explain the `KMP_DUPLICATE_LIB_OK` environment variable.
**Answer:** "On macOS systems (Intel or M-series), both NumPy and PyTorch might load their own version of the `libiomp5` library (OpenMP). This causes a 'multiple instances' crash. Setting the variable to `TRUE` tells the system to allow the duplication, ensuring the teaching demo doesn't crash during live training."

### Q20: How would you scale this to 1,000 students running it simultaneously?
**Answer:** "I would transition from the local scripts to a **Containerized Architecture**. I'd host an **Ollama Server** on an IIT Gandhinagar cluster with a GPU, and have the student scripts point to that URL via `OLLAMA_BASE_URL`. For the code, I would move the Vector Search from NumPy to a production-grade database like **Qdrant** or **Milvus**."

### Q21: Can I use a model downloaded via Ollama for fine-tuning?
**Answer:** "Generally, **no**. Ollama is optimized for **inference** and usually stores models as quantized blobs (GGUF). Fine-tuning requires the model in its 'raw' form (Safetensors/PyTorch) so that we can calculate gradients and update weights. While libraries like `unsloth` are narrowing this gap, the industry standard remains using HuggingFace weights for training and Ollama/vLLM for deployment."

---
© 2026 - IIT Gandhinagar Demo Preparation Guide.

## Appendix: "Master the Lines" Technical Quiz

### Q13: What does `model.resize_token_embeddings(len(tokenizer))` do?
**Answer:** If we add a special padding token (like we do in Activity 2), the model’s internal vocabulary size changes. This line ensures the neural network layer (the embedding layer) matches the size of the tokenizer so it doesn't crash when it sees the new token ID.

### Q14: Explain `Dataset.from_dict({"text": texts}).map(tokenize, batched=True)`.
**Answer:** The `.map()` function is the engine of the `datasets` library. It applies our `tokenize` function to the entire dataset. `batched=True` is the performance optimization—it processes groups of sentences at once (using multi-threading if available), which is significantly faster than one-by-one.

### Q15: In `activity1`, what is the purpose of `re.sub(r"\s+", " ", text)`?
**Answer:** PDF extraction often results in weird artifacts like triple spaces, tabs, or newlines in the middle of sentences. This regular expression "normalizes" the text by collapsing all whitespace into a single space, which makes the embeddings much more accurate.

### Q16: In Activity 3, why add `1e-12` to the denominator in the cosine function?
**Answer:** Code: `np.linalg.norm(a) * np.linalg.norm(b) + 1e-12`. This is a "Numerical Stability" trick. If a vector happens to be all zeros (empty content), its norm is 0. Dividing by 0 would crash the script. `1e-12` is a tiny "epsilon" that prevents division by zero without changing the result.

### Q17: What is the `report_to="none"` argument in `TrainingArguments`?
**Answer:** By default, HuggingFace tries to log your training to "Weights & Biases" or "MLFlow." In a classroom setting or on a local machine without internet configs, this would cause an error or a long wait time. `none` keeps the demo focused on local results.
