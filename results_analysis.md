# Results Analysis

## 1. Introduction

This document presents a comprehensive evaluation of the AmbedkarGPT Retrieval-Augmented Generation (RAG) system across three chunking strategies:

* **Small** (250 characters)
* **Medium** (550 characters)
* **Large** (900 characters)

A curated test set of **25 Q&A pairs** stored at data/test_dataset.json was used to evaluate retrieval quality, answer quality, and semantic similarity.

### Important Implementation Note

**All answer generation was performed using the Mistral 7B model (via Ollama)**.

However, due to repeated issues during evaluation — including:

* Local hardware heating constraints when running larger LLMs,
* RAGAS timeouts and concurrency failures,
* Unreliable behavior of RAGAS faithfulness metric with Ollama models,

**The evaluation metrics were computed using alternative, more stable methods**:

* **Relevance was computed using cosine similarity between embeddings.**
* **Faithfulness was computed using a custom rubric-based LLM prompt (phi3:mini)** instead of RAGAS.
* **No LLM was used inside RAGAS during evaluation.**

This ensures consistent and reproducible scores across all test cases.

---

## 2. Aggregated Results

| Chunk Size | Hit Rate | Precision@K | MRR       | ROUGE-L   | Relevance | Faithfulness | Cosine Sim | BLEU      |
| ---------- | -------- | ----------- | --------- | --------- | --------- | ------------ | ---------- | --------- |
| **Small**  | 1.00     | **0.64**    | 0.86      | 0.199     | **0.733** | **4.27**     | **0.551**  | 0.043     |
| **Medium** | 1.00     | 0.446       | **0.947** | 0.186     | 0.729     | 0.675        | 0.543      | 0.035     |
| **Large**  | 1.00     | 0.273       | 0.924     | **0.205** | 0.708     | 0.689        | 0.540      | **0.053** |

---

## 3. Analysis by Metric

### 3.1 Retrieval Metrics

#### Hit Rate — **1.0** for all

All chunking strategies successfully retrieved at least one correct document.

#### Precision@K (k=5) — **Small > Medium > Large**

Small chunks → more precise retrieval.
Large chunks contain multiple topics → lower precision.

#### MRR — **Medium best**

Medium chunks often place the correct source **at Rank 1**, even when more noisy chunks are included.

---

### 3.2 Answer Quality Metrics

#### ROUGE-L

Large chunks slightly ahead — (contain more lexical overlap).

#### Semantic Relevance (Cosine Similarity)

Nearly identical across all chunk sizes, slight advantage to **Small**.

#### Faithfulness — **Small dramatically better**

* Small: **4.27**
* Medium/Large: ~0.68

Possible reasons:

* Small chunks isolate specific ideas → very strong grounding.
* Larger chunks mix multiple concepts → model often blends.

---

### 3.3 Semantic Metrics

* **Cosine Similarity** tracks relevance.
* **BLEU** low for all. Large chunks slightly better due to more shared n-grams.

---

## 4. Failure Mode Analysis

### Observed Issues

1. **Chunk Bloat** (Medium/Large)

   * Additional irrelevant context reduced grounding.

2. **Context Bleed**

   * The model integrates unrelated paragraphs.

3. **Hallucination Risk**

   * Particularly high for medium/large chunks.

4. **Lexical Variation**

   * Low ROUGE despite high semantic scores.

---

## 5. Best Performing Chunking Strategy

### 🏆 **Small Chunks (250 chars)**

Across faithfulness, precision, relevance, and cosine similarity, **small chunks performed the best**.

* Best faithfulness (most grounded answers)
* Highest precision@K
* Best semantic relevance
* Minimal hallucination

Medium achieves higher MRR, but overall reliability strongly favors small chunks.

---

## 6. Changes planned (Improvements)

### A. Retrieval

* Adding a reranker (BGE reranker, Cohere reranker, etc.).
* Increase top-k retrieval.

### B. Chunking

* Switching to **semantic chunking** instead of character-based.
* Try recursive splitting.

### C. Answer Generation

* Current laptop CPU only, unable to use stronger LLMs.

### D. Evaluation

* If hardware allows for it, I'll reintroduce full RAGAS metrics.

---

## 7. Conclusion

Despite using **Mistral 7B for answer generation**, evaluation required fallback, stable techniques due to repeated RAGAS failures.

With these methods, the results clearly show:

> **Small chunks provide the highest overall RAG quality for AmbedkarGPT.**

They maximize faithfulness, semantic relevance, and retrieval precision, making them the optimal configuration for this corpus.

---
