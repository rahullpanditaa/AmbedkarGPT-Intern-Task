# AmbedkarGPT — RAG Evaluation Framework

This repository contains the full implementation of a Retrieval-Augmented Generation (RAG) evaluation system. It includes:

* A complete RAG pipeline using **LangChain**, **ChromaDB**, **HuggingFace Embeddings**, and **Ollama**
* Automated evaluation across multiple NLP metrics
* Comparative chunking analysis
* Aggregated performance reports
* A detailed results analysis document

---

# 📌 Project Structure

```
AmbedkarGPT-Intern-Task/
│
├── corpus/                      # Document corpus (Ambedkar writings)
├── data/
│   ├── test_dataset.json       # Provided test questions
│   ├── test_results.json       # Per-question evaluation results
│   └── aggregated_results.json # Metric averages per chunking strategy
│
├── lib/
│   ├── rag_chain.py            # RAG chain construction
│   ├── search/                 # Vector DB + retriever utilities
│   ├── metrics/                # Retrieval, answer-quality, semantic metrics
│   └── evaluation/             # Orchestration logic
│
├── results_analysis.md         # Detailed findings & recommendations
├── requirements.txt
└── main.py                     # Entry point to run complete evaluation
```

---

# ✨ Features

### **1. Retrieval-Augmented Generation (RAG)**

* Uses **Mistral 7B** (Ollama) for answer generation
* Embeddings from **all-MiniLM-L6-v2** via HuggingFace
* Document chunking with configurable sizes (small/medium/large)
* Vector store persisted with ChromaDB

### **2. Comprehensive Evaluation Metrics**

The system computes:

#### **Retrieval Metrics**

* Hit Rate
* Precision@K
* Mean Reciprocal Rank (MRR)

#### **Answer Quality Metrics**

> *Note:* Due to instability with RAGAS + local LLMs during testing (timeouts, heating), the following adjustments were made:
>
> * **Relevance** computed via cosine similarity instead of RAGAS
> * **Faithfulness** computed using a custom LLM rubric instead of RAGAS NLI module

* ROUGE-L
* Answer Relevance (Cosine similarity)
* Faithfulness (Custom LLM-scored 1–5 scale)

#### **Semantic Metrics**

* Cosine Similarity
* BLEU Score

---

# 🚀 How to Run

Make sure you have **Ollama** installed and the model downloaded:

```
ollama pull mistral
ollama pull phi3:mini
```

Install dependencies:

```
uv sync
```

Run the complete evaluation:

```
uv run main.py
```

This will:

1. Generate answers for all test questions for a given chunk size
2. Compute retrieval, answer-quality, and semantic metrics
3. Save results to `data/test_results.json`
4. Aggregate averages into `data/aggregated_results.json`

---

# 📊 Output Files

### **test_results.json**

Contains detailed per-question evaluation results across:

* Retrieved documents
* Generated answers
* Metrics (retrieval, answer quality, semantic)

### **aggregated_results.json**

Contains average values of all metrics across the 25 test questions.
Example:

```
{
  "small": { "avg_hit_rate": 1.0, ... },
  "medium": { ... },
  "large": { ... }
}
```

### **results_analysis.md**

Human-readable interpretation of results:

* Which chunk size performed best
* Failure modes
* Recommendations

---

# ⚙️ Chunking Configurations

You can adjust chunk sizes in:

```
CHUNK_CONFIGS = {
  "small":  {"chunk_size": 250, "chunk_overlap": 150},
  "medium": {"chunk_size": 550, "chunk_overlap": 150},
  "large":  {"chunk_size": 900, "chunk_overlap": 150}
}
```

The evaluation automatically runs per config.

---

# 📌 Notes & Limitations

* **Mistral 7B** used for *generation only*
* Local models struggle with RAGAS NLI modules → custom faithfulness metric used
* Relevance metric replaced with embedding similarity due to repeated timeouts
* Single-worker evaluation ensures stability on local hardware

These constraints are clearly documented in both the code and results analysis.

---

# 📝 Requirements

All dependencies are listed in `requirements.txt`.

Install via:

```
uv pip install -r requirements.txt
```
