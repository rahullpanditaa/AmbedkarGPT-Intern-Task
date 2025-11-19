# AmbedkarGPT — Internship Task

## Phase 1

A **Retrieval-Augmented Generation (RAG)** system built using **LangChain**, **ChromaDB**, **HuggingFace Embeddings**, and **Ollama (Mistral 7B)**. This project implements a CLI Q&A system that answers questions based solely on an excerpt from Dr. B.R. Ambedkar's *Annihilation of Caste*.

---

## 🚀 Features

* Loads a local text file (`speech.txt`)
* Splits text into overlapping chunks
* Computes embeddings locally using **sentence-transformers/all-MiniLM-L6-v2**
* Stores embeddings in a persistent **Chroma vector database**
* Retrieves relevant chunks using semantic similarity search
* Passes retrieved context + user question into **Ollama Mistral 7B**
* Provides grounded answers from the content

---

## 📦 Project Structure

```
AmbedkarGPT-Intern-Task/
│
├── data/
│   └── speech.txt                # Provided text for RAG
│
├── lib/                          # Core logic for retrieval + RAG
│   ├── search.py                 # SemanticSearch class (build/load vector DB)
│   ├── rag_chain.py              # RAG chain construction
│   └── search_utils.py           # Helper utilities
│
├── vector_db/                    # Chroma persistent DB (ignored in git)
│   ├── chroma.sqlite3
│   └── <HNSW index folder>
│
├── main.py                       # CLI interface
├── pyproject.toml                # Dependencies
├── README.md                     # (this file)
└── .gitignore                    # Clean repo configuration
```

---

## 🛠️ Setup Instructions

### 1. Install Dependencies

This project uses **uv** as the package manager.

```bash
uv sync
```

### 2. Install Ollama

Linux:

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 3. Pull Mistral 7B

```bash
ollama pull mistral
```

Make sure Ollama is running (it usually runs automatically):

```bash
ps aux | grep ollama
```

---

## ▶️ Running the CLI App

From the project root:

```bash
uv run main.py
```

### Example Session

```
Welcome to AmbedkarGPT.
Starting REPL...
> What is caste?
Caste is described as...

> Why is reform compared to gardening?
The text explains that...

> exit
```

---

## 🧠 How It Works

1. **Load the speech text** using LangChain's `TextLoader`.
2. **Chunk the text** with `CharacterTextSplitter`.
3. **Embed chunks** using `HuggingFaceEmbeddings` (MiniLM-L6-v2).
4. **Create/load ChromaDB** to persist embeddings.
5. **Build a VectorStoreRetriever** for similarity search.
6. **Assemble the RAG pipeline** using LangChain Runnables:

   * Retrieval → Merge Chunks → Prompt Template → Mistral LLM
7. **Answer user questions** purely from retrieved context.

---

## 📚 Code Overview

### 🔹 `SemanticSearch` (vector DB manager)

* Loads `speech.txt`
* Splits into chunks
* Creates/loads ChromaDB
* Returns a `VectorStoreRetriever`

### 🔹 `create_rag_chain()` (pipeline builder)

* Loads the retriever
* Prepares a doc-combiner runnable
* Builds a prompt template
* Connects everything into a runnable chain

### 🔹 `main.py` (CLI)

* Initializes the RAG chain once
* Prompts user
* Invokes the chain with the question

---

## 🧪 Example Questions to Try

* "What is caste?"
* "What is the real enemy according to Ambedkar?"
* "Why does Ambedkar compare social reform work to gardening?"
* "What must be destroyed to remove caste?"

---

## 🧹 .gitignore

* vector_db/
* .venv/
* **pycache**/
* uv.lock
