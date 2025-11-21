from pathlib import Path
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# project root / data directory
DATA_DIR_PATH = Path(__file__).parent.parent.parent.resolve() / "data"
TEST_RESULTS_PATH = DATA_DIR_PATH / "test_results.json"
TEST_DATASET_PATH = DATA_DIR_PATH / "test_dataset.json"
AGGREGATED_RESULTS_PATH = DATA_DIR_PATH / "aggregated_results.json"

CHROMA_DIR_PATH = Path(__file__).parent.parent.parent.resolve() / "vector_dbs"

# All possible configurations for evaluation runs
CHUNK_CONFIGS = {
    "small":  {"chunk_size": 250, "chunk_overlap": 150},
    "medium": {"chunk_size": 550, "chunk_overlap": 150},
    "large":  {"chunk_size": 900, "chunk_overlap": 150},
}

# Answer generation llm
LLM_ANSWER_GENERATION = OllamaLLM(model="mistral")

# Answer quality metrics utilities
HF_EMBEDDING = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
LLM_METRICS = OllamaLLM(model="phi3:mini", keep_alive="60m")

# RAG chain prompt
PROMPT = ChatPromptTemplate.from_template("""Answer the question based on the provided context,
    
Question: {question}

Context: 
{context}

Provide an answer that addresses the question:""")