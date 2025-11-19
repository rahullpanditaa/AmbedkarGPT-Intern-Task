import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

DATA_DIR_PATH = Path(__file__).parent.parent.resolve() / "data"

CHROMA_DIRS_PATH = Path(__file__).parent.parent.resolve() / "vector_dbs"
PERSIST_DIR_SMALL = "vector_db_small"
PERSIST_DIR_MEDIUM = "vector_db_medium"
PERSIST_DIR_LARGE = "vector_db_large"

CHUNK_SIZE_SMALL = 250
CHUNK_SIZE_MEDIUM = 550
CHUNK_SIZE_LARGE = 900

PROMPT = ChatPromptTemplate.from_template("""Answer the question based on the provided context,
    
Question: {question}

Context: 
{context}

Provide an answer that addresses the question:""")

def combine_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

CHUNK_CONFIGS = {
    "small":  {"chunk_size": 250, "chunk_overlap": 150},
    "medium": {"chunk_size": 550, "chunk_overlap": 150},
    "large":  {"chunk_size": 900, "chunk_overlap": 150},
}

def load_test_dataset() -> list[dict]:
    dataset = Path("test_dataset.json").resolve()
    with open(dataset, "r") as f:
        data = json.load(f)

    # return a list of dicts, 
    # each dict - {id, question, truth, source, ques_type, answerable}
    return data["test_questions"]