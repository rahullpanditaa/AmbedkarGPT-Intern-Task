import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

DATA_DIR_PATH = Path(__file__).parent.parent.parent.resolve() / "data"
CHROMA_DIRS_PATH = Path(__file__).parent.parent.parent.resolve() / "vector_dbs"
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


TEST_DATASET_PATH = DATA_DIR_PATH / "test_dataset.json"
def load_test_dataset() -> list[dict]:
    with open(TEST_DATASET_PATH, "r") as f:
        data = json.load(f)

    # return a list of dicts, 
    # each dict - {id, question, truth, source, ques_type, answerable}
    return data["test_questions"]