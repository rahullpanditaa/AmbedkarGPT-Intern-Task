import json
from langchain_core.documents import Document
from .constants import TEST_DATASET_PATH

def combine_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def load_test_dataset() -> list[dict]:
    with open(TEST_DATASET_PATH, "r") as f:
        data = json.load(f)
    return data["test_questions"]