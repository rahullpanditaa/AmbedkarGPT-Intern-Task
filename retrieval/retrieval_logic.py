from langchain_community.document_loaders import (
    TextLoader,
)
from .search_utils import SPEECH_TXT_PATH, DATA_DIR_PATH


class Search:
    def __init__(self):
        ...

    def build_vector_db(self):
        DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
        loader = TextLoader(SPEECH_TXT_PATH)
        documents = loader.load()
        return documents

