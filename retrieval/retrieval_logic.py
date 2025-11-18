from langchain_community.document_loaders import (
    TextLoader
)
from langchain_text_splitters import CharacterTextSplitter

from .search_utils import SPEECH_TXT_PATH, DATA_DIR_PATH


class Search:
    def __init__(self, model_name):
        self.model = model_name

    def build_vector_db(self):
        # load text
        DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
        loader = TextLoader(SPEECH_TXT_PATH)
        documents = loader.load()

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator=".", 
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False
        )

        docs_splits = text_splitter.split_documents(documents)
        


