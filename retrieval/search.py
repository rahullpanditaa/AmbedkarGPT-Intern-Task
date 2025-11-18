from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from .search_utils import (
    SPEECH_TXT_PATH, 
    DATA_DIR_PATH,
    CHROMA_DIR_PATH
)

class Search:    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model: VectorStoreRetriever = HuggingFaceEmbeddings(model_name=model_name)
        self.retriever = None

    def build_vector_db(self) -> None:
        # load text
        DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
        loader = TextLoader(SPEECH_TXT_PATH)
        documents = loader.load()

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator=".", 
            chunk_size=200,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False
        )

        docs_chunks = text_splitter.split_documents(documents)

        # create vector store
        CHROMA_DIR_PATH.mkdir(parents=True, exist_ok=True)
        vector_store = Chroma.from_documents(documents=docs_chunks, embedding=self.model, 
                                             collection_name="ambedkar-gpt",
                                             persist_directory=str(CHROMA_DIR_PATH))
        # retriever
        vector_store_retriever = vector_store.as_retriever(search_type="similarity",
                                                           search_kwargs={"k": 5})
        self.retriever = vector_store_retriever

    def load_or_create_vector_db(self):
        if self.retriever is None:
            self.build_vector_db()
        return self.retriever
