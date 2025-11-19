from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from .search_utils import (
    SPEECH_TXT_PATH, 
    DATA_DIR_PATH,
    CHROMA_DIR_PATH
)

class SemanticSearch: 
    """
    Semantic Search component used to build or load a local vector database for 
    Retrieval-Augmented Generation.
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = HuggingFaceEmbeddings(model_name=model_name)

    def build_vector_db(self) -> VectorStoreRetriever:
        """Creates a Chroma vector DB from scratch, returns its retriever."""

        # load text
        DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
        # loader = TextLoader(SPEECH_TXT_PATH)
        # documents = loader.load()

        # load all 6 files
        all_docs: list[Document] = []
        for i in range(1,7):
            doc = TextLoader(DATA_DIR_PATH / f"speech{i}.txt").load()
            all_docs.extend(doc)
        

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
        vector_store = Chroma.from_documents(documents=docs_chunks, 
                                             embedding=self.model,
                                             collection_name="ambedkar-gpt",                                             
                                             persist_directory=str(CHROMA_DIR_PATH))
        print("\n - Building a ChromaDB...")
        print(" - Created vector store!!\n")
        # retriever
        vector_store_retriever = vector_store.as_retriever(search_type="similarity",
                                                           search_kwargs={"k": 5})
        return vector_store_retriever

    def load_or_create_vector_db(self) -> VectorStoreRetriever:
        """Loads an existing Chroma DB if available; otherwise, builds a new one."""

        if CHROMA_DIR_PATH.exists() and (CHROMA_DIR_PATH / "chroma.sqlite3").exists():
            print("\n - Loading vector store from disk...\n")
            # load db
            vs = Chroma(collection_name="ambedkar-gpt", 
                        embedding_function=self.model, 
                        persist_directory=str(CHROMA_DIR_PATH))
            return vs.as_retriever(search_type="similarity", search_kwargs={"k":5})
        else:
            retriever = self.build_vector_db()
            return retriever
        