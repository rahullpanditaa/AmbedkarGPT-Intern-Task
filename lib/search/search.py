from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from .search_utils import ( 
    DATA_DIR_PATH,
    CHROMA_DIRS_PATH
)

class SemanticSearch: 
    """
    Semantic Search component used to build or load a local vector database for 
    Retrieval-Augmented Generation.
    """

    def __init__(self, chunk_size, chunk_overlap, persist_dir, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_dir = CHROMA_DIRS_PATH / persist_dir
        self.model = HuggingFaceEmbeddings(model_name=model_name)

    def build_vector_db(self) -> VectorStoreRetriever:
        """Creates a Chroma vector DB from scratch, returns its retriever."""

        DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
        # load all 6 files
        all_docs: list[Document] = []
        for i in range(1,7):
            doc = TextLoader(DATA_DIR_PATH / f"speech{i}.txt").load()
            all_docs.extend(doc)
        

        # split into chunks - different chunking strategy based on constructor args
        text_splitter = CharacterTextSplitter(
            separator=".", 
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )

        all_docs_chunks = text_splitter.split_documents(all_docs)

        # create vector store
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        print(f" - Creating a vector store at {self.persist_dir.name}...")
        vector_store = Chroma.from_documents(documents=all_docs_chunks, 
                                             embedding=self.model,
                                             collection_name="ambedkar-corpus",                                             
                                             persist_directory=str(self.persist_dir))        
        print(" - vector store created.")
        # retriever
        vector_store_retriever = vector_store.as_retriever(search_type="similarity",
                                                           search_kwargs={"k": 5})
        return vector_store_retriever

    def load_or_create_vector_db(self) -> VectorStoreRetriever:
        """Loads an existing Chroma DB if available; otherwise, builds a new one."""

        if self.persist_dir.exists() and (self.persist_dir / "chroma.sqlite3").exists():
            print(f" - Loading vector store from disk at {self.persist_dir.name}...")
            # load db
            vs = Chroma(collection_name="ambedkar-corpu", 
                        embedding_function=self.model, 
                        persist_directory=str(self.persist_dir))
            print(" - vector store loaded.")
            return vs.as_retriever(search_type="similarity", search_kwargs={"k":5})
        else:
            retriever = self.build_vector_db()
            return retriever
        