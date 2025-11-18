from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
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

        # print for debugging/verifying
        print("Text loaded:")
        for i, doc in enumerate(documents, 1):
            print(f"Doc {i}. {doc.page_content}")

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator=".", 
            chunk_size=200,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False
        )

        docs_chunks = text_splitter.split_documents(documents)

        # print for debugging/verifying
        print(f"Chunked text document into {len(docs_chunks)}")
        for i, chunk in enumerate(docs_chunks, 1):
            print(f"Chunk {i}. {chunk.page_content}")

        # create vector store
        CHROMA_DIR_PATH.mkdir(parents=True, exist_ok=True)
        vector_store = Chroma.from_documents(documents=docs_chunks, embedding=self.model, 
                                             collection_name="ambedkar-gpt",
                                             persist_directory=str(CHROMA_DIR_PATH))
        print("Created vector store!!")
        # retriever
        vector_store_retriever = vector_store.as_retriever(search_type="similarity",
                                                           search_kwargs={"k": 5})
        self.retriever = vector_store_retriever
        return

    def load_or_create_vector_db(self):
        if self.retriever is None:
            print("Building a Chroma DB, getting a Retriever...")
            self.build_vector_db()
        
        print("Loading in-memory vector db...")
        return self.retriever
