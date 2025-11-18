from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from .search_utils import SPEECH_TXT_PATH, DATA_DIR_PATH


class Search:
    chunks_metadata: list[dict]

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = HuggingFaceEmbeddings(model_name=model_name)
        self.chunk_embeddings = None
        self.chunks_metadata = None

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

        docs_chunks = text_splitter.split_documents(documents)

        # create chunk embeddings
        chunk_texts = []
        for chunk in docs_chunks:
            self.chunks_metadata.append(chunk.metadata)
            chunk_texts.append(chunk.page_content)
        
        self.chunk_embeddings = self.model.embed_documents(chunk_texts)

