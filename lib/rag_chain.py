from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from .search.search_utils import combine_docs, PROMPT, CHROMA_DIRS_PATH
from .search.search import SemanticSearch


# initialize Ollama LLM   
llm = OllamaLLM(model="mistral")

def create_rag_chain_for_config(config_name: str, chunk_size: int, chunk_overlap: int):
    """
    Constructs a fully configured RAG pipeline for a given chunking strategy.

    This function:
    - Initializes or loads a ChromaDB-based vector store via `SemanticSearch`.
    - Creates a retriever using the chosen chunk size and overlap.
    - Builds a LangChain runnable composed of:
        * document retrieval
        * context combination via `combine_docs`
        * insertion into a prompt template
        * generation via an Ollama-hosted LLM
    - Returns both the runnable RAG chain and the retriever itself.

    Args:
        config_name (str): Name of the chunk configuration (e.g. "small",
            "medium", "large"). Used for selecting a dedicated Chroma directory.
        chunk_size (int): Size of each text chunk for vectorization.
        chunk_overlap (int): Overlap between consecutive text chunks.

    Returns:
        tuple:
            - rag_chain: A LangChain runnable that takes a question and produces
                         an answer using retrieval + generation.
            - retriever: The underlying retriever used to fetch relevant chunks"""
    persist_dir = f"vector_db_{config_name}"

    # load or create retriever
    searcher = SemanticSearch(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        persist_dir=persist_dir
    )
    retriever = searcher.load_or_create_vector_db()

    # runnable that converts list[docs] into a string for llm
    doc_combiner = RunnableLambda(combine_docs)

    rag_inputs = {
        "context": retriever | doc_combiner,
        "question": RunnablePassthrough()
    }

    rag_chain = rag_inputs | PROMPT | llm

    #                 to get actual retrieved chunks
    return rag_chain, retriever