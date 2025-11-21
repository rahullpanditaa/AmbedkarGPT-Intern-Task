from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from .search.search_utils import combine_docs, PROMPT, CHROMA_DIRS_PATH
from .search.search import SemanticSearch


# initialize Ollama LLM   
llm = OllamaLLM(model="mistral")

# def create_rag_chain():
#     """
#     Build and return the complete Retrieval-Augmented Generation (RAG) pipeline
#     using LangChain Runnables, Chroma vector search, HuggingFace embeddings,
#     and the Ollama Mistral LLM.
    
    
#     Returns
#     -------
#     Runnable
#         A LangChain runnable that accepts a user question (str) and returns
#         an LLM-generated answer that's grounded in the retrieved context."""    

#     # load or create retriever
#     searcher = SemanticSearch()
#     retriever = searcher.load_or_create_vector_db()

#     # runnable that converts list[docs] into a string for llm
#     doc_combiner = RunnableLambda(combine_docs)

#     # langchain will call each key-value pair with the input question
#     rag_inputs = {
#         # context : retrieved docs -> combined into a str
#         "context": retriever | doc_combiner,
#         "question": RunnablePassthrough()
#     }

#     # rag_inputs becomes:
#     #   "context": "retrieved text chunks merged into a single string"
#     #   "question": "original question as it is"
    

#     # context and question -> prompt template -> LLM
#     rag_chain = rag_inputs | PROMPT | llm
#     return rag_chain

def create_rag_chain_for_config(config_name: str, chunk_size: int, chunk_overlap: int):
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