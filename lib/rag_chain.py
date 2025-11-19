from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from .search_utils import combine_docs, PROMPT
from .search import SemanticSearch


# invoke llm    
llm = OllamaLLM(model="mistral")

def create_rag_chain():
    # get retriever
    searcher = SemanticSearch()
    retriever = searcher.load_or_create_vector_db()

    # runnable that converts list[docs] into a string for llm
    doc_combiner = RunnableLambda(combine_docs)

    rag_inputs = {
        "context": retriever | doc_combiner,
        "question": RunnablePassthrough()
    }

    rag_chain = rag_inputs | PROMPT | llm
    return rag_chain

