from lib.search import SemanticSearch
from lib.search_utils import (
    CHUNK_SIZE_LARGE,
    CHUNK_SIZE_MEDIUM,
    CHUNK_SIZE_SMALL,
    PERSIST_DIR_SMALL,
    PERSIST_DIR_MEDIUM,
    PERSIST_DIR_LARGE
)


def main():
    searcher_small_chunks = SemanticSearch(chunk_size=CHUNK_SIZE_SMALL, 
                                           chunk_overlap=150, 
                                           persist_dir=PERSIST_DIR_SMALL)
    # searcher_medium_chunks = SemanticSearch(chunk_size=CHUNK_SIZE_MEDIUM,
    #                                         chunk_overlap=150, 
    #                                         persist_dir=PERSIST_DIR_MEDIUM)
    # searcher_large_chunks = SemanticSearch(chunk_size=CHUNK_SIZE_LARGE,
    #                                       chunk_overlap=150,
    #                                       persist_dir=PERSIST_DIR_LARGE)
    retriever1 = searcher_small_chunks.load_or_create_vector_db()
    # retriever2 = searcher_medium_chunks.load_or_create_vector_db()
    # retriever3 = searcher_large_chunks.load_or_create_vector_db()
    # chain = create_rag_chain()
    # print(" - Welcome to AmbedkarGPT.")
    # print(" - Starting REPL...\n")
    # while True:
    #     question = input("> ")
    #     if question.lower() == "exit":
    #         break
    #     # pass question to rag_inputs
    #     response = chain.invoke(question)
    #     print(response)


if __name__ == "__main__":
    main()
