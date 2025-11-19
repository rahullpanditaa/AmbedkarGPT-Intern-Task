from lib.rag_chain import create_rag_chain_for_config
from lib.search_utils import load_test_dataset
from langchain_core.documents import Document
from pathlib import Path

def evaluate_config(cfg_name, config):
    rag_chain, retriever = create_rag_chain_for_config(
        config_name=cfg_name,
        chunk_size=config['chunk_size'],
        chunk_overlap=config['chunk_overlap']
    )

    test_dataset_questions = load_test_dataset()
    results = []

    for q in test_dataset_questions:
        question = q["question"]
        ground_truth = q["ground_truth"]

        # speech1.txt, speech3.txt etc.
        source_docs = q["source_documents"]

        # retrieve relevant docs based on test question
        retrieved_docs: list[Document] = retriever._get_relevant_documents(query=question)
        # list of names of sources of retrieved docs
        retrieved_source_names = [
            Path(doc.metadata["source"]).resolve().name
            for doc in retrieved_docs
        ]

        # generate an answer to test question
        answer = rag_chain.invoke(question)

        results.append({
            "id": q["id"],
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": answer,
            "expected_docs_txt_files": ", ".join(source_docs),
            "retrieved_docs_txt_files": ", ".join(retrieved_source_names),
            "contexts": [doc.page_content for doc in retrieved_docs],
            "chunk_config": cfg_name,
            "question_type": q["question_type"],
            "answerable": q["answerable"]
        })

    return results


