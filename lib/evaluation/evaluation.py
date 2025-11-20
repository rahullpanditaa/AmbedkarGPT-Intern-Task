import json
from lib.rag_chain import create_rag_chain_for_config
from lib.search.search_utils import load_test_dataset
from langchain_core.documents import Document
from pathlib import Path
from metrics.retrieval_metrics import calculate_retrieval_metrics
from metrics.answer_quality_metrics import calculate_answer_quality_metrics
from metrics.semantic_metrics import calculate_semantic_metrics

TEST_RESULTS_PATH = Path(__file__).parent.parent.parent.resolve() / "data" / "test_results.json"

CHUNK_CONFIGS = {
    "small":  {"chunk_size": 250, "chunk_overlap": 150},
    "medium": {"chunk_size": 550, "chunk_overlap": 150},
    "large":  {"chunk_size": 900, "chunk_overlap": 150},
}


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

# def evaluate_results():
#     final_results = {}

#     for name, cfg in CHUNK_CONFIGS.items():
#         print(f"\n- Evaluating chunking strategy - '{name.upper()}', (Chunk overlap: {cfg['chunk_overlap']}).")
#         results = evaluate_config(name, cfg)
#         # dict where key = chunking strategy name, value = results
#         final_results[name] = results

#     # save results
#     with open(TEST_RESULTS_PATH, "w") as f:
#         json.dump(final_results, f, indent=2)

#     print("\n- Saved evaluation results to 'test_results.json'")

def complete_evaluation():
    final_results = {}

    for name, cfg in CHUNK_CONFIGS.items():
        print(f"\n- Evaluating chunking strategy - '{name.upper()}', (Chunk overlap: {cfg['chunk_overlap']}).")
        results = evaluate_config(name, cfg)
        # now, calculate all evaluation metrics
        r = calculate_retrieval_metrics(config_name=name, results=results)
        r = calculate_answer_quality_metrics(config_name=name, results=r)
        r = calculate_semantic_metrics(config_name=name, results=r)
        final_results[name] = r

    with open(TEST_RESULTS_PATH, "w") as f:
        json.dump(final_results, f, indent=2)

    print("\n- Saved evaluation results to 'data/test_results.json'")
