import json
from lib.rag_chain import create_rag_chain_for_config
from lib.search.search_utils import load_test_dataset
from langchain_core.documents import Document
from pathlib import Path
from metrics.retrieval_metrics import calculate_retrieval_metrics
from metrics.answer_quality_metrics import calculate_answer_quality_metrics
from metrics.semantic_metrics import calculate_semantic_metrics

TEST_RESULTS_PATH = Path(__file__).parent.parent.parent.resolve() / "data" / "test_results.json"
AGGREGATED_RESULTS_PATH = Path(__file__).parent.parent.parent.resolve() / "data" / "aggregated_results.json"

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

def aggregate_results():
    agg_results = {}

    with open(TEST_RESULTS_PATH, "r") as f:
        results = json.load(f)

    # results - dict where keys=small, medium, large, values = list[dict]
    for chunking_strategy, result in results.items():
        avg_hit_rate=avg_precision=avg_mrr=avg_rougel=avg_ans_rel=avg_faithfulness=avg_cos_sim=avg_bleu = 0.0

        # count = 25
        for result_per_question in result:
            # if None in result_per_question.values():
            #     count -= 1
            #     continue
            avg_hit_rate += result_per_question["hit_rate"] if result_per_question["hit_rate"] else 0
            avg_precision += result_per_question["precision_at_5"] if result_per_question["precision_at_5"] else 0
            avg_mrr += result_per_question["mrr"] if result_per_question["mrr"] else 0
            avg_rougel += result_per_question["rougeL"] if result_per_question["rougeL"] else 0
            avg_ans_rel += result_per_question["answer_relevance"] if result_per_question["answer_relevance"] else 0
            avg_faithfulness += result_per_question["faithfulness"] if result_per_question["faithfulness"] else 0
            avg_cos_sim += result_per_question["cosine_similarity"] if result_per_question["cosine_similarity"] else 0
            avg_bleu += result_per_question["bleu_score"] if result_per_question["bleu_score"] else 0
        agg_results[chunking_strategy] = {
            "avg_hit_rate": avg_hit_rate/25,
            "avg_precision": avg_precision/25,
            "avg_mrr": avg_mrr/25,
            "avg_rougeL": avg_rougel/25,
            "avg_relevance": avg_ans_rel/25,
            "avg_faithfulness": avg_faithfulness/25,
            "avg_cosine_similarity": avg_cos_sim/25,
            "avg_bleu_score": avg_bleu / 25
        }

    with open(AGGREGATED_RESULTS_PATH, "w") as f:
        json.dump(agg_results, f, indent=2)

    print("\n- Saved aggregated metrics to 'data/aggregated_results.json'")
        

