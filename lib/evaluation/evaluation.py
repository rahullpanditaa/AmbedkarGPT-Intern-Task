import json
from lib.rag_chain import create_rag_chain_for_config
from lib.search.search_utils import load_test_dataset
from langchain_core.documents import Document
from pathlib import Path
from lib.metrics.retrieval_metrics import calculate_retrieval_metrics
from lib.metrics.answer_quality_metrics import calculate_answer_quality_metrics
from lib.metrics.semantic_metrics import calculate_semantic_metrics

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
        retrieved_docs: list[Document] = retriever._get_relevant_documents(query=question, run_manager=None)
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

def complete_evaluation(cfg_name: str):
    cfg_results = {}

    print(f"\n- Evaluating chunking strategy - '{cfg_name.upper()}'...")
    results = evaluate_config(cfg_name=cfg_name, config=CHUNK_CONFIGS[cfg_name])
    r = calculate_retrieval_metrics(config_name=cfg_name, results=results)
    r = calculate_answer_quality_metrics(config_name=cfg_name ,results=r)
    r = calculate_semantic_metrics(config_name=cfg_name, results=r)
    cfg_results[cfg_name] = r


    # {cfg_name: list[dict]}

    # for name, cfg in CHUNK_CONFIGS.items():
    #     print(f"\n- Evaluating chunking strategy - '{name.upper()}', (Chunk overlap: {cfg['chunk_overlap']}).")
    #     results = evaluate_config(name, cfg)
    #     # now, calculate all evaluation metrics
    #     r = calculate_retrieval_metrics(config_name=name, results=results)
    #     r = calculate_answer_quality_metrics(config_name=name, results=r)
    #     r = calculate_semantic_metrics(config_name=name, results=r)
    #     final_results[name] = r

    # if yes, result for at least one config already written
    if TEST_RESULTS_PATH.exists():
        with open(TEST_RESULTS_PATH, "r") as f:
            test_results = json.load(f)
    else:
        test_results = {}

    # merge dicts
    final_results = test_results | cfg_results

    with open(TEST_RESULTS_PATH, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\n- Saved evaluation results for '{cfg_name.upper()}' to 'data/test_results.json'")

def aggregate_results(cfg_name: str):
    # load aggregated results if they exist
    if AGGREGATED_RESULTS_PATH.exists():
        with open(AGGREGATED_RESULTS_PATH, "r") as f:
            agg_results = json.load(f)
    else:
        agg_results = {}

    if not TEST_RESULTS_PATH.exists():
        print("File 'test_results.json' does not exist. Run evaluation first.")
        return

    with open(TEST_RESULTS_PATH, "r") as f:
        results = json.load(f)

    # results - {small: list[dict], medium: list[dict], large: list[dict]}
    cfg_results = results.get(cfg_name)

    if cfg_results is None:
        print(f"No results found for config '{cfg_name}'.")
        return

    # cfg_results = list[dict]
    # for result in cfg_results:
    agg_results[cfg_name] = {
            "avg_hit_rate": _calculate_avg_for_each_metric(results=cfg_results, metric="hit_rate"),
            "avg_precision": _calculate_avg_for_each_metric(results=cfg_results, metric="precision_at_5"),
            "avg_mrr": _calculate_avg_for_each_metric(results=cfg_results, metric="mrr"),
            "avg_rougeL": _calculate_avg_for_each_metric(results=cfg_results, metric="rougeL"),
            "avg_relevance": _calculate_avg_for_each_metric(results=cfg_results, metric="answer_relevance"),
            "avg_faithfulness": _calculate_avg_for_each_metric(results=cfg_results, metric="faithfulness"),
            "avg_cosine_similarity": _calculate_avg_for_each_metric(results=cfg_results, metric="cosine_similarity"),
            "avg_bleu": _calculate_avg_for_each_metric(results=cfg_results, metric="bleu_score"),
        }

    # for chunking_strategy, result in results.items():
    #     agg_results[chunking_strategy] = {
    #         "avg_hit_rate": _calculate_avg_for_each_metric(results=result, metric="hit_rate"),
    #         "avg_precision": _calculate_avg_for_each_metric(results=result, metric="precision_at_5"),
    #         "avg_mrr": _calculate_avg_for_each_metric(results=result, metric="mrr"),
    #         "avg_rougeL": _calculate_avg_for_each_metric(results=result, metric="rougeL"),
    #         "avg_relevance": _calculate_avg_for_each_metric(results=result, metric="answer_relevance"),
    #         "avg_faithfulness": _calculate_avg_for_each_metric(results=result, metric="faithfulness"),
    #         "avg_cosine_similarity": _calculate_avg_for_each_metric(results=result, metric="cosine_similarity"),
    #         "avg_bleu": _calculate_avg_for_each_metric(results=result, metric="bleu_score"),
    #     }
                

    with open(AGGREGATED_RESULTS_PATH, "w") as f:
        json.dump(agg_results, f, indent=2)

    print(f"\n- Saved aggregated metrics for '{cfg_name.upper()}' to 'data/aggregated_results.json'")

def _calculate_avg_for_each_metric(results: list[dict], metric: str) -> float:
    metric_sum = 0.0
    metric_count = 0.0
    for result_per_question in results:
        value = result_per_question[metric]
        if value is not None:
            metric_sum += value
            metric_count += 1

    return metric_sum / metric_count if metric_count > 0 else 0.0
    