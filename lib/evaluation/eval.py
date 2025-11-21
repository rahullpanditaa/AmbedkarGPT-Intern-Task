import json
import time
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


def evaluate_config(cfg_name):
    print(f"- Computing results for all test questions (pre evaluation metrics)...")
    cfg_to_use = CHUNK_CONFIGS[cfg_name]
    rag_chain, retriever = create_rag_chain_for_config(
        config_name=cfg_name,
        chunk_size=cfg_to_use['chunk_size'],
        chunk_overlap=cfg_to_use['chunk_overlap']
    )

    test_dataset_questions = load_test_dataset()
    results = []

    for i, q in enumerate(test_dataset_questions, 1):
        question = q["question"]
        ground_truth = q["ground_truth"]

        # speech1.txt, speech3.txt etc.
        source_docs = q["source_documents"]

        # retrieve relevant docs based on test question
        retrieved_docs: list[Document] = retriever.invoke(question)

        # list of names of sources of retrieved docs
        retrieved_source_names = [
            Path(doc.metadata.get("source", "unknown")).name
            for doc in retrieved_docs
        ]
        # retrieved_source_names = list(set(retrieved_source_names))

        # generate an answer to test question
        answer = rag_chain.invoke(question)

        time.sleep(5.0)

        results.append({
            "id": q["id"],
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": answer,
            "expected_docs_txt_files": source_docs,
            "retrieved_docs_txt_files": retrieved_source_names,
            "contexts": [doc.page_content for doc in retrieved_docs],
            "chunk_config": cfg_name,
            "question_type": q["question_type"],
            "answerable": q["answerable"]
        })
        
        print(f"- Generated answer for Q{i} ✔️")

    if TEST_RESULTS_PATH.exists():
        with open(TEST_RESULTS_PATH, "r") as f:
            results_before_eval = json.load(f)
    else:
        results_before_eval = {}
    results_before_eval[cfg_name] = results

    with open(TEST_RESULTS_PATH, "w") as f:
        json.dump(results_before_eval, f, indent=2)

    print(f" - Results for test questions (before evaluation) written to '{TEST_RESULTS_PATH.name}'")

def complete_evaluation_metrics(cfg_name: str):
    print(f"\n- Evaluating chunking strategy - '{cfg_name.upper()}'...")

    calculate_retrieval_metrics(cfg_name=cfg_name)
    calculate_answer_quality_metrics(cfg_name=cfg_name)
    calculate_semantic_metrics(cfg_name=cfg_name)

    print(f"\n- All metrics evaluated for '{cfg_name.upper()}'.")
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
    