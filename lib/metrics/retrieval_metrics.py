from pathlib import Path
import json

CHUNK_CONFIGS = {
    "small":  {"chunk_size": 250, "chunk_overlap": 150},
    "medium": {"chunk_size": 550, "chunk_overlap": 150},
    "large":  {"chunk_size": 900, "chunk_overlap": 150},
}
TEST_RESULTS_PATH = Path(__file__).parent.parent.parent.resolve() / "data" / "test_results.json"

def calculate_retrieval_metrics(cfg_name: str):
    """
    Computes retrieval quality metrics (Hit Rate, Precision@K, MRR) for all
    questions under the given chunking configuration and updates
    `test_results.json` with the new values.

    Retrieval metrics evaluate how well the retriever selected the correct
    source documents compared to the expected documents defined in the test
    dataset.

    Args:
        cfg_name (str): The name of the chunking configuration to evaluate.
                        Must be one of {"small", "medium", "large"}."""
    print(f"\n- Calculating Retrieval Quality metrics for chunking config '{cfg_name.upper()}'...")
    if TEST_RESULTS_PATH.exists():
        with open(TEST_RESULTS_PATH, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    if cfg_name not in all_results:
        print(f"No base results found for config '{cfg_name}'. Run evaluate_config() first.")
        return

    cfg_results = all_results[cfg_name]

    updated_cfg_results = []
    
    for result in cfg_results:
        expected_sources = result["expected_docs_txt_files"]
        retrieved_sources = result["retrieved_docs_txt_files"]
        hit_rate = _calculate_hit_rate(expected_sources=expected_sources,
                                      retrieved_sources=retrieved_sources) if result["answerable"] else None
        
        precision_score = _calculate_precision_score(expected_sources=expected_sources,
                                                    retrieved_sources=retrieved_sources) if result["answerable"] else None
        mrr = _calculate_mrr(expected_sources=expected_sources,
                            retrieved_sources=retrieved_sources) if result["answerable"] else None
        new_result = result.copy()
        new_result["hit_rate"] = hit_rate
        new_result[f"precision_at_5"] = precision_score
        new_result["mrr"] = mrr

        updated_cfg_results.append(new_result)

    # updated_results = {}
    # updated_results[cfg_name] = updated_cfg_results
    all_results[cfg_name] = updated_cfg_results

    # save updated results
    with open(TEST_RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n- Updated results, saved Retrieval Quality metrics, file at '{TEST_RESULTS_PATH.name}'. ")

def _calculate_hit_rate(expected_sources: list[str], retrieved_sources: list[str]) -> int:
    """
    Computes Hit Rate for a single question.

    Hit Rate = 1 if any retrieved document matches any expected document,
    otherwise 0.

    Args:
        expected_sources (list[str]): The ground-truth source document filenames.
        retrieved_sources (list[str]): The filenames returned by the retriever.

    Returns:
        int: 1 if at least one expected document was retrieved, else 0."""
    if len(retrieved_sources) == 0:
        return 0
    # return 1 if even one match, else 0
    for src in retrieved_sources:
        if src in expected_sources:
            return 1
    return 0

def _calculate_precision_score(expected_sources: list[str], retrieved_sources: list[str]) -> float:
    """
    Computes Precision@K for a single question, where K = number of retrieved
    documents.

    Precision@K = (# relevant retrieved docs) / (# total retrieved docs)

    Args:
        expected_sources (list[str]): The ground-truth source document filenames.
        retrieved_sources (list[str]): The filenames returned by the retriever.

    Returns:
        float: Precision value in the range [0, 1]. Returns 0.0 if no documents
               were retrieved."""
    if len(retrieved_sources) == 0:
        return 0.0
    # relevant retrieved / total retrieved
    relevant_retrieved = 0
    for src in retrieved_sources:
        if src in expected_sources:
            relevant_retrieved += 1
    return relevant_retrieved / len(retrieved_sources)

def _calculate_mrr(expected_sources: list[str], retrieved_sources: list[str]) -> float:
    """
    Computes the Mean Reciprocal Rank (MRR) for a single question.

    MRR = 1 / rank of the first correctly retrieved document, where rank is
    1-indexed. If no expected document appears in the retrieved list, returns 0.

    Args:
        expected_sources (list[str]): The ground-truth source document filenames.
        retrieved_sources (list[str]): The filenames returned by the retriever.

    Returns:
        float: Reciprocal rank of the first relevant document, or 0.0 if none
               were retrieved."""
    if len(retrieved_sources) == 0:
        return 0.0
    for i, retrieved in enumerate(retrieved_sources, 1):
        if retrieved in expected_sources:
            return 1 / i
            
    return 0.0

