CHUNK_CONFIGS = {
    "small":  {"chunk_size": 250, "chunk_overlap": 150},
    "medium": {"chunk_size": 550, "chunk_overlap": 150},
    "large":  {"chunk_size": 900, "chunk_overlap": 150},
}


def calculate_retrieval_metrics(config_name: str, results: list[dict]):
    # results: list[dict] = evaluate_config(cfg_name=config_name.lower(), config=CHUNK_CONFIGS[config_name.lower()])

    results_with_metrics = []
    
    for result in results:
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

        results_with_metrics.append(new_result)

    return results_with_metrics

def _calculate_hit_rate(expected_sources: list[str], retrieved_sources: list[str]) -> int:
    if len(retrieved_sources) == 0:
        return 0
    # return 1 if even one match, else 0
    for src in retrieved_sources:
        if src in expected_sources:
            return 1
    return 0

def _calculate_precision_score(expected_sources: list[str], retrieved_sources: list[str]) -> float:
    if len(retrieved_sources) == 0:
        return 0.0
    # relevant retrieved / total retrieved
    relevant_retrieved = 0
    for src in retrieved_sources:
        if src in expected_sources:
            relevant_retrieved += 1
    return relevant_retrieved / len(retrieved_sources)

def _calculate_mrr(expected_sources: list[str], retrieved_sources: list[str]) -> float:
    if len(retrieved_sources) == 0:
        return 0.0
    for i, retrieved in enumerate(retrieved_sources, 1):
        if retrieved in expected_sources:
            return 1 / i
            
    return 0.0

