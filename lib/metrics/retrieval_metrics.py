from evaluation.evaluation import evaluate_config

def calculate_retrieval_metrics(k: int=5):
    results: list[dict] = evaluate_config()

    results_with_metrics = []
    
    for result in results:
        expected_sources = result["expected_docs_txt_files"]
        retrieved_sources = result["retrieved_docs_txt_files"]
        hit_rate = calculate_hit_rate(expected_sources=expected_sources,
                                      retrieved_sources=retrieved_sources)
        
        precision_score = calculate_precision_score(expected_sources=expected_sources,
                                                    retrieved_sources=retrieved_sources)
        mrr = calculate_mrr(expected_sources=expected_sources,
                            retrieved_sources=retrieved_sources)
        new_result = result.copy()
        new_result["hit_rate"] = hit_rate if result["answerable"] else None
        new_result[f"precision_at_{k}"] = precision_score if result["answerable"] else None
        new_result["mrr"] = mrr if result["answerable"] else None

        results_with_metrics.append(new_result)

def calculate_hit_rate(expected_sources: list[str], retrieved_sources: list[str]) -> int:
    # return 1 if even one match, else 0
    for src in retrieved_sources:
        if src in expected_sources:
            return 1
    return 0

def calculate_precision_score(expected_sources: list[str], retrieved_sources: list[str]) -> float:
    # relevant retrieved / total retrieved
    relevant_retrieved = 0
    for src in retrieved_sources:
        if src in expected_sources:
            relevant_retrieved += 1
    return relevant_retrieved / len(retrieved_sources)

def calculate_mrr(expected_sources: list[str], retrieved_sources: list[str]):
    for i, retrieved in enumerate(retrieved_sources, 1):
        if retrieved in expected_sources:
            return 1 / i
            
    return 0

