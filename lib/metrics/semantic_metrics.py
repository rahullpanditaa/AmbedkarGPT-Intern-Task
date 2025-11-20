# from lib.evaluation.evaluation import evaluate_config
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

CHUNK_CONFIGS = {
    "small":  {"chunk_size": 250, "chunk_overlap": 150},
    "medium": {"chunk_size": 550, "chunk_overlap": 150},
    "large":  {"chunk_size": 900, "chunk_overlap": 150},
}

def calculate_semantic_metrics(results: list[dict]):
    # results: list[dict] = evaluate_config(cfg_name=config_name.lower(),
                            #   config=CHUNK_CONFIGS[config_name.lower()])
    
    results_with_semantic_metrics = []

    for result in results:
        ground_truth = result["ground_truth"]
        generated_answer = result["generated_answer"]

        cos_sim = _calculate_cosine_similarity(ground_truth=ground_truth,
                                               generated_answer=generated_answer) if result["answerable"] else None
        bleu_score = _calculate_bleu_score(ground_truth=ground_truth,
                                           generated_answer=generated_answer) if result["answerable"] else None
        
        new_result = result.copy()
        new_result["cosine_similarity"] = cos_sim
        new_result["bleu_score"] = bleu_score
        results_with_semantic_metrics.append(new_result)

    return results_with_semantic_metrics


def _calculate_cosine_similarity(ground_truth: str, generated_answer: str):
    if ground_truth.strip() == "" or generated_answer.strip() == "":
        return 0.0

    # need to embed both
    hf = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    truth_embedding = hf.embed_query(ground_truth)
    answer_embedding = hf.embed_query(generated_answer)

    cos_sim = cosine_similarity(X=[truth_embedding], Y=[answer_embedding])
    return float(cos_sim[0][0])

# n-gram overlap i.e. 
def _calculate_bleu_score(ground_truth: str, generated_answer: str):
    if ground_truth.strip() == "" or generated_answer.strip() == "":
        return 0.0
    
    # create ground_truth, answer tokens
    gt = ground_truth.split()
    ans = generated_answer.split()

    sm = SmoothingFunction().method1()

    bleu = sentence_bleu(references=[gt], hypothesis=ans, smoothing_function=sm)

    return float(bleu)
    # data = {
    #     "reference": ground_truth,
    #     "response": generated_answer
    # }

    # dataset = Dataset.from_dict(data)
    # result = evaluate(dataset=dataset, metrics=[])