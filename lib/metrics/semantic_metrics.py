# from lib.evaluation.evaluation import evaluate_config
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pathlib import Path
import json

TEST_RESULTS_PATH = Path(__file__).parent.parent.parent.resolve() / "data" / "test_results.json"

CHUNK_CONFIGS = {
    "small":  {"chunk_size": 250, "chunk_overlap": 150},
    "medium": {"chunk_size": 550, "chunk_overlap": 150},
    "large":  {"chunk_size": 900, "chunk_overlap": 150},
}
HF_EMBEDDING = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def calculate_semantic_metrics(cfg_name: str):
    print(f"\n- Calculating Semantic metrics for chunking config '{cfg_name.upper()}'...")
    # Load existing results
    if TEST_RESULTS_PATH.exists():
        with open(TEST_RESULTS_PATH, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    # Safety check: ensure this config exists
    if cfg_name not in all_results:
        print(f"No base results found for config '{cfg_name}'. Run evaluate_config() first.")
        return

    cfg_results = all_results[cfg_name]   # list of all test question results

    updated_cfg_results = []

    for result in cfg_results:
        ground_truth = result["ground_truth"]
        generated_answer = result["generated_answer"]

        cos_sim = _calculate_cosine_similarity(ground_truth=ground_truth,
                                               generated_answer=generated_answer) if result["answerable"] else None
        bleu_score = _calculate_bleu_score(ground_truth=ground_truth,
                                           generated_answer=generated_answer) if result["answerable"] else None
        
        new_result = result.copy()
        new_result["cosine_similarity"] = cos_sim
        new_result["bleu_score"] = bleu_score
        updated_cfg_results.append(new_result)

    # updated_results = {}
    # updated_results[cfg_name] = updated_cfg_results
    all_results[cfg_name] = updated_cfg_results

    # save updated results
    with open(TEST_RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n- Updated results, saved Semantic metrics, file at '{TEST_RESULTS_PATH.name}'. ")


def _calculate_cosine_similarity(ground_truth: str, generated_answer: str):
    if ground_truth.strip() == "" or generated_answer.strip() == "":
        return 0.0

    # need to embed both
    # hf = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    truth_embedding = HF_EMBEDDING.embed_query(ground_truth)
    answer_embedding = HF_EMBEDDING.embed_query(generated_answer)

    cos_sim = cosine_similarity(X=[truth_embedding], Y=[answer_embedding])
    return float(cos_sim[0][0])

# n-gram overlap i.e. 
def _calculate_bleu_score(ground_truth: str, generated_answer: str):
    if ground_truth.strip() == "" or generated_answer.strip() == "":
        return 0.0
    
    # create ground_truth, answer tokens
    gt = ground_truth.lower().split()
    ans = generated_answer.lower().split()

    sm = SmoothingFunction().method1()

    bleu = sentence_bleu(references=[gt], hypothesis=ans, smoothing_function=sm)

    return float(bleu)