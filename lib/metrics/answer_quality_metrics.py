import json
from pathlib import Path
from rouge_score import rouge_scorer
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from .semantic_metrics import _calculate_cosine_similarity

TEST_RESULTS_PATH = Path(__file__).parent.parent.parent.resolve() / "data" / "test_results.json"

CHUNK_CONFIGS = {
    "small":  {"chunk_size": 250, "chunk_overlap": 150},
    "medium": {"chunk_size": 550, "chunk_overlap": 150},
    "large":  {"chunk_size": 900, "chunk_overlap": 150},
}

HF_EMBEDDING = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
LLM_METRICS = OllamaLLM(model="phi3:mini", keep_alive="60m")

def calculate_answer_quality_metrics(cfg_name: str):
    """
    Computes answer-quality metrics (ROUGE-L, relevance, faithfulness) for all
    test questions under a given chunking configuration and updates the
    `test_results.json` file in place.

    This function:
    - Loads previously generated RAG outputs for the selected chunk config.
    - Computes ROUGE-L between ground truth and generated answer.
    - Uses RAGAS to compute answer relevance and faithfulness using a local LLM.
    - Computes semantic relevance using cosine similarity (Temporary fix)
    - Computes faithfulness using a custom LLM-scored rubric (Temporary fix)
    - Inserts the measured metrics back into the results structure and saves.

    Args:
        cfg_name (str): The name of the chunking configuration whose results
            should be evaluated. Must be one of {"small", "medium", "large"}."""
    print(f"\n- Calculating Answer Quality metrics for chunking config '{cfg_name.upper()}'...")
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

    for i, result in enumerate(cfg_results, 1):
        ground_truth = result["ground_truth"]
        generated_answer = result["generated_answer"]
        if result["answerable"]:
            rougeL= _calculate_rouge_score(ground_truth=ground_truth, generated_answer=generated_answer)
            ans_relevance = _calculate_answer_relevance(result)
            ans_faithfulness = _calculate_faithfulness_custom(result)
        else:
            rougeL = None
            ans_relevance = None
            ans_faithfulness = None  
        new_result = result.copy()
        new_result["rougeL"] = rougeL
        new_result["answer_relevance"] = ans_relevance
        new_result["faithfulness"] = ans_faithfulness
        updated_cfg_results.append(new_result)
        print(f"- Calculated answer quality metrics for test question {i} ✔️")
    
    all_results[cfg_name] = updated_cfg_results

    # save updated results
    with open(TEST_RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n- Updated results, saved Answer Quality metrics, file at '{TEST_RESULTS_PATH.name}'. ")

# similarity - how much of the ground truth appears in the 
# generated answer
scorer_rouge = rouge_scorer.RougeScorer(["rougeL"])
def _calculate_rouge_score(ground_truth: str, generated_answer: str) -> float:
    """
    Computes the ROUGE-L F-measure score between the ground truth answer and the
    generated answer.

    ROUGE-L evaluates the longest common subsequence between the two texts,
    providing a measure of lexical similarity.

    Args:
        ground_truth (str): The expected (reference) answer.
        generated_answer (str): The system-generated answer.

    Returns:
        float: ROUGE-L F-measure score. Returns 0 if either string is empty"""
    if generated_answer.strip() == "" or ground_truth.strip() == "":
        return 0
    # rouge-l - longest common subsequence b/w the 2 texts
    # scorer = rouge_scorer.RougeScorer(["rougeL"])
    
    score_dict = scorer_rouge.score(ground_truth, generated_answer)
    
    # returns a Score(precision=.., recall=.., fmeasure=..) object
    rouge_l_score = score_dict["rougeL"]

    # use fmeasure (balances both recall, precision)
    return rouge_l_score.fmeasure

# answer relevance - check whether the generated answer is
# actually about the question i.e semantic alignment b/w
# question and answer
# faithfulness - check for hallucinations. Factual consistence
# of response relative to the retrieved context
# response considered faithful if all claims in it can be 
# supported by retrieved docs
def _calculate_answer_relevance(result: dict) -> float:
    """
    Computes semantic relevance using cosine similarity

    Returns:
        float or None
    """
    if not result["answerable"]:
        return None
    
    if result["generated_answer"].strip() == "":
        return 0.0
    
    relevance = _calculate_cosine_similarity(
    ground_truth=result["question"],
    generated_answer=result["generated_answer"])

    return relevance

FAITHFULNESS_PROMPT = """
You are evaluating whether an answer is faithful to the provided context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}

Score from 1 to 5:
1 = Completely hallucinated or unsupported
2 = Mostly unsupported with minor grounding
3 = Partially supported but contains some unsupported claims
4 = Mostly supported with minor hallucination
5 = Fully supported by the context, no hallucinations

Respond with only a single integer.
"""

def _calculate_faithfulness_custom(result: dict) -> float:
    if not result["answerable"]:
        return None
    
    if result["generated_answer"].strip() == "":
        return 0.0

    context = "\n\n".join(result["contexts"])

    prompt = FAITHFULNESS_PROMPT.format(
        context=context,
        question=result["question"],
        answer=result["generated_answer"],
    )

    raw = LLM_METRICS.invoke(prompt)
    
    # Extract first digit (1–5)
    for ch in raw:
        if ch in "12345":
            return float(ch)

    return 0.0
