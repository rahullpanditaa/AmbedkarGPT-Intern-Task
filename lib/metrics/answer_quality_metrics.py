import json
from pathlib import Path
from rouge_score import rouge_scorer
from ragas.metrics import answer_relevancy, faithfulness
from ragas import evaluate
from datasets import Dataset
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.run_config import RunConfig

import os
os.environ["OPENAI_API_KEY"] = "0"   # forces ragas to NOT use OpenAI embeddings
TEST_RESULTS_PATH = Path(__file__).parent.parent.parent.resolve() / "data" / "test_results.json"

CHUNK_CONFIGS = {
    "small":  {"chunk_size": 250, "chunk_overlap": 150},
    "medium": {"chunk_size": 550, "chunk_overlap": 150},
    "large":  {"chunk_size": 900, "chunk_overlap": 150},
}

HF_EMBEDDING = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
LLM_METRICS = OllamaLLM(model="phi3:mini", keep_alive="120m")

RAGAS_RUN_CONFIG = RunConfig(   
    timeout=120,     # seconds per operation
)

def calculate_answer_quality_metrics(cfg_name: str):
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

    for result in cfg_results:
        ground_truth = result["ground_truth"]
        generated_answer = result["generated_answer"]
        rougeL = _calculate_rouge_score(ground_truth=ground_truth,
                                        generated_answer=generated_answer) if result["answerable"] else None  
        if result["answerable"]:
            ans_relevance, ans_faithfulness = _calculate_answer_relevance_and_faithfulness(result)
        else:
            ans_relevance, ans_faithfulness = None, None       
        new_result = result.copy()
        new_result["rougeL"] = rougeL
        new_result["answer_relevance"] = ans_relevance
        new_result["faithfulness"] = ans_faithfulness
        updated_cfg_results.append(new_result)
    
    all_results[cfg_name] = updated_cfg_results

    # save updated results
    with open(TEST_RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n- Updated results, saved Answer Quality metrics, file at '{TEST_RESULTS_PATH.name}'. ")

# similarity - how much of the ground truth appears in the 
# generated answer
scorer_rouge = rouge_scorer.RougeScorer(["rougeL"])
def _calculate_rouge_score(ground_truth: str, generated_answer: str) -> float:
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
def _calculate_answer_relevance_and_faithfulness(result: dict) -> tuple[float, float]:
    if result["generated_answer"].strip() == "":
        return 0.0

    data = {
        "question": [result["question"]],
        "answer": [result["generated_answer"]],
        "contexts": [result["contexts"]]
    }
    dataset = Dataset.from_dict(data)
    scores = evaluate(dataset=dataset, 
                      metrics=[answer_relevancy, faithfulness], 
                      llm=LLM_METRICS, 
                      embeddings=HF_EMBEDDING,
                      run_config=RAGAS_RUN_CONFIG,
                      show_progress=False)
    row = scores.scores[0]
    return float(row["answer_relevancy"]), float(row["faithfulness"])