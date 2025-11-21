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
# LLM = OllamaLLM(model="mistral")
LLM = OllamaLLM(model="deepseek-r1:1.5b")

RAGAS_RUN_CONFIG = RunConfig(   
    timeout=120,     # seconds per operation
)

def calculate_answer_quality_metrics(cfg_name: str):
    # results: list[dict] = evaluate_config(cfg_name=config_name.lower(), config=CHUNK_CONFIGS[config_name.lower()])

    if TEST_RESULTS_PATH.exists():
        with open(TEST_RESULTS_PATH, "r") as f:
            previous_test_results = json.load(f)
    else:
        previous_test_results = {}

    # previous_test_results is either an empty dict
    # OR, it is a dict with at least 1 key value pair
    # where key=config_name (small, medium, or large)
    # value = list[dict] -> results per test_question so far

    if cfg_name in previous_test_results.keys():
        cfg_results_so_far = previous_test_results.get(cfg_name)
    
    results_with_answer_quality_metrics = []

    for result in cfg_results_so_far:
        ground_truth = result["ground_truth"]
        generated_answer = result["generated_answer"]
        rougeL = _calculate_rouge_score(ground_truth=ground_truth,
                                        generated_answer=generated_answer) if result["answerable"] else None
        
        # ans_relevance_and_faithfulness = _calculate_answer_relevance(result=result) if result["answerable"] else None
        
        # ans_faithfulness = _calculate_answer_faithfulness(result=result) if result["answerable"] else None

        ans = _calculate_answer_relevance_and_faithfulness(result=result) if result["answerable"] else None


        # temporarily disable
        # ans_faithfulness = None

        new_result = result.copy()
        new_result["rougeL"] = rougeL
        new_result["answer_relevance"] = ans[0]["answer_relevancy"] if ans else None
        new_result["faithfulness"] = ans[0]["faithfulness"] if ans else None
        results_with_answer_quality_metrics.append(new_result)
    
    updated_results = {}
    updated_results[cfg_name] = results_with_answer_quality_metrics

    # save updated results
    with open(TEST_RESULTS_PATH, "w") as f:
        json.dump(updated_results, f, indent=2)

    print(f"\n- Updated results, saved Answer Quality metrics, file at '{TEST_RESULTS_PATH.name}'. ")
    



# similarity - how much of the ground truth appears in the 
# generated answer
def _calculate_rouge_score(ground_truth: str, generated_answer: str) -> float:
    if generated_answer.strip() == "" or ground_truth.strip() == "":
        return 0
    # rouge-l - longest common subsequence b/w the 2 texts
    scorer = rouge_scorer.RougeScorer(["rougeL"])
    
    # remove stop words and punctuation ??
    score_dict = scorer.score(ground_truth, generated_answer)
    
    # returns a Score(precision=.., recall=.., fmeasure=..) object
    rouge_l_score = score_dict["rougeL"]

    # use fmeasure (balances both recall, precision)
    return rouge_l_score.fmeasure

# answer relevance - check whether the generated answer is
# actually about the question i.e semantic alignment b/w
# question and answer
def _calculate_answer_relevance_and_faithfulness(result: dict) -> list[dict]:
    if result["generated_answer"].strip() == "":
        return 0.0
    data = {
        "question": [result["question"]],
        "answer": [result["generated_answer"]],
        "contexts": [result["contexts"]]
    }
    dataset = Dataset.from_dict(data)
    # llm = OllamaLLM(model="deepseek-r1:1.5b")
    # llm = OllamaLLM(model="mistral")
    scores = evaluate(dataset=dataset, 
                      metrics=[answer_relevancy, faithfulness], 
                      llm=LLM, 
                      embeddings=HF_EMBEDDING,
                      run_config=RAGAS_RUN_CONFIG,
                      show_progress=False)

    # time.sleep(5.0)
    
    # return float(scores["answer_relevancy"])
    return scores.scores

# faithfulness - check for hallucinations. Factual consistence
# of response relative to the retrieved context
# response considered faithful if all claims in it can be 
# supported by retrieved docs
# def _calculate_answer_faithfulness(result: dict):
#     if result["generated_answer"].strip() == "":
#         return 0.0
#     data = {
#         "question": [result["question"]],
#         "answer": [result["generated_answer"]],
#         "contexts": [result["contexts"]]
#     }

#     dataset = Dataset.from_dict(data)
#     # llm = OllamaLLM(model="deepseek-r1:1.5b")
#     # llm = OllamaLLM(model="mistral")
#     scores = evaluate(dataset, 
#                       metrics=[faithfulness], 
#                       llm=LLM,
#                       embeddings=HF_EMBEDDING,
#                       run_config=RAGAS_RUN_CONFIG,
#                       show_progress=False)

#     # time.sleep(15.0)

#     # return float(scores["faithfulness"])
#     return float(scores.scores[0]["faithfulness"])


