import time
from rouge_score import rouge_scorer
from ragas.metrics import answer_relevancy, faithfulness
from ragas import evaluate
from datasets import Dataset
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings



CHUNK_CONFIGS = {
    "small":  {"chunk_size": 250, "chunk_overlap": 150},
    "medium": {"chunk_size": 550, "chunk_overlap": 150},
    "large":  {"chunk_size": 900, "chunk_overlap": 150},
}

HF_EMBEDDING = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# LLM = OllamaLLM(model="mistral")
LLM = OllamaLLM(model="deepseek-r1:1.5b")

def calculate_answer_quality_metrics(results: list[dict]):
    # results: list[dict] = evaluate_config(cfg_name=config_name.lower(), config=CHUNK_CONFIGS[config_name.lower()])

    results_with_answer_quality_metrics = []

    for result in results:
        ground_truth = result["ground_truth"]
        generated_answer = result["generated_answer"]
        rougeL = _calculate_rouge_score(ground_truth=ground_truth,
                                        generated_answer=generated_answer) if result["answerable"] else None
        
        ans_relevance = _calculate_answer_relevance(result=result) if result["answerable"] else None
        
        # ans_faithfulness = _calculate_answer_faithfulness(result=result) if result["answerable"] else None

        # temporarily disable
        ans_faithfulness = None

        new_result = result.copy()
        new_result["rougeL"] = rougeL
        new_result["answer_relevance"] = ans_relevance
        new_result["faithfulness"] = ans_faithfulness
        results_with_answer_quality_metrics.append(new_result)
    
    return results_with_answer_quality_metrics



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
def _calculate_answer_relevance(result: dict) -> float:
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
                      metrics=[answer_relevancy], 
                      llm=LLM, 
                      embeddings=HF_EMBEDDING,
                      show_progress=True)

    time.sleep(15.0)
    
    # return float(scores["answer_relevancy"])
    return float(scores.scores[0]["answer_relevancy"])

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
#                       show_progress=True)

#     time.sleep(15.0)

#     # return float(scores["faithfulness"])
#     return float(scores.scores[0]["faithfulness"])


