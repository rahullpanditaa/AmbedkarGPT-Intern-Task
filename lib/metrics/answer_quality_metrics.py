from evaluation.evaluation import evaluate_config
from rouge_score import rouge_scorer
from ragas.metrics import answer_relevancy
from ragas import evaluate
from datasets import Dataset

def calculate_answer_quality_metrics():
    ...


# similarity - how much of the ground truth appears in the 
# generated answer
def calculate_rouge_score(ground_truth: str, generated_answer: str) -> float:
    if generated_answer == "" or ground_truth == "":
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
def calculate_answer_relevance(result: dict):
    data = {
        "question": [result["question"]],
        "answer": [result["generated_answer"]],
        "contexts": [result["contexts"]]
    }
    dataset = Dataset.from_dict(data)
    score = evaluate(dataset=dataset, metrics=[answer_relevancy])
    
    return score["answer_relevancy"]