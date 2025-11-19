from evaluation.evaluation import evaluate_config
from rouge_score import rouge_scorer
def calculate_answer_quality_metrics():
    ...

def calculate_rouge_score(ground_truth: str, generated_answer: str) -> float:
    # rouge-l - longest common subsequence b/w the 2 texts
    scorer = rouge_scorer.RougeScorer(["rougeL"])

    # similarity - how much of the ground truth appears in the 
    # generated answer
    # remove stop words and punctuation ??
    score_dict = scorer.score(ground_truth, generated_answer)
    
    # returns a Score(precision=.., recall=.., fmeasure=..) object
    rouge_l_score = score_dict["rougeL"]

    # use fmeasure (balances both recall, precision)
    return rouge_l_score.fmeasure