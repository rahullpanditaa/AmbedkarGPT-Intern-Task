from evaluation.evaluation import evaluate_config
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.metrics import BleuScore
from ragas import evaluate
from datasets import Dataset

CHUNK_CONFIGS = {
    "small":  {"chunk_size": 250, "chunk_overlap": 150},
    "medium": {"chunk_size": 550, "chunk_overlap": 150},
    "large":  {"chunk_size": 900, "chunk_overlap": 150},
}

def calculate_semantic_metrics(config_name: str):
    results = evaluate_config(cfg_name=config_name.lower(),
                              config=CHUNK_CONFIGS[config_name.lower()])
    
    results_with_semantic_metrics = []


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
    
    # data = {
    #     "reference": ground_truth,
    #     "response": generated_answer
    # }

    # dataset = Dataset.from_dict(data)
    # result = evaluate(dataset=dataset, metrics=[])