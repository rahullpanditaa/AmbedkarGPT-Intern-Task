from lib.evaluation.eval import complete_evaluation_metrics, aggregate_results, evaluate_config


def main():
    complete_evaluation_metrics()
    aggregate_results("small")
    aggregate_results("medium")
    aggregate_results("large")


if __name__ == "__main__":
    main()
