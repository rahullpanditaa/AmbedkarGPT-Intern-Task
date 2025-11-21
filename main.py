from lib.evaluation.eval import complete_evaluation_metrics, aggregate_results, evaluate_config

import ragas


def main():
    complete_evaluation_metrics(cfg_name="large")
    aggregate_results(cfg_name="large")


if __name__ == "__main__":
    main()
