from lib.evaluation.eval import complete_evaluation_metrics, aggregate_results, evaluate_config

def main():
    evaluate_config(cfg_name="small")
    # complete_evaluation_metrics(cfg_name="small")
    # aggregate_results(cfg_name="small")


if __name__ == "__main__":
    main()
