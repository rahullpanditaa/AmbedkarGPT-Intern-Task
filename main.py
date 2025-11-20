from lib.evaluation.eval import complete_evaluation, aggregate_results

def main():
    complete_evaluation(cfg_name="small")
    aggregate_results(cfg_name="small")


if __name__ == "__main__":
    main()
