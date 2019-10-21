import argparse
import json

from hw1.datasets import labels, test_dataset, training_datasets
from hw1.evaluation import evaluate, print_heatmap
from hw1.learning import learn
from hw1.prediction import predict


def main(action, dataset_dir, out_dir, algs=None, cache=False):
    if not algs:
        algs = ["bernoulli", "random_forest", "svc", "linear_svc"]

    if action == "evaluate":
        X, y = training_datasets(dataset_dir, cache)
        scores, confusion_matrices = evaluate(X, y, algs)
        with open(f"{out_dir}/scores", "w") as fout:
            json.dump(scores, fout, indent=4, sort_keys=True)
        for alg in scores:
            print_heatmap(confusion_matrices[alg], alg, out_dir, labels())

    elif action == "learn":
        X, y = training_datasets(dataset_dir, cache)
        learn(X, y, algs, out_dir)

    elif action == "predict":
        X = test_dataset(dataset_dir)
        predict(X, dataset_dir, out_dir, algs)

    else:
        raise KeyError(f"Unknown action '{action}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("action", choices=["evaluate", "learn", "predict"])
    parser.add_argument("dataset-directory")
    parser.add_argument("output-directory")

    # Optional arguments
    parser.add_argument("--algorithms", nargs="*")
    parser.add_argument("--no-cache", const=True, default=False, nargs="?")

    args = vars(parser.parse_args())

    main(args["action"], args["dataset_directory"], args["output_directory"], args["algorithms"], not args["no_cache"])
