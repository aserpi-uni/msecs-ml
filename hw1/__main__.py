import argparse
import json

from hw1.datasets import labels, test_dataset, training_datasets
from hw1.evaluation import evaluate, print_heatmap
from hw1.learning import learn
from hw1.prediction import consolidate, predict


def main(action, classifier, dataset_dir, out_dir, algs=None, cache=False):  # TODO create directories
    if not algs:
        algs = ["bernoulli", "random_forest", "svc", "linear_svc"]

    if action == "consolidate":
        if len(algs) != 2:
            raise ValueError("Specify two algorithms to consolidate"
                             "(the first for the compiler, the second for the optimization level).")
        consolidate(algs[0], algs[1], dataset_dir, out_dir)

    elif action == "evaluate":
        X, y = training_datasets(dataset_dir, classifier, cache)
        scores, confusion_matrices = evaluate(X, y, algs, classes=labels(classifier))
        with open(f"{out_dir}/scores/{classifier}", "w") as fout:
            json.dump(scores, fout, indent=4, sort_keys=True)
        for alg in scores:
            print_heatmap(confusion_matrices[alg], alg, out_dir, labels(classifier))

    elif action == "learn":
        X, y = training_datasets(dataset_dir, classifier, cache)
        learn(X, y, classifier, algs, out_dir)

    elif action == "predict":
        X = test_dataset(dataset_dir)
        predict(X, classifier, algs, dataset_dir, out_dir)

    else:
        raise KeyError(f"Unknown action '{action}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("action", choices=["consolidate", "evaluate", "learn", "predict"])
    parser.add_argument("classifier", choices=["compiler", "opt"])
    parser.add_argument("dataset-directory")
    parser.add_argument("output-directory")

    # Optional arguments
    parser.add_argument("--algorithms", nargs="*")
    parser.add_argument("--no-cache", const=True, default=False, nargs="?")

    args = vars(parser.parse_args())

    main(args["action"],
         args["classifier"],
         args["dataset_directory"],
         args["output_directory"],
         args["algorithms"],
         not args["no_cache"])
