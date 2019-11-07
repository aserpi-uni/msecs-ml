import argparse
import errno
import json
import os

from hw1.datasets import labels, test_dataset, train_datasets
from hw1.evaluation import evaluate, print_heatmap
from hw1.learning import learn
from hw1.prediction import consolidate, predict


def main(action, classifier, dataset_dir, out_dir, algs=None, cache=False, truncate=None):
    if not algs:
        algs = ["naive_bayes", "random_forest", "svc", "linear_svc"]

    try:
        os.makedirs(out_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    if action == "consolidate":
        if len(algs) != 2:
            raise ValueError("Specify two algorithms to consolidate"
                             "(the first for the compiler, the second for the optimization level).")
        consolidate(algs[0], algs[1], dataset_dir, out_dir)

    elif action == "evaluate":
        x, y = train_datasets(dataset_dir, classifier, cache=False, truncate=truncate)
        scores, confusion_matrices = evaluate(x, y, algs, classes=labels(classifier))
        with open(f"{out_dir}/scores_{classifier}.json", "w") as fout:
            json.dump(scores, fout, indent=4, sort_keys=True)
        for alg in scores:
            print_heatmap(confusion_matrices[alg], classifier, alg, out_dir, labels=labels(classifier))

    elif action == "learn":
        x, y = train_datasets(dataset_dir, classifier, cache=cache, truncate=truncate)
        learn(x, y, classifier, algs, out_dir)

    elif action == "predict":
        x = test_dataset(dataset_dir, truncate=truncate)
        predict(x, classifier, algs, dataset_dir, out_dir)

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
    parser.add_argument("--algorithms", nargs="*", choices=["naive_bayes", "svc", "linear_svc", "random_forest"])
    parser.add_argument("--no-cache", const=True, default=False, nargs="?")
    parser.add_argument("--truncate", nargs="?", type=int)

    args = vars(parser.parse_args())

    main(args["action"],
         args["classifier"],
         args["dataset-directory"],
         args["output-directory"],
         args["algorithms"],
         not args["no_cache"],
         args["truncate"])
