import argparse
import json
import os
import pandas as pd
import pickle

from sklearn.preprocessing import MultiLabelBinarizer

from hw1.datasets import family_dataframe, feature_dataframe, one_hot_encode
from hw1.evaluation import evaluate, print_heatmap
from hw1.learning import learn
from hw1.prediction import predict


def family_classification(action, directory, directory_results, algorithms, keys):
    try:
        with open(f"{directory}/X_family.zip", "rb") as fin_X, \
                open(f"{directory}/y_family.zip", "rb") as fin_y, \
                open(f"{directory}/labels_family.zip", "rb") as fin_labels:
            X = pickle.load(fin_X)
            y = pickle.load(fin_y)
            labels = pickle.load(fin_labels)

    except (FileNotFoundError, TypeError):
        # Get a DataFrame containing only malware in large families
        families = family_dataframe(directory).groupby("family").filter(lambda f: len(f) > 20) \
            .set_index("sha256").sort_index()

        features = feature_dataframe(families.index.values, directory, keys)

        # Preprocess datasets
        labels = pd.unique(families.family)
        X = one_hot_encode(MultiLabelBinarizer, features, features.columns)
        y = families.family.ravel()

        # Delete unnecessary data structures
        del families
        del features

        with open(f"{directory}/X_family.zip", "wb") as fout_X:
            pickle.dump(X, fout_X, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"{directory}/y_family.zip", "wb") as fout_y:
            pickle.dump(y, fout_y, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"{directory}/labels_family.zip", "wb") as fout_labels:
            pickle.dump(labels, fout_labels, protocol=pickle.HIGHEST_PROTOCOL)

    if action == "evaluate":
        scores, confusion_matrices = evaluate(X, y, algorithms)

        # Save scores
        with open(f"{directory_results}/scores", "w") as fout:
            json.dump(scores, fout, indent=4, sort_keys=True)

        # Print confusion matrices
        for alg in scores:
            print_heatmap(confusion_matrices[alg], alg, directory_results, labels)

    elif action == "learn":
        learn(X, y, algorithms, directory_results)

    else:
        raise KeyError


def malware_classification(action, directory, directory_results, algorithms, keys):
    try:
        with open(f"{directory}/X_malware.zip", "rb") as fin_X, \
                open(f"{directory}/y_malware.zip", "rb") as fin_y:
            X = pickle.load(fin_X)
            y = pickle.load(fin_y)

    except (FileNotFoundError, TypeError):
        malwares = family_dataframe(directory).set_index("sha256").sort_index()
        features = feature_dataframe(sorted(os.listdir(f"{directory}/feature_vectors")), directory, keys)

        # Preprocess datasets
        malwares["family"] = "Malware"
        malwares = malwares.rename(columns={"family": "malware"}).reindex(features.index, fill_value="Safe")
        X = one_hot_encode(MultiLabelBinarizer, features, features.columns)
        y = malwares.malware.ravel()

        # Delete unnecessary data structures
        del malwares
        del features

        with open(f"{directory}/X_malware.zip", "wb") as fout_X:
            pickle.dump(X, fout_X, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"{directory}/y_malware.zip", "wb") as fout_y:
            pickle.dump(y, fout_y, protocol=pickle.HIGHEST_PROTOCOL)

    if action == "evaluate":
        scores, confusion_matrices = evaluate(X, y, algorithms)

        # Save scores
        with open(f"{directory_results}/scores", "w") as fout:
            json.dump(scores, fout, indent=4, sort_keys=True)

        # Print confusion matrices
        for alg in scores:
            print_heatmap(confusion_matrices[alg], alg, directory_results, ["Safe", "Malware"])

    elif action == "learn":
        learn(X, y, algorithms, directory_results)

    else:
        raise KeyError


def main(action, classifier, drebin, out, algs=None, feats=None):
    if not algs:
        algs = ["bernoulli", "random_forest", "svc", "linear_svc"]

    if action == "predict":
        predict(drebin, out, algs, feats)
    elif action == "evaluate" or action == "learn":
        if classifier == "malware":
            malware_classification(action, drebin, out, algs, feats)
        elif classifier == "family":
            family_classification(action, drebin, out, algs, feats)
        else:
            raise KeyError
    else:
        raise KeyError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--action")
    parser.add_argument("--classifier")
    parser.add_argument("--drebin")
    parser.add_argument("--output")

    # Optional arguments
    parser.add_argument("--algorithms", nargs="*")
    parser.add_argument("--features", nargs="*")

    args = vars(parser.parse_args())

    main(args["action"], args["classifier"], args["drebin"], args["output"], args["algorithms"], args["features"])
