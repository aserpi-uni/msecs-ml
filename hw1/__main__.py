import argparse
import json

from sklearn.preprocessing import MultiLabelBinarizer

from hw1.datasets import family_dataframe, feature_dataframe, one_hot_encode
from hw1.evaluation import evaluate, print_heatmap


def family_classification(directory, directory_results, algorithms, keys):
    # Get a DataFrame containing only malwares in large families
    families = family_dataframe(directory).groupby("family").filter(lambda f: len(f) > 20) \
        .set_index("sha256").sort_index()

    features = feature_dataframe(families.index.values, directory)
    if keys:
        features = features[keys]

    # Replace NaNs with empty list
    for col in features:
        for row in features.loc[features[col].isna(), col].index:
            features[col][row] = []

    # Preprocess datasets
    labels = families.family.unique()
    X = one_hot_encode(MultiLabelBinarizer(), features, features.columns)
    y = families.family.ravel()

    scores, confusion_matrices = evaluate(X, y, algorithms)

    # Save scores
    with open("{}/scores".format(directory_results), "w") as fout:
        json.dump(scores, fout, indent=4, sort_keys=True)

    # Print confusion matrices
    for alg in scores:
        print_heatmap(confusion_matrices[alg], alg, directory_results, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--classifier")
    parser.add_argument("--drebin")
    parser.add_argument("--output")

    # Optional arguments
    parser.add_argument("--algorithms", nargs="*")
    parser.add_argument("--features", nargs="*")

    args = vars(parser.parse_args())
    if not args["algorithms"]:
        args["algorithms"] = ["bernoulli", "random_forest", "svc", "linear_svc"]

    if args["classifier"] == "malware":
        pass  # TODO
    elif args["classifier"] == "family":
        family_classification(args["drebin"], args["output"], args["algorithms"], args["features"])
    else:
        raise KeyError
