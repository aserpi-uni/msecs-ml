from matplotlib import pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict, KFold

from hw1.classifiers import classifier


def evaluate(X, y, algs, classes=None):
    if not classes:
        classes = pd.unique(y)
    confusion_matrices = {}
    scores = {}

    for alg in algs:
        clf, partial_fit = classifier(alg)

        # X is too big to fit in memory and the algorithm supports partial fits
        pred = []
        if partial_fit and len(X) * len(X.columns) > 350000000:
            batches = int(len(X) / 645)
            print(f"Evaluating {alg}...")

            kf = KFold(n_splits=5, shuffle=True)
            for idx_kfold, (train_index, test_index) in enumerate(kf.split(X)):
                print(f"KFold iteration {idx_kfold + 1} of 5...")
                x_train, x_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y[train_index], y[test_index]

                for idx in range(0, batches):
                    print(f"Batch {idx + 1} of {batches}...", end="\t")
                    idx_start = int((len(X) / batches * idx))
                    idx_finish = int(len(X) / batches * (idx + 1)) - 1
                    print("Slicing...", end="\t")

                    X_train_batch = x_train.iloc[idx_start:idx_finish]
                    y_train_batch = y_train[idx_start:idx_finish]

                    print(f"Fitting...", end="\t")
                    clf.partial_fit(X_train_batch, y_train_batch, classes=classes)
                    print(f"Done.")

                print(f"Predicting...", end="\t")
                pred.append(pd.Series(clf.predict(x_test)))
                print(f"Done.")

            pred = pd.concat(pred)

        else:
            print(f"Evaluating {alg}...", end="\t")
            pred = cross_val_predict(clf, X, y, cv=5, n_jobs=1, pre_dispatch="n_jobs", verbose=2)
            print("Done.")

        scores[alg], confusion_matrices[alg] = __evaluate_algorithm(y, pred)
    return scores, confusion_matrices


def print_heatmap(cf, algorithm, directory, labels=None):
    plt.figure(figsize=(34, 30), dpi=400)
    sn.heatmap(cf.applymap(lambda cell: 0 if cell == 0 else np.log10(cell)), annot=True, square=True,
               xticklabels=labels, yticklabels=labels) \
        .get_figure().savefig(f"{directory}/heatmap_{algorithm}", dpi=400, format="pdf")
    plt.clf()


def __evaluate_algorithm(y_true, y_pred, confusion_matrix_labels=None):
    scores = {"accuracy": accuracy_score(y_true, y_pred),
              "classification_report": classification_report(y_true, y_pred, output_dict=True)}
    cf = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=confusion_matrix_labels))

    return scores, cf


rc("font", **{"family": "Latin Modern Roman", "size": 26})
