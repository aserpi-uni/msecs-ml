import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

from matplotlib import rc
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict, KFold

from hw1.classifiers import classifier


def evaluate(X, y, algs):
    classes = pd.unique(y)
    confusion_matrices = {}
    scores = {}

    for alg in algs:
        clf = classifier(alg)

        # X is too big to fit in memory: partial fits
        pred = []
        if len(X)*len(X.columns) > 50000**2:
            batches = 39
            print(f"Evaluating {alg}")

            kf = KFold(n_splits=5, shuffle=True)
            for idx_kfold, (train_index, test_index) in enumerate(kf.split(X)):
                print(f"\nKFold iteration {idx_kfold}")
                x_train, x_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y[train_index], y[test_index]

                for idx in range(0, batches - 1):
                    idx_start = int((len(X)/batches)*idx)
                    idx_finish = int((len(X)/batches)*(idx + 1))
                    print(f"Slicing batch {idx}: from {idx_start} to {idx_finish}")

                    X_train_batch = x_train.iloc[idx_start:idx_finish - 1]
                    y_train_batch = y_train[idx_start:idx_finish - 1]

                    if len(y_train_batch) == 0:
                        continue

                    print(f"Fitting batch {idx}")

                    clf.partial_fit(X_train_batch, y_train_batch, classes=classes)

                print(f"Predicting KFold {idx_kfold}")
                pred.append(pd.Series(clf.predict(x_test)))

            pred = pd.concat(pred)

        # X fits in memory: direct computation
        else:
            print(f"Evaluating {alg}...", end="\t")
            pred = cross_val_predict(clf, X, y, cv=5, n_jobs=1, pre_dispatch="n_jobs", verbose=2)

        scores[alg], confusion_matrices[alg] = __evaluate_algorithm(y, pred)
        print("Done")

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
