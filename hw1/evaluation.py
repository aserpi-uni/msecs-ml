from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict, KFold

from hw1.classifiers import classifier


def evaluate(x, y, algs, classes=None):
    if not classes:
        classes = pd.unique(y)
    confusion_matrices = {}
    scores = {}

    for alg in algs:
        clf, partial_fit = classifier(alg)

        # X is too big to fit in memory and the algorithm supports partial fits
        pred = []
        if partial_fit and len(x) * len(x.columns) > 350000000:
            print(f"Evaluating {alg}...")

            kf = KFold(n_splits=5, shuffle=True)
            for idx_kfold, (train_index, test_index) in enumerate(kf.split(x)):
                print(f"KFold iteration {idx_kfold + 1} of 5...")
                x_train, x_test = x.iloc[train_index], x.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y[test_index]
                batches = int(len(x_train) / 645)

                for idx in range(0, batches):
                    print(f"Batch {idx + 1} of {batches}...", end="\t")
                    idx_start = int((len(x_train) / batches * idx))
                    idx_finish = int(len(x_train) / batches * (idx + 1)) - 1
                    print("Slicing...", end="\t")

                    x_train_batch = x_train[idx_start:idx_finish]
                    y_train_batch = y_train[idx_start:idx_finish]

                    print(f"Fitting...", end="\t")
                    clf.partial_fit(x_train_batch, y_train_batch, classes=classes)
                    print(f"Done.")

                print(f"Predicting...", end="\t")
                pred.append(pd.Series(clf.predict(x_test)))
                print(f"Done.")

            pred = pd.concat(pred)

        else:
            print(f"Evaluating {alg}...", end="\t")
            pred = cross_val_predict(clf, x, y, cv=5, n_jobs=-1, pre_dispatch="n_jobs", verbose=1)
            print("Done.")

        scores[alg], confusion_matrices[alg] = __evaluate_algorithm(y, pred)
    return scores, confusion_matrices


def print_heatmap(cm, classifier, algorithm, directory, labels=None):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalise confusion matrix

    plt.figure(figsize=(34, 30), dpi=400)
    ax = sn.heatmap(cm, annot=True, cmap="Blues", fmt=".2f", square=True, vmin=0, vmax=1,
                    xticklabels=labels, yticklabels=labels)

    ax.set_xlabel("Predicted classes")
    ax.set_ylabel("True classes")

    ax.get_figure().savefig(f"{directory}/heatmap_{classifier}_{algorithm}.pdf", dpi=400, format="pdf")
    plt.clf()


def __evaluate_algorithm(y_true, y_pred, confusion_matrix_labels=None):
    scores = {"accuracy": accuracy_score(y_true, y_pred),
              "classification_report": classification_report(y_true, y_pred, output_dict=True)}
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=confusion_matrix_labels))

    return scores, cm


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = []
plt.rcParams["font.size"] = 75
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = [r"\usepackage{lmodern}"]
