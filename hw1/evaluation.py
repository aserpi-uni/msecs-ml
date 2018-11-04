import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn

from matplotlib import rc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC, LinearSVC


def evaluate(X, y, algs):
    confusion_matrices = {}
    scores = {}

    for alg in algs:
        if "bernoulli" == alg:
            clf = BernoulliNB()
        elif "random_forest" == alg:
            clf = RandomForestClassifier(n_estimators=20)
        elif "svc" == alg:
            clf = SVC()
        elif "linear_svc" == alg:
            clf = LinearSVC(max_iter=5000)
        else:
            raise KeyError("Unknown algorithm: {}".format(alg))

        print("Evaluating {}...".format(alg), end="\t")
        pred = cross_val_predict(clf, X, y, cv=5)
        scores[alg], confusion_matrices[alg] = __evaluate_algorithm(y, pred)
        print("Done")

    return scores, confusion_matrices


def print_heatmap(cf, algorithm, directory, labels=None):
    plt.figure(figsize=(34, 30), dpi=400)
    sn.heatmap(cf.applymap(lambda cell: 0 if cell == 0 else np.log10(cell)), annot=True, square=True,
               xticklabels=labels, yticklabels=labels) \
        .get_figure().savefig("{}/heatmap_{}".format(directory, algorithm), dpi=400, format="pdf")
    plt.clf()


def __evaluate_algorithm(y_true, y_pred, confusion_matrix_labels=None):
    scores = {"accuracy": accuracy_score(y_true, y_pred),
              "classification_report": classification_report(y_true, y_pred, output_dict=True)}
    cf = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=confusion_matrix_labels))

    return scores, cf


rc("font", **{"family": "Latin Modern Roman", "size": 26})
