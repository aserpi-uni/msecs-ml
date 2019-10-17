import pandas as pd
import pickle

from hw1.classifiers import classifier


def learn(X, y, algs, out):
    classes = pd.unique(y)

    for alg in algs:
        clf = classifier(alg)

        # X is too big to fit in memory: partial fits
        if len(X) * len(X.columns) > 50000 ** 2:
            for i in range(0, len(y), 5000):
                clf.partial_fit(X[i:i+5000], y[i:i+5000], classes=classes)

        # X fits in memory: direct computation
        else:
            clf.fit(X, y)

        pickle.dump(clf, f"{out}/{alg}.zip", protocol=pickle.HIGHEST_PROTOCOL)
