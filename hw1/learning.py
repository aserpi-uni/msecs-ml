import math
import pandas as pd
import pickle

from hw1.classifiers import classifier


def learn(X, y, algs, out):
    classes = pd.unique(y)

    for alg in algs:
        print(f"\nFitting {alg}...", end="\t")
        clf = classifier(alg)

        # X is too big to fit in memory: partial fits
        if len(X) * len(X.columns) > 50000:
            iterations = math.ceil(len(X) / 5000)
            print("")
            for i in range(0, len(y), 5000):
                print(f"Fitting partition {i + 1} of {iterations}...", end="\t")
                clf.partial_fit(X[i:i+4999], y[i:i+4999], classes=classes)
                print("Done.")

        # X fits in memory: direct computation
        else:
            clf.fit(X, y)
            print("Done.")

        pickle.dump(clf, f"{out}/{alg}.zip", protocol=pickle.HIGHEST_PROTOCOL)
