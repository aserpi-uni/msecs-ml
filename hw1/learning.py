import pickle

from hw1.classifiers import classifier


def learn(X, y, algs, out):
    classes = y.unique()

    for alg in algs:
        clf = classifier(alg)

        # X is too big to fit in memory: partial fits
        i = 0
        if len(X)*len(X.columns) > 50000**2:
            while i < len(y):
                clf.partial_fit(X[i:i+5000], y[i:i+5000], classes=classes)
                i += 5000

        # X fits in memory: direct computation
        else:
            clf.fit(X, y)

        pickle.dump(clf, "{}/{}.zip".format(out, alg), protocol=pickle.HIGHEST_PROTOCOL)
