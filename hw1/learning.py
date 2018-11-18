import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC, LinearSVC


def learn(X, y, algs, out):
    classes = y.unique()

    for alg in algs:
        if "bernoulli" == alg:
            clf = BernoulliNB()
        elif "random_forest" == alg:
            clf = RandomForestClassifier(n_estimators=20, verbose=2)
        elif "svc" == alg:
            clf = SVC(verbose=2)
        elif "linear_svc" == alg:
            clf = LinearSVC(max_iter=5000, verbose=2)
        else:
            raise KeyError("Unknown algorithm: {}".format(alg))

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
