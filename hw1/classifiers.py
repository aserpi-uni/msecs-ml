from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC, LinearSVC


def classifier(alg):
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
    return clf
