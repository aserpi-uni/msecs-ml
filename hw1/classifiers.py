from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC, LinearSVC


def classifier(alg):
    if "bernoulli" == alg:
        clf = BernoulliNB()
        partial_fit = True
    elif "random_forest" == alg:
        clf = RandomForestClassifier(n_estimators=20, verbose=2)
        partial_fit = False
    elif "svc" == alg:
        clf = SVC(verbose=2)
        partial_fit = False
    elif "linear_svc" == alg:
        clf = LinearSVC(max_iter=5000, verbose=2)
        partial_fit = False
    else:
        raise KeyError(f"Unknown algorithm: {alg}")

    return clf, partial_fit
