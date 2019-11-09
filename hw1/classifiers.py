from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC


def classifier(alg):
    if "naive_bayes" == alg:
        clf = MultinomialNB()
        partial_fit = False
    elif "random_forest" == alg:
        clf = RandomForestClassifier(n_estimators=100, verbose=1)
        partial_fit = False
    elif "svc" == alg:
        clf = SVC(gamma='scale', verbose=1)
        partial_fit = False
    elif "linear_svc" == alg:
        clf = LinearSVC(verbose=1)
        partial_fit = False
    else:
        raise KeyError(f"Unknown algorithm: {alg}")

    return clf, partial_fit
