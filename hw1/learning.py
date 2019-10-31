import pandas as pd
import pickle
import os

from hw1.classifiers import classifier


def learn(x, y, classifier_type, algs, out, classes=None):
    if not classes:
        classes = pd.unique(y)

    for alg in algs:
        print(f"\nFitting {alg}...", end="\t")
        clf, partial_fit = classifier(alg)

        # X is too big to fit in memory and the algorithm supports partial fits
        if partial_fit and len(x) * len(x.columns) > 350000000:
            try:
                with open(f"{out}/{classifier_type}_{alg}_batch.zip", "rb") as fin_batch, \
                        open(f"{out}/{classifier_type}_{alg}.zip", "rb") as fin_classifier:
                    (batches, first_batch) = pickle.load(fin_batch)
                    clf = pickle.load(fin_classifier)
            except (FileNotFoundError, TypeError):
                batches = int(len(x) / 645)
                first_batch = 0
            print("")

            for i in range(first_batch, batches):
                start_idx = int(len(x) / batches * i)
                end_idx = int(len(x) / batches * (i + 1)) - 1
                print(f"Fitting partition {i + 1} of {batches}...", end="\t")
                clf.partial_fit(x[start_idx:end_idx], y[start_idx:end_idx], classes=classes)

                with open(f"{out}/{classifier_type}_{alg}.zip", "wb") as fout_classifier:
                    pickle.dump(clf, fout_classifier, protocol=pickle.HIGHEST_PROTOCOL)
                with open(f"{out}/{classifier_type}_{alg}_batch.zip", "wb") as fout_batch:
                    pickle.dump((batches, i + 1), fout_batch, protocol=pickle.HIGHEST_PROTOCOL)
                print("Done.")

            os.remove(f"{out}/{classifier_type}_{alg}_batch.zip")

        else:
            clf.fit(x, y)
            with open(f"{out}/{classifier_type}_{alg}.zip", "wb") as fout_classifier:
                pickle.dump(clf, fout_classifier, protocol=pickle.HIGHEST_PROTOCOL)
            print("Done.")
