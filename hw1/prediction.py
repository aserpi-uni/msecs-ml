from contextlib import suppress
import os
import pickle


def predict(X, dataset_dir, results_dir, algorithms):
    for alg in algorithms:
        pred = pickle.load(f"{dataset_dir}/classifiers/{alg}").predict(X)

        pred.to_pickle(f"{results_dir}/{alg}.zip")  # TODO: remove
        with suppress(FileNotFoundError):
            os.remove(f"{results_dir}/{alg}.csv")
        with open(f"{results_dir}/{alg}.csv", "a") as fout:
            for i in pred:
                print(pred[i].replace("-", ","), file=fout)
