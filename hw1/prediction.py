import os
import pickle
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

from hw1.datasets import feature_dataframe, one_hot_encode


def predict(directory, directory_results, algorithms, keys):
    features = feature_dataframe(sorted(os.listdir(f"{directory}/feature_vectors")), directory, keys)
    X = one_hot_encode(MultiLabelBinarizer, features, features.columns)
    del features

    for alg in algorithms:
        classifier = pickle.load(f"{directory}/classifiers/{alg}")
        pd.Series(classifier.predict(X)).to_pickle(f"{directory_results}/{alg}.zip")
