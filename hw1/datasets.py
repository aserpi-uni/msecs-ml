import pandas as pd
import pickle
from sklearn.preprocessing import MultiLabelBinarizer


def labels():
    return ["clang-HIGH", "clang-LOW", "icc-HIGH", "icc-LOW", "gcc-HIGH", "gcc-LOW"]


def training_datasets(dataset_dir, cache=False):
    try:
        with open(f"{dataset_dir}/X.zip", "rb") as fin_X, \
                open(f"{dataset_dir}/y.zip", "rb") as fin_y:
            X = pickle.load(fin_X)
            y = pickle.load(fin_y)

    except (FileNotFoundError, TypeError):
        with open(f"{dataset_dir}/dataset.json") as fin:  # TODO
            data = pd.read_json(fin)
        X = MultiLabelBinarizer().fit_transform(data["instructions"])
        y = data.apply(lambda fun: f"{fun['compiler']}-{fun['optimization']}", axis=1)

        if cache:
            with open(f"{dataset_dir}/X.zip", "wb") as fout_X, \
                    open(f"{dataset_dir}/y.zip", "wb") as fout_y:
                pickle.dump(X, fout_X)
                pickle.dump(y, fout_y)

    return X, y


def test_dataset(dataset_dir):
    try:
        with open(f"{dataset_dir}/X.zip", "rb") as fin_X:
            X_train = pickle.load(fin_X)
    except (FileNotFoundError, TypeError):
        with open(f"{dataset_dir}/dataset.json") as fin:  # TODO
            X_train = MultiLabelBinarizer().fit_transform(pd.read_json(fin)["instructions"])
    
    with open(f"{dataset_dir}/dataset.json") as fin:  # TODO
        X = MultiLabelBinarizer(classes=X_train.classes_).fit_transform(pd.read_json(fin)["instructions"])

    return X