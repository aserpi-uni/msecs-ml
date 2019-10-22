import pandas as pd
import pickle
from sklearn.preprocessing import MultiLabelBinarizer


def __X(data):
    for fun_idx, fun in enumerate(data):
        for instr_idx, instr in enumerate(fun):
            fun[instr_idx] = instr.split(" ")[0]


def labels(classifier):
    if classifier == "compiler":
        return ["clang", "icc", "gcc"]
    elif classifier == "opt":
        return ["H", "L"]


def training_datasets(dataset_dir, classifier, cache=False):
    try:
        with open(f"{dataset_dir}/X.zip", "rb") as fin_X, \
                open(f"{dataset_dir}/y_{classifier}.zip", "rb") as fin_y:
            X = pickle.load(fin_X)
            y = pickle.load(fin_y)

    except (FileNotFoundError, TypeError):
        data = pd.read_json(f"{dataset_dir}/train_dataset.jsonl", lines=True)
        mlb = MultiLabelBinarizer(sparse_output=True)
        X = mlb.fit_transform(__X(data["instructions"]))  # TODO
        y = data[classifier]

        if cache:
            with open(f"{dataset_dir}/X.zip", "wb") as fout_X, \
                    open(f"{dataset_dir}/y_{classifier}.zip", "wb") as fout_y, \
                    open(f"{dataset_dir}/features.zip", "wb") as fout_features:
                pickle.dump(X, fout_X, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(y, fout_y, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(mlb.classes_, fout_features, protocol=pickle.HIGHEST_PROTOCOL)

    return X, y


def test_dataset(dataset_dir):
    try:
        with open(f"{dataset_dir}/features.zip", "rb") as fin_features:
            features = pickle.load(fin_features)
    except (FileNotFoundError, TypeError):
        mlb = MultiLabelBinarizer(sparse_output=True)
        mlb.fit_transform(pd.read_json(f"{dataset_dir}/test_dataset_blind.jsonl", lines=True)["instructions"])  # TODO
        features = mlb.classes_
    
    with open(f"{dataset_dir}/train_dataset.jsonl") as fin:
        X = MultiLabelBinarizer(classes=features, sparse_output=True).fit_transform(__X(pd.read_json(fin)["instructions"]))  # TODO

    return X
