import pandas as pd
import pickle


# 0: missing value
# 1: unknown instruction
# >1: known value
def __x(data, features=None, truncate=None, return_features=False):
    if features:
        for fun_idx, fun in enumerate(data):
            if truncate:  # and len(fun) > truncate * 2:
                fun = fun[:truncate] + fun[-truncate:]
                data[fun_idx] = fun
            for instr_idx, instr in enumerate(fun):
                instr = instr.split(" ", 1)[0]
                try:
                    fun[instr_idx] = features[instr]
                except KeyError:
                    fun[instr_idx] = 1

    else:
        instructions = {}
        max_id = 1
        for fun_idx, fun in enumerate(data):
            if truncate:  # and len(fun) > truncate * 2:
                fun = fun[:truncate] + fun[-truncate:]
                data[fun_idx] = fun
            for instr_idx, instr in enumerate(fun):
                instr = instr.split(" ")[0]
                try:
                    fun[instr_idx] = instructions[instr]
                except KeyError:
                    max_id += 1
                    instructions[instr] = max_id
                    fun[instr_idx] = instructions[instr]

    df = pd.DataFrame.from_records(data).fillna(0)
    if return_features:
        return df, (features if features else instructions)
    else:
        return df


def labels(classifier):
    if classifier == "compiler":
        return ["clang", "icc", "gcc"]
    elif classifier == "opt":
        return ["H", "L"]


def train_datasets(dataset_dir, classifier, cache=False, truncate=None):
    try:
        with open(f"{dataset_dir}/x_{truncate}.zip", "rb") as fin_X, \
                open(f"{dataset_dir}/y_{classifier}.zip", "rb") as fin_y:
            x = pickle.load(fin_X)
            y = pickle.load(fin_y)

    except (FileNotFoundError, TypeError):
        data = pd.read_json(f"{dataset_dir}/train_dataset.jsonl", lines=True)
        x, features = __x(data["instructions"], return_features=True, truncate=truncate)
        y = data[classifier]

        if cache:
            with open(f"{dataset_dir}/x_{truncate}.zip", "wb") as fout_X, \
                    open(f"{dataset_dir}/y_{classifier}.zip", "wb") as fout_y, \
                    open(f"{dataset_dir}/features_{truncate}.zip", "wb") as fout_features:
                pickle.dump(x, fout_X, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(y, fout_y, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(features, fout_features, protocol=pickle.HIGHEST_PROTOCOL)

    return x, y


def test_dataset(dataset_dir, truncate=None):
    try:
        with open(f"{dataset_dir}/features_{truncate}.zip", "rb") as fin_features:
            features = pickle.load(fin_features)
    except (FileNotFoundError, TypeError):
        x_train, features = __x(pd.read_json(f"{dataset_dir}/train_dataset.jsonl", lines=True)["instructions"],
                                truncate=truncate, return_features=True)

    return __x(pd.read_json(f"{dataset_dir}/test_dataset_blind.jsonl", lines=True)["instructions"],
               features=features, truncate=truncate)
