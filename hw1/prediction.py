import numpy as np
import pickle


def consolidate(compiler_algorithm, optimization_algorithm, dataset_directory, output_directory):
    with open(f"{dataset_directory}/compiler_{compiler_algorithm}.csv") as fin_compiler, \
            open(f"{dataset_directory}/opt_{optimization_algorithm}.csv") as fin_optimization, \
            open(f"{output_directory}/results.csv", "w") as fout:

        while True:
            compiler = fin_compiler.readline()
            opt = fin_optimization.readline()

            if (compiler and not opt) or (not compiler and opt):
                raise ValueError("The files have not the same number of lines")
            if not compiler and not opt:
                break

            print(f"{compiler.strip()},{opt.strip()}", file=fout)


def predict(x, classifier, algorithms, dataset_dir, results_dir):
    for alg in algorithms:
        try:
            with open(f"{dataset_dir}/classifiers/{classifier}_{alg}.zip", "rb") as fin:
                pred = pickle.load(fin).predict(x)
        except (FileNotFoundError, TypeError):
            print(f"ERROR: No classifier '{alg}' for task '{classifier}'")
            continue

        np.savetxt(f"{results_dir}/{classifier}_{alg}.csv", pred, fmt="%s")
