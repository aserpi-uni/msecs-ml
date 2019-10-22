import pickle


def consolidate(compiler_algorithm, optimization_algorithm, dataset_directory, output_directory):
    with open(f"{dataset_directory}/compiler/{compiler_algorithm}.csv") as fin_compiler, \
            open(f"{dataset_directory}/opt/{optimization_algorithm}.csv") as fin_optimization, \
            open(f"{output_directory}/results.csv", "w") as fout:

        while True:
            compiler = fin_compiler.readline()
            opt = fin_optimization.readline()

            if (compiler and not opt) or (not compiler and opt):
                raise ValueError("The files have not the same number of lines")
            if not compiler and not opt:
                break

            print(f"{compiler},{opt}", file=fout)


def predict(X, classifier, algorithms, dataset_dir, results_dir):
    for alg in algorithms:
        try:
            pred = pickle.load(f"{dataset_dir}/{classifier}/{alg}").predict(X)
        except (FileNotFoundError, TypeError):
            print(f"ERROR: No classifier '{alg}' for task '{classifier}'")
            continue

        pred.to_pickle(f"{results_dir}/{classifier}/{alg}_results.zip")  # TODO: remove

        pred.to_csv(f"{results_dir}/{classifier}/{alg}.csv", "a")
