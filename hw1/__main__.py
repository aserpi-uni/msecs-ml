import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--classifier")
    parser.add_argument("--drebin")
    parser.add_argument("--output")

    # Optional arguments
    parser.add_argument("--algorithms", nargs="*")
    parser.add_argument("--features", nargs="*")

    args = vars(parser.parse_args())
    if not args["algorithms"]:
        args["algorithms"] = ["bernoulli", "random_forest", "svc", "linear_svc"]

    if args["classifier"] == "malware":
        pass  # TODO
    elif args["classifier"] == "family":
        family_classification(args["drebin"], args["output"], args["algorithms"], args["features"])
    else:
        raise KeyError
