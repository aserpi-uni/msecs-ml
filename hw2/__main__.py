import argparse
from pathlib import Path

from hw2.data import ImageSize
from hw2.learning import tune


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("net", choices=["inception", "resnet50", "vgg16"], help="neural network")
    parser.add_argument("epochs", help="epochs", type=int)
    parser.add_argument("train_directory", help="train dataset directory", type=Path)
    parser.add_argument("test_directory", help="test dataset directory", type=Path)
    parser.add_argument("out_directory", help="output directory", type=Path)

    # Optional arguments
    parser.add_argument("-b", "--batch-size", help="batch size", type=int)
    parser.add_argument("-p", "--persistence",
                        choices=["all", "best", "last"], default=[], help="save models", nargs="*")
    parser.add_argument("-s", "--stats", action="store_true", help="display statistics")

    args = vars(parser.parse_args())

    history_file = tune(ImageSize(240, 800),
                        args["net"],
                        args["epochs"],
                        args["train_directory"],
                        args["test_directory"],
                        args["out_directory"],
                        args["batch_size"],
                        set(args["persistence"]))

    if args["stats"]:
        from hw2.evaluation import plot_metrics
        plot_metrics(args["net"], history_file)
