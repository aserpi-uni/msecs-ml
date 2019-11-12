import argparse
from pathlib import Path

from hw2.data import ImageSize, initialise_dir
from hw2.learning import tune


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("net", choices=["inception", "resnet50", "vgg16"], help="neural network")
    parser.add_argument("epochs", help="epochs", type=int)
    parser.add_argument("working_directory", help="working directory", type=Path)

    # Optional arguments

    parser.add_argument("-b", "--batch-size", help="batch size", type=int)
    parser.add_argument("-i", "--initialise", action="store_true", help="initialise working directory")
    parser.add_argument("-p", "--persistence", choices=["all", "best", "last"], help="save models", nargs="?")
    parser.add_argument("-s", "--stats", action="store_true", help="display statistics")

    args = vars(parser.parse_args())

    if args["initialise"]:
        initialise_dir(args["working_directory"])

    tune(ImageSize(240, 800),
         args["working_directory"],
         args["net"],
         args["epochs"],
         args["batch_size"],
         args["stats"],
         args["persistence"])
