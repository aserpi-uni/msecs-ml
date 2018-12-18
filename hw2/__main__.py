import argparse
import re

from hw2.image_size import ImageSize
from hw2.learning import tune


def initialize_dir(work_dir):
    import pandas as pd
    import shutil

    from pathlib import Path

    Path(f"{work_dir}/keras").mkdir(exist_ok=True)

    images = pd.read_csv(f"{work_dir}/test_images/ground_truth.txt", sep=";", header=None, names=["image", "class"])

    images["class"] = images["class"].apply(lambda cls: re.sub(r'[^\w]', '', cls))

    for boat_type in images["class"].unique():
        Path(f"{work_dir}/test_images/{boat_type}").mkdir()

    for idx, row in images.iterrows():
        Path(f"{work_dir}/test_images/{row['image']}").rename(f"{work_dir}/test_images/{row['class']}/{row['image']}")

    test_classes = [p.name for p in Path(f"{work_dir}/test_images").glob("*") if p.is_dir()]
    train_classes = [p.name for p in Path(f"{work_dir}/train_images").glob("*") if p.is_dir()]

    for c in test_classes:
        if c not in train_classes:
            shutil.rmtree(f"{work_dir}/test_images/{c}")

    for c in train_classes:
        if c not in test_classes:
            shutil.rmtree(f"{work_dir}/train_images/{c}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("-b", "--batch-size", help="batch size")
    parser.add_argument("-d", "--work-dir", help="working directory")
    parser.add_argument("-e", "--epochs", help="epochs")
    parser.add_argument("-n", "--net", help="neural network")

    # Optional arguments
    parser.add_argument("-i", "--initialize", action="store_true", help="initialize working directory")
    parser.add_argument("-p", "--persistence", help="save models", nargs="*")
    parser.add_argument("-s", "--stats", action="store_true", help="display statistics")

    args = vars(parser.parse_args())

    if args["initialize"]:
        initialize_dir(args["work_dir"])

    tune(ImageSize(240, 800),
         args["work_dir"],
         args["net"],
         int(args["epochs"]),
         int(args["batch_size"]),
         args["stats"],
         args["persistence"][0] if args["persistence"] else None)
