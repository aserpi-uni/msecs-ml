# TODO: rewrite
def initialize_dir(work_dir):
    import pandas as pd
    import re
    import shutil

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
    import argparse
    from pathlib import Path

    from hw2.image_size import ImageSize
    from hw2.learning import tune

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
