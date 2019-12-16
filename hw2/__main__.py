from pathlib import Path
from typing import Optional, Sequence, Tuple

from hw2.learning import train


def main(net: str, epochs: int, train_directory: Path, test_directory: Path, out_directory: Path,
         batch_size: Optional[int] = None, image_size: Optional[Tuple[int, int]] = None,
         save_models: Optional[Sequence[str]] = None, stats: Optional[bool] = None,
         predict_directory: Optional[Path] = None):

    if save_models is None:
        save_models = []

    history_file = train(net, epochs, train_directory, test_directory, out_directory, batch_size, image_size,
                         set(save_models))

    if stats:
        from hw2.evaluation import plot_metrics
        plot_metrics(net, history_file)

    if predict_directory:
        from hw2.prediction import predict
        predict(net, epochs, out_directory, predict_directory, batch_size, image_size)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("net", choices=["earenet", "inception", "resnet50", "vgg16"], help="neural network")
    parser.add_argument("epochs", help="epochs", type=int)
    parser.add_argument("train_directory", help="train dataset directory", type=Path)
    parser.add_argument("test_directory", help="test dataset directory", type=Path)
    parser.add_argument("out_directory", help="output directory", type=Path)

    # Optional arguments
    parser.add_argument("-b", "--batch-size", help="batch size", type=int)
    parser.add_argument("-i", "--image-size", help="image size")
    parser.add_argument("-m", "--save-models",
                        choices=["all", "best", "last"], default=[], help="save models", nargs="*")
    parser.add_argument("-p", "--predict-directory", help="predict classes for images in directory", type=Path)
    parser.add_argument("-s", "--stats", action="store_true", help="display statistics")

    args = vars(parser.parse_args())

    if args["image_size"]:
        def invalid_image_size():
            parser.error("--image-size must be in the form '<int>,<int>'")
        try:
            args["image_size"] = tuple(map(int, args["image_size"].split(",", 1)))
        except ValueError:
            invalid_image_size()
        if len(args["image_size"]) != 2:
            invalid_image_size()

    main(args["net"],
         args["epochs"],
         args["train_directory"],
         args["test_directory"],
         args["out_directory"],
         args["batch_size"],
         args["image_size"],
         args["save_models"],
         args["stats"],
         args["predict_directory"])
