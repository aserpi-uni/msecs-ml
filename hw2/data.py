class ImageSize:
    def __init__(self, width, height):
        self.height = height
        self.channels_order = "channels_last"
        self.width = width

    def dimensions(self):
        return self.height, self.width

    def rgb_dimensions(self):
        return self.height, self.width, 3


def initialise_dir(work_dir):
    import pandas as pd
    import re
    import shutil

    test_dir = work_dir / "test_images"
    test_groud_truths = test_dir / "ground_truth.txt"
    train_dir = work_dir / "train_images"

    test_images = pd.read_csv(test_groud_truths, sep=";", header=None, names=["image", "class"])
    test_images["class"] = test_images["class"].apply(lambda c: re.sub(r'[^\w]', '', c))

    # Not all classes in the train dataset are also in the test dataset and vice versa
    test_classes = set(test_images["class"].unique())
    train_classes = set([p.name for p in train_dir.iterdir() if p.is_dir()])
    common_classes = test_classes.intersection(train_classes)

    for boat_type in common_classes:
        (test_dir / boat_type).mkdir()

    def __move_or_delete_image(row):
        image = test_dir / row["image"]
        if row["class"] in common_classes:
            image.rename(test_dir / row["class"] / row["image"])
        else:
            image.unlink()
    test_images.apply(lambda r: __move_or_delete_image(r), axis=1)

    for c in train_classes:
        if c not in common_classes:
            shutil.rmtree(train_dir / c)

    test_groud_truths.unlink()
    (work_dir / "keras").mkdir(exist_ok=True)
