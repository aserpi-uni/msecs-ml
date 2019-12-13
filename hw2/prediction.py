import math
import pathlib
from typing import Tuple

from keras.engine.saving import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd

from hw2.models import default_image_size


def predict(net: str, epoch: int,
            working_dir: pathlib.Path, test_dir: pathlib.Path,
            batch_size: int = None, image_size: Tuple[int, int] = None) -> None:
    model = load_model(working_dir / "models" / f"{net}-{epoch}.h5")
    out_dir = working_dir / "predictions"

    if not image_size:
        image_size = default_image_size(net)

    test_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        test_dir,
        class_mode=None,
        color_mode="rgb",
        shuffle=False,
        target_size=image_size,
        **{k: v for k, v in {"batch_size": batch_size}.items() if v})
    preds = np.argmax(model.predict_generator(test_generator,
                                              steps=math.ceil(test_generator.n / test_generator.batch_size),
                                              verbose=1),
                      axis=1)

    classes = pd.read_csv(working_dir / "classes.csv", header=0, index_col="id", squeeze=True)
    pred_classes = pd.Series(preds.flatten()).apply(lambda p: classes.loc[p])

    out_dir.mkdir(exist_ok=True, parents=True)
    pred_classes.to_csv(out_dir / f"{net}-{epoch}.csv", header=None, index=False)
