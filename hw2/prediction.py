import math
import pathlib
from typing import Tuple

from keras.engine.saving import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


def predict(net: str, epoch: int,
            working_dir: pathlib.Path, test_dir: pathlib.Path,
            batch_size: int = None, image_size: Tuple[int, int] = None) -> None:
    model = load_model(working_dir / "models" / f"{net}-{epoch}.h5")
    out_dir = working_dir / "predictions"

    if not image_size:
        if net in ("earenet", "inception"):
            image_size = (299, 299)
        elif net in ("resnet50", "vgg16"):
            image_size = (224, 224)

    test_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        test_dir,
        class_mode=None,
        color_mode="rgb",
        shuffle=False,
        target_size=image_size,
        **{k: v for k, v in {"batch_size": batch_size}.items() if v})
    preds = model.predict_generator(test_generator, steps=math.ceil(test_generator.n / test_generator.batch_size),
                                    verbose=1)

    preds = np.argmax(preds, axis=1)
    out_dir.mkdir(exist_ok=True, parents=True)
    np.savetxt(out_dir / f"{net}-{epoch}.csv", preds)
