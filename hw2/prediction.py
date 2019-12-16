from keras.engine.saving import load_model
from keras.preprocessing.image import ImageDataGenerator
from math import ceil
import numpy as np


def predict(net, epoch, working_dir, test_dir, batch_size=None, image_size=None):
    model = load_model(working_dir / "models" / f"{net}-{epoch}.h5")
    out_dir = working_dir / "predictions"

    if not image_size:
        from hw2.data import ImageSize
        if net == "earenet" or net == "inception":
            image_size = ImageSize(299, 299)
        elif net == "resnet50" or net == "vgg16":
            image_size = ImageSize(224, 224)

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(test_dir,
                                                      class_mode=None,
                                                      color_mode="rgb",
                                                      shuffle=False,
                                                      target_size=image_size.dimensions(),
                                                      **{k: v for k, v in {"batch_size": batch_size}.items() if v})
    preds = model.predict_generator(test_generator, steps=ceil(test_generator.n / test_generator.batch_size), verbose=1)

    preds = np.argmax(preds, axis=1)
    out_dir.mkdir(exist_ok=True, parents=True)
    np.savetxt(out_dir / f"{net}-{epoch}.csv", preds)
