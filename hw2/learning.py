import argparse
import csv
from contextlib import suppress
from pathlib import Path
import re
from typing import Set, Tuple

from keras.callbacks import Callback, ModelCheckpoint
from keras.engine.saving import load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import natsort


def train(net: str, epochs: int,
          train_dir: Path, test_dir: Path, out_dir: Path,
          batch_size: int = None, image_size: Tuple[int, int] = None, save_models: Set[str] = None) -> Path:
    if not image_size:
        if net == "earenet" or net == "inception":
            image_size = (299, 299)
        elif net == "resnet50" or net == "vgg16":
            image_size = (224, 224)
    if save_models is None:
        save_models = []

    try:
        model_file = natsort.natsorted([f for f in (out_dir / "models").glob(f"{net}-*.h5")
                                        if re.match(fr"{net}-\d+\.h5", f.name)])[-1]
        initial_epoch = int(re.match(fr"{net}-(\d+)\.h5", model_file.name).group(1))
        model = load_model(model_file)
        print(f"Loaded model {model_file.name}")

    except IndexError:
        initial_epoch = 0
        num_classes = len([c for c in train_dir.iterdir() if c.is_dir()])
        (out_dir / "models").mkdir(exist_ok=True, parents=True)
        if net == "earenet":
            from hw2.models import earenet
            model = earenet(image_size, num_classes)
        elif net == "inception":
            from hw2.models import inception
            model = inception(image_size, num_classes)
        elif net == "resnet50":
            from hw2.models import resnet50
            model = resnet50(image_size, num_classes)
        elif net == "vgg16":
            from hw2.models import vgg16
            model = vgg16(image_size, num_classes)
        else:
            raise argparse.ArgumentError(f"Unknown network '{net}'")

    # Show model summary
    model.summary()

    # Read images and augment data
    train_datagen = ImageDataGenerator(fill_mode="reflect",
                                       height_shift_range=0.2,
                                       horizontal_flip=True,
                                       rescale=1. / 255,
                                       rotation_range=20,
                                       shear_range=0.2,
                                       width_shift_range=0.2)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    # Use batch_size only if it is a meaningful value
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=image_size,
                                                        class_mode='categorical',
                                                        **{k: v for k, v in {"batch_size": batch_size}.items() if v})
    validation_generator = validation_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        class_mode='categorical',
        shuffle=False,
        **{k: v for k, v in {"batch_size": batch_size}.items() if v})

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=(1e-3 if net == "earenet" else 1e-4)),
                  metrics=['acc'])

    # Define callbacks
    history_checkpoint = HistoryCheckPoint(net, out_dir)
    callbacks_list = [history_checkpoint]
    if "best" in save_models:
        callbacks_list.append(ModelCheckpoint((out_dir / "models" / f"{net}-best.h5").as_posix(), period=1,
                                              save_best_only=True, verbose=1))
    if "all" in save_models:
        callbacks_list.append(ModelCheckpoint((out_dir / "models" / (net + "-{epoch:02d}.h5")).as_posix(), period=1,
                                              verbose=1))
    elif "last" in save_models:
        callbacks_list.append(LastModelCheckpoint((out_dir / "models" / (net + "-{epoch:02d}.h5")).as_posix(), period=1,
                                                  verbose=1))

    # Train model
    model.fit_generator(train_generator,
                        callbacks=callbacks_list,
                        epochs=epochs,
                        initial_epoch=initial_epoch,
                        steps_per_epoch=(train_generator.samples / train_generator.batch_size),
                        validation_data=validation_generator,
                        validation_steps=(validation_generator.samples / validation_generator.batch_size),
                        verbose=1)

    return history_checkpoint.history


class HistoryCheckPoint(Callback):
    metrics = ["acc", "val_acc", "loss", "val_loss"]

    def __init__(self, net, out_dir):
        super().__init__()
        self.history = out_dir / f"{net}.csv"

        if not self.history.is_file():
            with open(self.history, "w", newline='') as fout:
                writer = csv.DictWriter(fout, fieldnames=HistoryCheckPoint.metrics)
                writer.writeheader()

    def on_epoch_end(self, epoch, logs=None):
        with open(self.history, "a", newline='') as fout:
            csv.DictWriter(fout, extrasaction="ignore", fieldnames=HistoryCheckPoint.metrics, restval=0) \
                .writerow(logs)


class LastModelCheckpoint(ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        if self.epochs_since_last_save + 1 >= self.period:
            with suppress(FileNotFoundError):  # New in Python 3.8: Path.unlink(missing_ok=True)
                Path(self.filepath.format(epoch=epoch-self.epochs_since_last_save)).unlink()

        super().on_epoch_end(epoch, logs=logs)
