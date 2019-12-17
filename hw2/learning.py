from contextlib import suppress
from pathlib import Path
import re
from typing import Optional, Set, Tuple

from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.engine.saving import load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import natsort
import pandas as pd

from hw2.models import create_model, default_image_size


def train(net: str,
          epochs: int,
          train_dir: Path,
          val_dir: Path,
          out_dir: Path,
          batch_size: Optional[int] = None,
          image_size: Optional[Tuple[int, int]] = None,
          save_models: Optional[Set[str]] = None) -> Path:
    if not image_size:
        image_size = default_image_size(net)
    if save_models is None:
        save_models = []

    try:
        model_file = natsort.natsorted([f for f in (out_dir / "models").glob(f"{net}-*.h5")
                                        if re.match(fr"{net}-\d+\.h5", f.name)
                                        ])[-1]  # yapf:disable
        initial_epoch = int(re.match(fr"{net}-(\d+)\.h5", model_file.name).group(1))
        model = load_model(model_file)
        print(f"Loaded model {model_file.name}")

    except IndexError:
        initial_epoch = 0
        num_classes = len([c for c in train_dir.iterdir() if c.is_dir()])
        (out_dir / "models").mkdir(exist_ok=True, parents=True)
        model = create_model(net, image_size, num_classes)

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
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        class_mode='categorical',
        **{k: v for k, v in {"batch_size": batch_size}.items() if v})
    validation_generator = validation_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        class_mode='categorical',
        shuffle=False,
        **{k: v for k, v in {"batch_size": batch_size}.items() if v})

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=(1e-3 if net == "earenet" else 1e-4)),
                  metrics=['acc'])

    # Define callbacks
    history_file = out_dir / f"{net}.csv"
    callbacks_list = [(CSVLogger(str(history_file), append=True))]
    if "best" in save_models:
        callbacks_list.append(
            ModelCheckpoint((out_dir / "models" / f"{net}-best.h5").as_posix(),
                            period=1,
                            save_best_only=True,
                            verbose=1))  # TODO: restore precedent history
    if "all" in save_models:
        callbacks_list.append(
            ModelCheckpoint((out_dir / "models" / (net + "-{epoch:02d}.h5")).as_posix(),
                            period=1,
                            verbose=1))
    elif "last" in save_models:
        callbacks_list.append(
            LastModelCheckpoint((out_dir / "models" / (net + "-{epoch:02d}.h5")).as_posix(),
                                period=1,
                                verbose=1))

    # Train model
    model.fit_generator(train_generator,
                        callbacks=callbacks_list,
                        epochs=epochs,
                        initial_epoch=initial_epoch,
                        steps_per_epoch=(train_generator.samples / train_generator.batch_size),
                        validation_data=validation_generator,
                        validation_steps=(validation_generator.samples /
                                          validation_generator.batch_size),
                        verbose=1)

    class_names = pd.Series(sorted([d.name for d in train_dir.iterdir() if d.is_dir()]),
                            name="class")
    class_names.index.name = "id"
    class_names.to_csv(out_dir / "classes.csv", header=True, index=True)

    return history_file


class LastModelCheckpoint(ModelCheckpoint):

    def on_epoch_end(self, epoch, logs=None):
        if self.epochs_since_last_save + 1 >= self.period:
            with suppress(FileNotFoundError):  # New in Python 3.8: Path.unlink(missing_ok=True)
                Path(self.filepath.format(epoch=epoch - self.epochs_since_last_save)).unlink()

        super().on_epoch_end(epoch, logs=logs)
