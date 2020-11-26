from contextlib import suppress
from pathlib import Path
import re
from typing import Optional, Set, Tuple, Union

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

    history_file = out_dir / f"{net}.csv"

    try:
        model_file = natsort.natsorted([f for f in (out_dir / "models").glob(f"{net}-*.h5")
                                        if re.match(fr"{net}-\d+\.h5", f.name)
                                        ])[-1]  # yapf:disable
        initial_epoch = int(re.match(fr"{net}-(\d+)\.h5", model_file.name).group(1))
        model = load_model(model_file)
        print(f"Loaded model {model_file.name}")
        previous_best = pd.read_csv(history_file)["val_loss"].max()

    except IndexError:
        initial_epoch = 0
        num_classes = len([c for c in train_dir.iterdir() if c.is_dir()])
        (out_dir / "models").mkdir(exist_ok=True, parents=True)
        model = create_model(net, image_size, num_classes)
        previous_best = None

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
    callbacks_list = [(CSVLogger(str(history_file), append=True))]
    if "best" in save_models:
        callbacks_list.append(
            ResumableModelCheckpoint(str(out_dir / "models" / f"{net}-best.h5"),
                                     period=1,
                                     previous_best=previous_best,
                                     save_best_only=True,
                                     verbose=1))
    if "all" in save_models:
        callbacks_list.append(
            ResumableModelCheckpoint(str(out_dir / "models" / (net + "-{epoch:02d}.h5")),
                                     period=1,
                                     verbose=1))
    elif "last" in save_models:
        callbacks_list.append(
            ResumableModelCheckpoint(str(out_dir / "models" / (net + "-{epoch:02d}.h5")),
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


class ResumableModelCheckpoint(ModelCheckpoint):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled with the values of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: interval (number of epochs) between checkpoints.
        previous_best: value of the monitored metric in the last-saved model.
        previous_filepath: path of the last-saved model.
    """

    def __init__(self,
                 filepath,
                 previous_best: Optional[float] = None,
                 previous_filepath: Optional[Union[Path, str]] = None,
                 **kwargs):
        super().__init__(filepath, **kwargs)

        if previous_best is not None:
            self.best = previous_best
        self.previous_filepath = previous_filepath

    def on_epoch_end(self, epoch, logs=None):
        if self.epochs_since_last_save + 1 >= self.period:
            if not self.save_best_only or self.monitor_op(logs[self.monitor], self.best):
                with suppress(FileNotFoundError):
                    # New in Python 3.8: Path.unlink(missing_ok=True)
                    Path(self.previous_filepath).unlink()
                self.previous_filepath = self.filepath.format(epoch=epoch, **logs)

        super().on_epoch_end(epoch, logs=logs)
