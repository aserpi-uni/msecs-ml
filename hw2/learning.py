from argparse import ArgumentError
import csv
from natsort import natsorted
import re
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import Callback, CSVLogger, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def tune(image_size, work_dir, net, epochs, batch_size=None, evaluate=False, persistence=None):
    if persistence is None:
        persistence = []

    try:
        model_file = natsorted([f for f in (work_dir / "keras").glob(f"{net}-*.h5")
                                if re.match(fr"{net}-\d+\.h5", f.name)])[-1]
        initial_epoch = int(re.match(fr"{net}-(\d+)\.h5", model_file.name).group(1))
        model = load_model(model_file)
        print(f"Loaded model {model_file.name}")

    except IndexError:
        initial_epoch = 0
        if net == "inception":
            from hw2.models import inception
            model = inception(image_size)
        elif net == "resnet50":
            from hw2.models import resnet50
            model = resnet50(image_size)
        elif net == "vgg16":
            from hw2.models import vgg16
            model = vgg16(image_size)
        else:
            raise ArgumentError(f"Unknown network '{net}'")

    # Show model summary
    model.summary()

    # Read images and augment data
    train_datagen = ImageDataGenerator(fill_mode="nearest",
                                       height_shift_range=0.2,
                                       horizontal_flip=True,
                                       rescale=1. / 255,
                                       rotation_range=20,
                                       shear_range=0.2,
                                       width_shift_range=0.2)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(work_dir / "train_images",
                                                        target_size=image_size.dimensions(),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')
    validation_generator = validation_datagen.flow_from_directory(work_dir / "test_images",
                                                                  target_size=image_size.dimensions(),
                                                                  batch_size=batch_size,
                                                                  class_mode='categorical',
                                                                  shuffle=False)

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=1e-4),
                  metrics=['acc'])

    # Define callbacks
    history_checkpoint = HistoryCheckPoint(net, work_dir)
    callbacks_list = [history_checkpoint]
    if "best" in persistence:
        callbacks_list.append(ModelCheckpoint((work_dir / "keras" / f"{net}-best.h5").as_posix(), save_freq="epoch",
                                              save_best_only=True, verbose=1))
    if "all" in persistence:
        callbacks_list.append(ModelCheckpoint((work_dir / "keras" / (net + "-{epoch:02d}.h5")).as_posix(),
                                              save_freq="epoch", verbose=1))
    elif "last" in persistence:
        callbacks_list.append(ModelCheckpoint((work_dir / "keras" / f"{net}.h5").as_posix(), save_freq="epoch",
                                              verbose=1))

    # Train model
    model.fit_generator(train_generator,
                        callbacks=callbacks_list,
                        epochs=epochs,
                        initial_epoch=initial_epoch,
                        steps_per_epoch=train_generator.samples / train_generator.batch_size,
                        validation_data=validation_generator,
                        validation_steps=validation_generator.samples / validation_generator.batch_size,
                        verbose=1)

    if evaluate:
        from hw2.evaluation import plot_metrics
        plot_metrics(history_checkpoint.history, net)


class HistoryCheckPoint(Callback):
    metrics = ["acc", "val_acc", "loss", "val_loss"]

    def __init__(self, net, work_dir):
        super().__init__()
        self.history = work_dir / "keras" / f"{net}.history"

        if not self.history.is_file():
            with open(self.history, "w", newline='') as fout:
                writer = csv.DictWriter(fout, fieldnames=HistoryCheckPoint.metrics)
                writer.writeheader()

    def on_epoch_end(self, epoch, logs=None):
        with open(self.history, "a", newline='') as fout:
            csv.DictWriter(fout, extrasaction="ignore", fieldnames=HistoryCheckPoint.metrics, restval=0) \
                .writerow(logs)
