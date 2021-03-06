from typing import Tuple

from keras.layers import BatchNormalization, Dense, Flatten, Dropout
from keras.models import Sequential


def create_model(net: str, image_size: Tuple[int, int], num_classes: int) -> Sequential:
    if net == "earenet":
        model = earenet(image_size, num_classes)
    elif net == "inception":
        model = inception(image_size, num_classes)
    elif net == "resnet50":
        model = resnet50(image_size, num_classes)
    elif net == "vgg16":
        model = vgg16(image_size, num_classes)
    else:
        raise ValueError(f"Unknown network '{net}'")
    return model


def default_image_size(net: str) -> Tuple[int, int]:
    if net in ("earenet", "inception"):
        image_size = (299, 299)
    elif net in ("resnet50", "vgg16"):
        image_size = (224, 224)
    else:
        raise ValueError(f"No default image size for network '{net}'")
    return image_size


def earenet(image_size: Tuple[int, int], num_classes: int) -> Sequential:
    from keras.layers import Conv2D, MaxPooling2D

    model = Sequential(name="EareNet")

    model.add(
        Conv2D(64,
               kernel_size=(3, 3),
               strides=(1, 1),
               activation='relu',
               padding="same",
               name="bolck1_conv",
               input_shape=(*image_size, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block1_pool'))
    model.add(BatchNormalization(name='block1_norm'))

    model.add(
        Conv2D(128,
               kernel_size=(3, 3),
               strides=(1, 1),
               activation='relu',
               padding='same',
               name="bolck2_conv"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block2_pool'))
    model.add(BatchNormalization(name='block2_norm'))

    model.add(
        Conv2D(256,
               kernel_size=(2, 2),
               strides=(1, 1),
               activation='relu',
               padding='valid',
               name="bolck3_conv"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='block3_pool'))
    model.add(BatchNormalization(name='block3_norm'))

    model.add(Flatten(name="top_flatten"))
    model.add(Dense(512, activation='relu', name="top_dense1"))
    model.add(BatchNormalization(name="top_norm"))
    model.add(Dropout(0.5, name="top_dropout"))
    model.add(Dense(num_classes, activation='softmax', name="top_dense2"))

    return model


def inception(image_size: Tuple[int, int], num_classes: int) -> Sequential:
    from keras.applications import InceptionV3
    inception_conv = InceptionV3(weights="imagenet",
                                 include_top=False,
                                 input_shape=(*image_size, 3))

    return _fine_tuning_model(inception_conv, num_classes)


def resnet50(image_size: Tuple[int, int], num_classes: int) -> Sequential:
    from keras.applications import ResNet50V2
    resnet_conv = ResNet50V2(weights="imagenet", include_top=False, input_shape=(*image_size, 3))

    return _fine_tuning_model(resnet_conv, num_classes)


def vgg16(image_size: Tuple[int, int], num_classes: int) -> Sequential:
    from keras.applications import VGG16
    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(*image_size, 3))

    return _fine_tuning_model(vgg_conv, num_classes)


def _fine_tuning_model(original_model, num_classes):
    model = Sequential(name=f"fine_tuning_{original_model.name}")

    # Freeze all layers except last 4
    for layer in original_model.layers[:-4]:
        layer.trainable = False

    # Add original network as the first layer
    model.add(original_model)

    # Add new layers
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    return model
