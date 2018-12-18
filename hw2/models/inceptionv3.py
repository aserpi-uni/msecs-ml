from keras import models, layers
from keras.applications import InceptionV3


def inceptionv3(image_size):
    inception_conv = InceptionV3(weights="imagenet", include_top=False, input_shape=image_size.rgb_dimensions())

    # Freeze all layers
    for layer in inception_conv.layers:
        layer.trainable = False

    # Create model
    model = models.Sequential()

    # Add Inception convolutional base model
    model.add(inception_conv)

    # Add new layers
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(18, activation='softmax'))

    return model
