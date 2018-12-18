from keras import models, layers
from keras.applications import ResNet50


def resnet50(image_size):
    resnet_conv = ResNet50(weights="imagenet", include_top=False, input_shape=image_size.rgb_dimensions())

    # Freeze all layers
    for layer in resnet_conv.layers:
        layer.trainable = False

    # Create model
    model = models.Sequential()

    # Add ResNet convolutional base model
    model.add(resnet_conv)

    # Add new layers
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(18, activation='softmax'))

    return model
