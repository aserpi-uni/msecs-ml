from keras import models, layers
from keras.applications import VGG16


def vgg16(image_size):
    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=image_size.rgb_dimensions())

    # Freeze all layers except last 4
    for layer in vgg_conv.layers[:-4]:
        layer.trainable = False

    # Create model
    model = models.Sequential()

    # Add VGG convolutional base model
    model.add(vgg_conv)

    # Add new layers
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(18, activation='softmax'))

    return model
