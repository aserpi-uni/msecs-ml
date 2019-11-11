from keras import models, layers


def inception(image_size):
    from keras.applications import InceptionV3
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


def resnet50(image_size):
    from keras.applications import ResNet50V2
    resnet_conv = ResNet50V2(weights="imagenet", include_top=False, input_shape=image_size.rgb_dimensions())

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


def vgg16(image_size):
    from keras.applications import VGG16
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
