from keras import models, layers


def inception(image_size, num_classes):
    from keras.applications import InceptionV3
    inception_conv = InceptionV3(weights="imagenet", include_top=False, input_shape=image_size.rgb_dimensions())

    # Freeze all layers
    for layer in inception_conv.layers:
        layer.trainable = False

    return add_bottom(inception_conv, num_classes)


def resnet50(image_size, num_classes):
    from keras.applications import ResNet50V2
    resnet_conv = ResNet50V2(weights="imagenet", include_top=False, input_shape=image_size.rgb_dimensions())

    # Freeze all layers
    for layer in resnet_conv.layers:
        layer.trainable = False

    return add_bottom(resnet_conv, num_classes)


def vgg16(image_size, num_classes):
    from keras.applications import VGG16
    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=image_size.rgb_dimensions())

    # Freeze all layers except last 4
    for layer in vgg_conv.layers[:-4]:
        layer.trainable = False

    return add_bottom(vgg_conv, num_classes)


def add_bottom(original_model, num_classes):
    model = models.Sequential()
    model.add(original_model)

    # Add new layers
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'), )

    return model