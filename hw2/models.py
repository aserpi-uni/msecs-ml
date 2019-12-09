from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Flatten, Dropout


def inception(image_size, num_classes):
    from keras.applications import InceptionV3
    inception_conv = InceptionV3(weights="imagenet", include_top=False, input_shape=image_size.rgb_dimensions())

    return __fine_tuning_model(inception_conv, num_classes)


def resnet50(image_size, num_classes):
    from keras.applications import ResNet50V2
    resnet_conv = ResNet50V2(weights="imagenet", include_top=False, input_shape=image_size.rgb_dimensions())

    return __fine_tuning_model(resnet_conv, num_classes)


def vgg16(image_size, num_classes):
    from keras.applications import VGG16
    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=image_size.rgb_dimensions())

    return __fine_tuning_model(vgg_conv, num_classes)


def __fine_tuning_model(original_model, num_classes):
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
