import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import (
    ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2,
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4,
    EfficientNetB5, EfficientNetB6, EfficientNetB7, MobileNet, MobileNetV2, VGG16, VGG19,
    InceptionV3, Xception, InceptionResNetV2, DenseNet121, DenseNet169, DenseNet201)
from models.simple_cnn import simple_cnn


def get_model(cnn_model, input_shape, activation):
    """
    EfficientNet
    Args:
      cnn_model {str} -- name of resnets with different layers, i.e. 'ResNet101'.
      weights {str} -- model weights from imagenet
      input_shape {tuple} -- folder path to save model
      freeze_layer {int} -- number of layers to freeze
      activation {str} -- activation function in last layer
    Returns:
        cnn model
    """
    if cnn_model == 'simple_cnn':
        model = simple_cnn(input_shape=input_shape, activation=activation)
        model.summary()
    else:
        include_top = False
        weights = None
        # Inception
        if cnn_model == 'InceptionV3':
            base_model = InceptionV3(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'Xception':
            base_model = Xception(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'InceptionResNetV2':
            base_model = InceptionResNetV2(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)

        # ResNet
        elif cnn_model == 'ResNet50V2':
            base_model = ResNet50V2(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'ResNet101V2':
            base_model = ResNet101V2(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'ResNet152V2':
            base_model = ResNet152V2(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        if cnn_model == 'ResNet50':
            base_model = ResNet50(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'ResNet101':
            base_model = ResNet101(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'ResNet152':
            base_model = ResNet152(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)

        # EfficientNet
        elif cnn_model == 'EfficientNetB0':
            base_model = EfficientNetB0(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'EfficientNetB1':
            base_model = EfficientNetB1(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'EfficientNetB2':
            base_model = EfficientNetB2(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        if cnn_model == 'EfficientNetB3':
            base_model = EfficientNetB3(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'EfficientNetB4':
            base_model = EfficientNetB4(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'EffcientNetB5':
            base_model = EfficientNetB5(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'EfficientNetB6':
            base_model = EfficientNetB6(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'EffcientNetB7':
            base_model = EfficientNetB7(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)

        # MobileNet
        if cnn_model == 'MobileNet':
            base_model = MobileNet(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'MobileNetV2':
            base_model = MobileNetV2(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)

        # VGG
        if cnn_model == 'VGG16':
            base_model = VGG16(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'VGG19':
            base_model = VGG19(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)

        # DenseNet
        if cnn_model == 'DenseNet121':
            base_model = DenseNet121(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'DenseNet169':
            base_model = VGG19(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'DenseNet201':
            base_model = DenseNet201(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        
        base_model.trainable = True

        # create top model
        inputs = base_model.input
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation=activation)(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        freeze_layer = False
        if freeze_layer:
            for layer in model.layers[0:6]:
                layer.trainable = False
            for layer in model.layers:
                print(layer, layer.trainable)

        model.summary()

    return model




