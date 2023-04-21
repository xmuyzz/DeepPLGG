import os
import numpy as np
from tensorflow.keras.utils import plot_model
from models.simple_cnn import simple_cnn
from models.EfficientNet import EfficientNet
from models.ResNet import ResNet
from models.Inception import Inception
from models.VGGNet import VGGNet
from models.TLNet import TLNet




def generate_model(out_dir, cnn_model, activation, input_shape, freeze_layer=None, transfer=False):
    
    """
    generate cnn models

    Args:
        run_model {str} -- choose specific CNN model type;
        activation {str or function} -- activation function in last layer: 'sigmoid', 'softmax', etc;
    
    Keyword args:i
        input_shape {np.array} -- input data shape;
        transfer {boolean} -- decide if transfer learning;
        freeze_layer {int} -- number of layers to freeze;
    
    Returns:
        deep learning model;
    
    """
    
    # simple cnn
    if cnn_model == 'simple_cnn':
        my_model = simple_cnn(input_shape=input_shape, activation=activation)

    # ResNet 50, 101, 152
    elif cnn_model == 'ResNet50':
        my_model = ResNet(resnet='ResNet50', transfer=transfer, freeze_layer=freeze_layer,
            input_shape=input_shape, activation=activation)
    elif cnn_model == 'ResNet101':
        my_model = ResNet(resnet='ResNet101', transfer=transfer, freeze_layer=freeze_layer,
            input_shape=input_shape, activation=activation)
    elif cnn_model == 'ResNet152':
        my_model = ResNet(resnet='ResNet152', transfer=transfer, freeze_layer=freeze_layer,
            input_shape=input_shape, activation=activation)
    elif cnn_model == 'ResNet50V2':
        my_model = ResNet(resnet='ResNet50V2', transfer=transfer, freeze_layer=freeze_layer,
            input_shape=input_shape, activation=activation)
    elif cnn_model == 'ResNet101V2':
        my_model = ResNet(resnet='ResNet101V2', transfer=transfer, freeze_layer=freeze_layer,
            input_shape=input_shape, activation=activation)
    elif cnn_model == 'ResNet152V2':
        my_model = ResNet(resnet='ResNet152V2', transfer=transfer, freeze_layer=freeze_layer,
            input_shape=input_shape, activation=activation)
    
    # EfficientNet b0 - b7
    elif cnn_model == 'EfficientNetB0':
        my_model = EfficientNet(effnet='EfficientNetB0', transfer=transfer,
            freeze_layer=freeze_layer, input_shape=input_shape, activation=activation)
    elif cnn_model == 'EfficientNetB1':
        my_model = EfficientNet(effnet='EfficientNetB1', transfer=transfer,
            freeze_layer=freeze_layer, input_shape=input_shape, activation=activation)
    elif cnn_model == 'EfficientNetB2':
        my_model = EfficientNet(effnet='EfficientNetB2', transfer=transfer,
            freeze_layer=freeze_layer, input_shape=input_shape, activation=activation)
    elif cnn_model == 'EfficientNetB3':
        my_model = EfficientNet(effnet='EfficientNetB3', transfer=transfer,
            freeze_layer=freeze_layer, input_shape=input_shape, activation=activation)
    elif cnn_model == 'EfficientNetB4':
        my_model = EfficientNet(effnet='EfficientNetB4', transfer=transfer,
            freeze_layer=freeze_layer, input_shape=input_shape, activation=activation)
    elif cnn_model == 'EfficientNetB5':
        my_model = EfficientNet(effnet='EfficientNetB5', transfer=transfer,
            freeze_layer=freeze_layer, input_shape=input_shape, activation=activation)
    elif cnn_model == 'EfficientNetB6':
        my_model = EfficientNet(effnet='EfficientNetB6', transfer=transfer,
            freeze_layer=freeze_layer, input_shape=input_shape, activation=activation)
    elif cnn_model == 'EfficientNetB7':
        my_model = EfficientNet(effnet='EfficientNetB7', transfer=transfer,
            freeze_layer=freeze_layer, input_shape=input_shape, activation=activation)

    # MobileNet V1, V2
    elif cnn_model == 'MobileNet':
        my_model = EfficientNet(effnet='MobileNet', transfer=transfer,
            freeze_layer=freeze_layer, input_shape=input_shape, activation=activation)
    elif cnn_model == 'MobileNetV2':
        my_model = EfficientNet(effnet='MobileNetV2', transfer=transfer,
            freeze_layer=freeze_layer, input_shape=input_shape, activation=activation)

    # DenseNet 121, 169, 201
    elif cnn_model == 'DenseNet121':
        my_model = EfficientNet(effnet='DenseNet121', transfer=transfer,
            freeze_layer=freeze_layer, input_shape=input_shape, activation=activation)
    elif cnn_model == 'DenseNet169':
        my_model = EfficientNet(effnet='DenseNet169', transfer=transfer,
            freeze_layer=freeze_layer, input_shape=input_shape, activation=activation)
    elif cnn_model == 'DenseNet201':
        my_model = EfficientNet(effnet='DenseNet201', transfer=transfer,
            freeze_layer=freeze_layer, input_shape=input_shape, activation=activation)

    # GoogLeNet
    elif cnn_model == 'InceptionV3':
        my_model = EfficientNet(effnet='InceptionV3', transfer=transfer,
            freeze_layer=freeze_layer, input_shape=input_shape, activation=activation)
    elif cnn_model == 'Xception':
        my_model = EfficientNet(effnet='Xception', transfer=transfer,
            freeze_layer=freeze_layer, input_shape=input_shape, activation=activation)
    
    # VGG
    elif cnn_model == 'VGG16':
        my_model = EfficientNet(effnet='VGG16', transfer=transfer,
            freeze_layer=freeze_layer, input_shape=input_shape, activation=activation)
    elif cnn_model == 'VGG19':
        my_model = EfficientNet(effnet='VGG19', transfer=transfer,
            freeze_layer=freeze_layer, input_shape=input_shape, activation=activation)

    # Transfer learning
    elif cnn_model == 'TLNet':
        my_model = TLNet(resnet='ResNet101V2', input_shape=input_shape, activation=activation)

    return my_model




