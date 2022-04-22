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
    
    if cnn_model == 'cnn':
        my_model = simple_cnn(
            input_shape=input_shape,
            activation=activation)
    elif cnn_model == 'ResNet50V2':
        my_model = ResNet(
            resnet='ResNet50V2',
            transfer=transfer,
            freeze_layer=freeze_layer,
            input_shape=input_shape,
            activation=activation)
    elif cnn_model == 'ResNet101V2':
        my_model = ResNet(
            resnet='ResNet101V2',
            transfer=transfer,
            freeze_layer=freeze_layer,
            input_shape=input_shape,
            activation=activation)
    elif cnn_model == 'EfficientNetB4':
        my_model = EfficientNet(
            effnet='EfficientNetB4',
            transfer=transfer,
            freeze_layer=freeze_layer,
            input_shape=input_shape,
            activation=activation)
    elif cnn_model == 'TLNet':
        my_model = TLNet(
            resnet='ResNet101V2',
            input_shape=input_shape,
            activation=activation)
    elif cnn_model == 'InceptionV3':
        my_model = Inception(
            inception='InceptionV3',
            transfer=transfer,
            freeze_layer=freeze_layer,
            input_shape=input_shape,
            activation=activation)

    return my_model




