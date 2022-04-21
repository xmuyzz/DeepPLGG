import os
import numpy as np
import pandas as pd
import seaborn as s1n
import matplotlib.pyplot as plt
import glob
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import InceptionResNetV2



def EfficientNet(effnet, transfer, freeze_layer, input_shape, activation):


    """
    EfficientNet

    @params:
      effnet       - required : name of resnets with different layers, i.e. 'ResNet101'.
      transfer     - required : decide if want do transfer learning
      model_dir    - required : folder path to save model
      freeze_layer - required : number of layers to freeze
      activation   - required : activation function in last layer
    
    """

    ### Keras CNN models for use: https://keras.io/api/applications/
    ### InceptionV3(top1 acc 0.779)
    ### InceptionResnetV2(top1 acc 0.803),
    ### ResNet152V2(top1 acc 0.780)
    ### EficientNetB4

 ### determine if use transfer learnong or not
    if transfer == True:
        weights = 'imagenet'
    elif transfer == False:
        weights = None

    ### determine input shape
    default_shape = (224, 224, 3)
    if input_shape == default_shape:
        include_top = True
    else:
        include_top = False

    ### determine ResNet base model
    if effnet == 'EffNetB3':
        base_model = EfficientNetB3(
            weights=weights,
            include_top=include_top,
            input_shape=input_shape,
            pooling=None
            )
    elif effnet == 'EffNetB4':
        base_model = EfficientNetB4(
            weights=weights,
            include_top=include_top,
            input_shape=input_shape,
            pooling=None
            )
    elif effnet == 'EffNetB5':
        base_model = EfficientNetB5(
            weights=weights,
            include_top=include_top,
            input_shape=input_shape,
            pooling=None
            )
    base_model.trainable = True

 ### create top model
    inputs = base_model.input
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(1000, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation=activation)(x)
    model = Model(inputs=inputs, outputs=outputs)

    ### freeze specific number of layers
    if freeze_layer == 1:
        for layer in base_model.layers[0:5]:
            layer.trainable = False
        for layer in base_model.layers:
            print(layer, layer.trainable)
    if freeze_layer == 5:
        for layer in base_model.layers[0:16]:
            layer.trainable = False
        for layer in base_model.layers:
            print(layer, layer.trainable)
    else:
        for layer in base_model.layers:
            layer.trainable = True
    model.summary()


    return model
    

    
