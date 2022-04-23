import os
import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.applications import ResNet152V2



def ResNet(resnet, transfer, freeze_layer, input_shape, activation):

    """
    ResNet from Keras

    Args:
      resnet       - required : name of resnets with different layers, i.e. 'ResNet101'.
      transfer     - required : decide if want do transfer learning
      model_dir    - required : folder path to save model
      freeze_layer - required : number of layers to freeze
      activation   - required : activation function in last layer
    
    """

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
    
    ## determine n_output
    if activation == 'softmax':
        n_output = 2
    elif activation == 'sigmoid':
        n_output = 1

    ### determine ResNet base model
    if resnet == 'ResNet50V2':
        base_model = ResNet50V2(weights=None, include_top=include_top,
            input_shape=input_shape, pooling=None)                
    elif resnet == 'ResNet101V2':
        base_model = ResNet101V2(weights=None, include_top=include_top,
            input_shape=input_shape, pooling=None)                
    elif resnet == 'ResNet152V2': 
        base_model = ResNet152V2(weights=None, include_top=include_top,
            input_shape=input_shape, pooling=None)               
    if resnet == 'ResNet50':
        base_model = ResNet50(weights=None, include_top=include_top,
            input_shape=input_shape, pooling=None)
    elif resnet == 'ResNet101':
        base_model = ResNet101(weights=None, include_top=include_top,
            input_shape=input_shape, pooling=None)       
    elif resnet == 'ResNet152':
        base_model = ResNet152(weights=None, include_top=include_top,
            input_shape=input_shape, pooling=None)

    ### create top model
    inputs = base_model.input
    x = base_model.output  
    x = GlobalAveragePooling2D()(x)
    x = Dense(1000, activation='relu')(x)
    outputs = Dense(n_output, activation=activation)(x)
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





    

    
