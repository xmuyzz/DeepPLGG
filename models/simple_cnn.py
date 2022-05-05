import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import glob
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy




def simple_cnn(input_shape, activation):

    """
    simple CNN model

    @params:
      lr           - required : learning rate
      loss_fn      - required : loss function
      opt          - required : optimizer, 'adam', 'sgd'...
      activation   - required : activation function in last layer, 'relu', 'elu'...

    """
    
    ## determine n_output
    if activation == 'softmax':
        n_output = 2
    elif activation == 'sigmoid':
        n_output = 1

    model = Sequential()

    model.add(Conv2D(16, kernel_size=(3, 3), activation='leaky_relu', input_shape=input_shape))
    model.add(BatchNormalization(momentum=0.95))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.3))

    model.add(BatchNormalization(momentum=0.95))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='leaky_relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.3))
    
    model.add(BatchNormalization(momentum=0.95))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='leaky_relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.3))

    model.add(BatchNormalization(momentum=0.95))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='leaky_relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(BatchNormalization(momentum=0.95))
    model.add(Dense(1024, activation='leaky_relu'))
    model.add(Dropout(0.3))
    
    model.add(BatchNormalization(momentum=0.95))
    model.add(Dense(256, activation='leaky_relu'))
    model.add(Dropout(0.3))
    model.add(Dense(n_output, activation=activation))

    model.summary()

    return model




    
