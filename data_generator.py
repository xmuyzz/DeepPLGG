import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from PIL import Image
import glob
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def train_generator(cls_task, tumor_cls_dir, braf_cls_dir, batch_size, channel):
    """
    create data generator for training dataset;
    Arguments:
        out_dir {path} -- path to output results;
        batch_size {int} -- batch size for data generator;
        input_channel {int} -- input channel for image;
    Return:
        Keras data generator;
    """
    ### load train data based on input channels

    # classification task
    if cls_task == 'tumor':
        data_dir = tumor_cls_dir
    elif cls_task in ['V600E', 'fusion', 'wild_type']:
        data_dir = braf_cls_dir
    else:
        print('wrong classification task!')
    
    # image channel
    if channel == 1:
        fn = 'tr_arr_1ch.npy'
        #fn = 'va_arr_1ch.npy'
    elif channel == 3:
        fn = 'tr_arr_3ch.npy'

    # load numpy array and labels
    x_tr = np.load(data_dir + '/' + fn)
    df = pd.read_csv(data_dir + '/' + '/tr_img_df.csv')
    #df = pd.read_csv(data_dir + '/' + '/va_img_df.csv')
    y_tr = np.asarray(df[cls_task]).astype('int').reshape((-1, 1))

    ## data generator
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=None,
        shear_range=0.0,
        zoom_range=0,
        channel_shift_range=0.0,
        fill_mode='nearest',
        cval=0.0,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0,
        dtype=None)

   ### Train generator
    tr_gen = datagen.flow(
        x=x_tr,
        y=y_tr,
        subset=None,
        batch_size=batch_size,
        seed=42,
        shuffle=True)
    print('Train generator created!')

    return tr_gen


def val_generator(cls_task, tumor_cls_dir, braf_cls_dir, batch_size, channel):
    """
    create data generator for validation dataset;
    Arguments:
        out_dir {path} -- path to output results;
        batch_size {int} -- batch size for data generator;
        input_channel {int} -- input channel for image;
    Return:
    Keras data generator;
    """
    ### load val data based on input channels
    # classification task
    if cls_task == 'tumor':
        data_dir = tumor_cls_dir
    elif cls_task in ['V600E', 'fusion', 'wild_type']:
        data_dir = braf_cls_dir
    else:
        print('wrong classification task!')
    
    # image channel
    if channel == 1:
        fn = 'va_arr_1ch.npy'
    elif channel == 3:
        fn = 'va_arr_3ch.npy'

    # load numpy array and labels
    x_va = np.load(data_dir + '/' + fn)
    df = pd.read_csv(data_dir + '/' + '/va_img_df.csv')
    y_va = np.asarray(df[cls_task]).astype('int').reshape((-1, 1))

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        brightness_range=None,
        shear_range=0.0,
        zoom_range=0,
        channel_shift_range=0.0,
        fill_mode='nearest',
        cval=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0,
        dtype=None)
    
    datagen = ImageDataGenerator()
    va_gen = datagen.flow(
        x=x_va,
        y=y_va,
        subset=None,
        batch_size=batch_size,
        seed=42,
        shuffle=True)
    print('test generator created!')

    return x_va, y_va, va_gen



