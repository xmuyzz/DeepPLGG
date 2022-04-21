import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib
import matplotlib.pyplot as plt
import glob
from time import gmtime, strftime
from datetime import datetime
import timeit
import argparse
import pydot
import pydotplus
import graphviz
import random
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from data_generator import train_generator
from data_generator import test_generator
from generate_model import generate_model
from train import train
from callbacks import callbacks
from opts import parse_opts



if __name__ == '__main__':

    opt = parse_opts()

    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    tf.random.set_seed(opt.manual_seed)

    if opt.root_dir is not None:
        opt.out_dir = os.path.join(opt.root_dir, opt.out)
        opt.model_dir = os.path.join(opt.root_dir, opt.model)
        opt.pro_data_dir = os.path.join(opt.root_dir, opt.pro_data)
        opt.log_dir = os.path.join(opt.root_dir, opt.log)
        if not os.path.exists(opt.out_dir):
            os.makedirs(opt.out_dir)
        if not os.path.exists(opt.model_dir):
            os.makedirs(opt.model_dir)
        if not os.path.exists(opt.pro_data_dir):
            os.makefirs(opt.pro_data_dir)
        if not os.path.exists(opt.log_dir):
            os.makedirs(opt.log_dir)

    # data generator for train and val data
    train_gen = train_generator(
        pro_data_dir=opt.pro_data_dir,
        batch_size=opt.batch_size)
    x_test, y_test, test_gen = test_generator(
        pro_data_dir=opt.pro_data_dir,
        batch_size=opt.batch_size)

    # get CNN model 
    my_model = generate_model(
        out_dir=opt.out_dir,
        cnn_model=opt.cnn_model, 
        activation=opt.activation, 
        input_shape=opt.input_shape)

    ## train model
    if opt.train:
        train(
            root_dir=opt.root_dir,
            out_dir=opt.out_dir,
            log_dir=opt.og_dir,
            model_dir=opt.model_dir,
            model=my_model,
            cnn_model=opt.cnn_model,
            train_gen=train_gen,
            val_gen=val_gen,
            batch_size=opt.batch_size,
            epoch=opt.epoch,
            optimizer=optimizer,
            loss_function=opt.loss_function,
            lr=opt.lr)
    # test model
    if opt.test:
        test(
            run_type=opt.run_type, 
            model_dir=opt.model_dir, 
            pro_data_dir=opt.pro_data_dir, 
            saved_model=opt.saved_model, 
            threshold=opt.img_threshold, 
            activation=opt.activation)




