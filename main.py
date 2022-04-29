import os
import numpy as np
import pandas as pd
import seaborn as sn
import glob
from time import gmtime, strftime
from datetime import datetime
import timeit
import argparse
import random
import tensorflow as tf
from data_generator import train_generator
from data_generator import val_generator
from generate_model import generate_model
from train import train
from test import test
from opts import parse_opts
from statistics.get_stats_plots import get_stats_plots



def main(opt):

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
    else:
        print('get correct root dir to start.')

    # data generator for train and val data
    if opt.load_data:
        # data generator
        train_gen = train_generator(
            task=opt.task,
            pro_data_dir=opt.pro_data_dir,
            batch_size=opt.batch_size,
            channel=opt.channel)
        x_val, y_val, val_gen = val_generator(
            task=opt.task,
            pro_data_dir=opt.pro_data_dir,
            batch_size=opt.batch_size,
            channel=opt.channel)

    # get CNN model
    #cnns = ['simple_cnn', 'ResNet101V2', 'EfficientNetB4', 'MobileNetV2', 
    #        'DenseNet121', 'IncepttionV3', 'VGG16']
    cnns = ['ResNet101V2']
    for cnn_model in cnns:
        my_model = generate_model(
            cnn_model=opt.cnn_model,
            weights=opt.weights,
            freeze_layer=opt.freeze_layer,
            input_shape=opt.input_shape,
            activation=opt.activation,
            loss_function=opt.loss_function,
            lr=opt.lr)

        ## train model
        if opt.train:
            train(
                root_dir=opt.root_dir,
                out_dir=opt.out_dir,
                log_dir=opt.log_dir,
                model_dir=opt.model_dir,
                model=my_model,
                cnn_model=cnn_model,
                train_gen=train_gen,
                val_gen=val_gen,
                x_val=x_val,
                y_val=y_val,
                batch_size=opt.batch_size,
                epoch=opt.epoch,
                loss_function=opt.loss_function,
                lr=opt.lr,
                task=opt.task)
            print('training complete!')

    # test model
    if opt.test:
        loss, acc = test(
            task=opt.task,
            model=my_model,
            run_type=opt.run_type, 
            channel=opt.channel,
            model_dir=opt.model_dir, 
            pro_data_dir=opt.pro_data_dir, 
            saved_model=opt.saved_model,
            lr=opt.lr,
            loss_function=opt.loss_function,
            threshold=opt.thr_img, 
            activation=opt.activation)
        print('testing complete!')

    # get stats and plots
    if opt.stats_plots:
        get_stats_plots(
            task=opt.task,
            channel=opt.channel,
            pro_data_dir=opt.pro_data_dir,
            root_dir=opt.root_dir,
            run_type=opt.run_type,
            run_model=opt.cnn_model,
            loss=None,
            acc=None,
            saved_model=opt.cnn_model,
            epoch=opt.epoch,
            batch_size=opt.batch_size,
            lr=opt.lr,
            thr_img=opt.thr_img,
            thr_prob=opt.thr_prob,
            thr_pos=opt.thr_pos,
            bootstrap=opt.n_bootstrap)


if __name__ == '__main__':

    opt = parse_opts()

    main(opt)




