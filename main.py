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
from opts import parse_opts



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
    train_gen = train_generator(
        pro_data_dir=opt.pro_data_dir,
        batch_size=opt.batch_size,
        channel=opt.channel)

    x_val, y_val, val_gen = val_generator(
        pro_data_dir=opt.pro_data_dir,
        batch_size=opt.batch_size,
        channel=opt.channel)

    # get CNN model 
    my_model = generate_model(
        out_dir=opt.out_dir,
        cnn_model=opt.cnn_model, 
        activation=opt.activation, 
        input_shape=opt.input_shape)

    ## train model
    if opt.train:
        cnn_model = 'simple_cnn'
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


if __name__ == '__main__':

    opt = parse_opts()

    main(opt)




