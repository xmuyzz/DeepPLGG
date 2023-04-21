import os
import numpy as np
import random
import tensorflow as tf
from data_generator import train_generator
from data_generator import val_generator
from get_model import get_model
from transfer_model import transfer_model
from train import train
from test import test
from opts import parse_opts
from get_stats_plots import get_stats_plots


def main(opt):

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # physical_devices = tf.config.experimental.list_physical_devices('CPU')
    print('physical_devices-------------', len(physical_devices))
    #tf.config.experimental.set_memory_growth(physical_devices[3], True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    tf.random.set_seed(opt.manual_seed)

    if opt.proj_dir is not None:
        out_dir = opt.proj_dir + '/output/' + opt.cls_task + '/' + opt.cnn_model
        log_dir = opt.proj_dir + '/log/' + opt.cls_task + '/' + opt.cnn_model
        tumor_cls_dir = opt.proj_dir + '/tumor_cls' 
        braf_cls_dir = opt.proj_dir + '/braf_cls'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    else:
        print('get correct root dir to start.')

    # data generator for train and val data
    if opt.load_data:
        # data generator
        train_gen = train_generator(
            cls_task=opt.cls_task,
            tumor_cls_dir=tumor_cls_dir,
            braf_cls_dir=braf_cls_dir,
            batch_size=opt.batch_size,
            channel=opt.channel)
        x_val, y_val, val_gen = val_generator(
            cls_task=opt.cls_task,
            tumor_cls_dir=tumor_cls_dir,
            braf_cls_dir=braf_cls_dir,
            batch_size=opt.batch_size,
            channel=opt.channel)
        print('train and val dataloader sucessful!')

    # get CNN model
    #cnns = ['ResNet50', 'EfficientNetB4', 'MobileNet', 
    #        'DenseNet121', 'IncepttionV3', 'VGG16']
    if opt.load_model:
        if opt.transfer_learning:
            my_model = transfer_model(
                cnn_model=opt.cnn_model,
                input_shape=opt.input_shape,
                activation=opt.activation,
                freeze_layer=opt.freeze_layer,
                model_dir=opt.model_dir,
                trained_weights=opt.trained_weights,
                saved_model=opt.saved_model,
                tune_step=opt.tune_step)
        else:
            my_model = get_model(
                cnn_model=opt.cnn_model,
                input_shape=opt.input_shape,
                activation=opt.activation)
            # train model
            if opt.train:
                train(
                    log_dir=log_dir,
                    model=my_model,
                    cnn_model=opt.cnn_model,
                    train_gen=train_gen,
                    val_gen=val_gen,
                    x_va=x_val,
                    y_va=y_val,
                    batch_size=opt.batch_size,
                    epoch=opt.epoch,
                    loss_function=opt.loss_function,
                    lr=opt.lr,
                    cls_task=opt.cls_task)
                print('training complete!')

    # test model
    if opt.test:
        test(
            proj_dir=opt.proj_dir,
            cls_task=opt.cls_task,
            model=my_model,
            cnn_model=opt.cnn_model,
            run_type=opt.run_type, 
            channel=opt.channel,
            saved_model=opt.saved_model,
            lr=opt.lr,
            loss_function=opt.loss_function,
            threshold=opt.thr_img, 
            activation=opt.activation,
            load_model_type=opt.load_model_type)
        print('testing complete!')

    # get stats and plots
    if opt.stats_plots:
        get_stats_plots(
            cls_task=opt.cls_task,
            cnn_model=opt.cnn_model,
            channel=opt.channel,
            proj_dir=opt.proj_dir,
            run_type=opt.run_type,
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




