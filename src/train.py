import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from my_callback import my_callback
from logger import tr_logger
from utils.plot_train_curve import plot_train_curve
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoard


def train(log_dir, model, cnn_model, train_gen, val_gen, x_va, y_va, batch_size, epoch, loss_function, lr, cls_task): 
    """
    train model
    Args:
        model {cnn model} -- cnn model;
        run_model {str} -- cnn model name;
        train_gen {Keras data generator} -- training data generator with data augmentation;
        val_gen {Keras data generator} -- val data generator;
        x_val {np.array} -- np array for validation data;
        y_val {np.array} -- np array for validation label;
        batch_size {int} -- batch size for data loading;
        epoch {int} -- training epoch;
        out_dir {path} -- path for output files;
        opt {str or function} -- optimized function: 'adam';
        loss_func {str or function} -- loss function: 'binary_crossentropy';
        lr {float} -- learning rate;
    Returns:
        training accuracy, loss, model
    """
        ## save validation results to txt file 

    save_logger = True
    if save_logger:
        print('logging traininig process...')
        tr_logger(log_dir, cls_task, cnn_model, epoch, batch_size, lr, y_va)
    
    # model compile
    auc = tf.keras.metrics.AUC()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss_function,
        metrics=[auc])
     
    ## call back functions
    #check_point = tf.keras.callbacks.ModelCheckpoint(filepath=model_dir + '_{epoch:02d}_{val_auc:.2f}.h5',
     #   monitor='va_auc', save_best_only=True, save_weights_only=True, mode='max')  
    #early_stopping = EarlyStopping(monitor='va_auc', patience=100)
    tensor_board = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False, update_freq='epoch')
    callback = my_callback(model, log_dir, x_va, y_va)
    my_callbacks = [callback]
    #my_callbacks = [check_point, early_stopping, tensor_board]

    # fit models
    if cls_task == 'BRAF_status':
        class_weight = {0: 3, 1: 1}
    elif cls_task == 'BRAF_fusion':
        class_weight = {0: 2, 1: 1}
    elif cls_task == 'tumor':
        class_weight = {0: 1, 1: 6}
    elif cls_task == 'PFS_3yr':
        class_weight = {0: 5, 1: 3}
    elif cls_task == 'PFS_2yr':
        class_weight = {0: 1, 1: 1}

    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.n//batch_size,
        epochs=epoch,
        validation_data=val_gen,
        #validation_data=(x_val, y_val),
        validation_steps=val_gen.n//batch_size,
        #validation_steps=y_val.shape[0]//batch_size,
        verbose=1,
        callbacks=my_callbacks,
        validation_split=None,
        shuffle=True,
        class_weight=class_weight,
        sample_weight=None,
        initial_epoch=0)
    


    

    
