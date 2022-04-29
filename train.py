import os
import numpy as np
import pandas as pd
import seaborn as sn
import glob
from datetime import datetime
from time import localtime, strftime
import tensorflow as tf
from tensorflow.keras.models import Model
from callback import callback
from utils.plot_train_curve import plot_train_curve
from tensorflow.keras.optimizers import Adam
from statistics.write_txt import write_txt
from tensorflow.keras.callbacks import (
    EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoard)



def train(root_dir, out_dir, log_dir, model_dir, model, cnn_model, train_gen, 
          val_gen, x_val, y_val, batch_size, epoch, loss_function, lr, task): 

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
    

    ## call back functions
    check_point = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, '{epoch:02d}-{val_auc:.2f}.h5'),
        monitor='val_auc',  
        save_best_only=True,
        save_weights_only=True,
        mode='max')  # determine better models according to "max" AUC.
    early_stopping = EarlyStopping(
        monitor='val_auc',
        min_delta=0,
        patience=100,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=False)
    tensor_board = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        update_freq='epoch',
        profile_batch=2,
        embeddings_freq=0,
        embeddings_metadata=None)
    my_callbacks = [check_point, early_stopping, tensor_board]
    
    # model compile
    auc = tf.keras.metrics.AUC()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss_function,
        metrics=[auc])
    model.load_weights(os.path.join(model_dir, '33-0.91.h5'))
    ## fit models
    if task == 'BRAF_status':
        class_weight = {0: 3, 1: 1}
    elif task == 'BRAF_fusion':
        class_weight = {0: 2, 1: 1}
    elif task == 'tumor':
        class_weight = {0: 6, 1: 1}
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
    
    ## valudation acc and loss
    score = model.evaluate(x_val, y_val)
    loss = np.around(score[0], 3)
    acc = np.around(score[1], 3)
    print('val loss:', loss)
    print('val acc:', acc)

    ## save final model
    saved_model = str(cnn_model) + 'final.h5'
    model.save_weights(os.path.join(model_dir, saved_model), save_format='h5',)
    print(saved_model)
    
    ## save validation results to txt file 
    _write_txt = False
    if _write_txt:
        write_txt(
            run_type='train',
            root_dir=root_dir,
            loss=1,
            acc=1,
            cms=None,
            cm_norms=None,
            reports=None,
            prc_aucs=None,
            roc_stats=None,
            run_model=cnn_model,
            saved_model=saved_model,
            epoch=epoch,
            batch_size=batch_size,
            lr=lr)

    


    

    
