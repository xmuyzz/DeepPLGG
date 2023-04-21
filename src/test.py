import os
import timeit
import numpy as np
import pandas as pd
import glob
from datetime import datetime
from time import gmtime, strftime
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, roc_auc_score


def test(proj_dir, cls_task, model, cnn_model, run_type, channel, saved_model, lr, loss_function,
         threshold=0.5, activation='sigmoid', load_model_type='weights'):    
    """
    Evaluate model for validation/test/external validation data;
    Args:
        out_dir {path} -- path to main output folder;
        proj_dir {path} -- path to main project folder;
        saved_model {str} -- saved model name;
        tuned_model {Keras model} -- finetuned model for chest CT;
    Keyword args:
        threshold {float} -- threshold to determine postive class;
        activation {str or function} -- activation function, default: 'sigmoid';
    Returns:
        training accuracy, loss, model
    """

    """ classification task:
         1) tumor vs. benign; 
         2) classify BRAF-V600E vs. BRAF-Fusion & BRAF-Wild Type;
    """

    # create data dir and file names
    #-------------------------------
    # model dirs
    log_dir = proj_dir + '/log/' + cls_task + '/' + cnn_model
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # output dirs
    tr_dir = proj_dir + '/output/' + cls_task + '/' + cnn_model + '/train'
    va_dir = proj_dir + '/output/' + cls_task + '/' + cnn_model + '/val'
    ts_dir = proj_dir + '/output/' + cls_task + '/' + cnn_model + '/test'
    tx_dir = proj_dir + '/output/' + cls_task + '/' + cnn_model + '/external'
    if not os.path.exists(tr_dir):
        os.makedirs(tr_dir)
    if not os.path.exists(va_dir):
        os.makedirs(va_dir)
    if not os.path.exists(ts_dir):
        os.makedirs(ts_dir)
    if not os.path.exists(tx_dir):
        os.makedirs(tx_dir)
    
    # data dirs
    if cls_task == 'tumor':
        data_dir = proj_dir + '/tumor_cls'
    elif cls_task in ['V600E', 'fusion', 'wild_type']:
        data_dir = proj_dir + '/braf_cls_dir'
    else:
        print('wrong classification task!')

    # run type: validation, test, external test
    if run_type == 'val':
        if channel == 1:
            fn_data = 'va_arr_1ch.npy'
        elif channel == 3:
            fn_data = 'va_arr_3ch.npy'
        fn_label = 'va_img_df.csv'
        fn_pred = 'va_img_pred.csv'
        save_dir = va_dir
    elif run_type == 'test':
        if channel == 1:
            fn_data = 'ts_arr_1ch.npy'
        elif channel == 3:
            fn_data = 'ts_arr_3ch.npy'
        fn_label = 'ts_img_df.csv'
        fn_pred = 'ts_img_pred.csv'
        save_dir = ts_dir
    elif run_type == 'external':
        if channel == 1:
            fn_data = 'tx_arr_1ch.npy'
        elif channel == 3:
            fn_data = 'tx_arr_3ch.npy'
        fn_label = 'tx_img_df.csv'
        fn_pred = 'tx_img_pred.csv'
        save_dir = tx_dir

    # load numpy array and labels
    x_data = np.load(data_dir + '/' + fn_data)
    df = pd.read_csv(data_dir + '/' + fn_label)
    y_label = np.asarray(df[cls_task]).astype('int').reshape((-1, 1))

    # test model
    #------------------------------
    if load_model_type == 'load_model':
        model = load_model(os.path.join(log_dir, saved_model))
    elif load_model_type == 'load_weights':    # model compile
        auc = tf.keras.metrics.AUC()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=loss_function,
            metrics=[auc])
        model.load_weights(os.path.join(log_dir, saved_model))
    model.load_weights(os.path.join(log_dir, saved_model))
    y_pred = model.predict(x_data)
    # score = model.evaluate(x_data, y_label)
    # loss = np.around(score[0], 3)
    # acc = np.around(score[1], 3)
    # print('loss:', loss)
    # print('acc:', acc)
    auc = roc_auc_score(y_label, y_pred)
    auc = np.around(auc, 3)
    print('auc:', auc)
    
    if activation == 'sigmoid':
        y_pred = model.predict(x_data)
        y_pred_class = [1 * (x[0] >= threshold) for x in y_pred]
    elif activation == 'softmax':
        y_pred_prob = model.predict(x_data)
        y_pred = y_pred_prob[:, 1]
        y_pred_class = np.argmax(y_pred_prob, axis=1)

    # save a dataframe
    #------------------
    ID = []
    for file in df['fn']:
        if run_type in ['val', 'test', 'external']:
            id = file.split('\\')[-1].split('_')[0].strip()
        elif run_type == 'tune2':
            id = file.split('\\')[-1].split('_s')[0].strip()
        ID.append(id)
    df['ID'] = ID
    df['y_pred'] = y_pred
    df['y_pred_class'] = y_pred_class
    #df_test_pred = df[['ID', 'fn', 'label', 'y_pred', 'y_pred_class']]
    df.to_csv(save_dir + '/' + fn_pred, index=False)
    

        



    

    
