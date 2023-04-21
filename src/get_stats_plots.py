import os
import numpy as np
import pandas as pd
from statistics.cm_all import cm_all
from statistics.roc_all import roc_all
from statistics.prc_all import prc_all
#from utils.acc_loss import acc_loss
from logger import test_logger


def get_stats_plots(cls_task, cnn_model, channel, proj_dir, run_type, 
                    saved_model, epoch, batch_size, lr, thr_img, thr_prob, thr_pos, bootstrap):
    """
    generate model val/test statistics and plot curves;
    Args:
        loss {float} -- validation loss;
        acc {float} -- validation accuracy;
        run_model {str} -- cnn model name;
        batch_size {int} -- batch size for data loading;
        epoch {int} -- training epoch;
        out_dir {path} -- path for output files;
        opt {str or function} -- optimized function: 'adam';
        lr {float} -- learning rate;
    Keyword args:
        bootstrap {int} -- number of bootstrap to calculate 95% CI for AUC;
        thr_img {float} -- threshold to determine positive class on image level;
        thr_prob {float} -- threshold to determine positive class on patient 
                            level (mean prob score);
        thr_pos {float} -- threshold to determine positive class on patient 
                           level (positive class percentage);
    Returns:
       Model prediction statistics and plots: ROC, PRC, confusion matrix, etc.
    """
    tumor_cls_dir = proj_dir + '/tumor_cls'
    braf_cls_dir = proj_dir + '/braf_cls'
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
 
    if cls_task == 'tumor':
        data_dir = tumor_cls_dir
    elif cls_task in ['V600E', 'fusion', 'wild_type']:
        data_dir = braf_cls_dir
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

    cms = []
    cm_norms = []
    reports = []
    roc_stats = []
    prc_aucs = []
    if cls_task == 'tumor':
        levels = ['img']
    else:
        levels = levels = ['img', 'patient_mean_prob', 'patient_mean_pos']
    for level in levels:
        ## confusion matrix
        cm, cm_norm, report = cm_all(
            cls_task=cls_task,
            level=level,
            thr_img=thr_img,
            thr_prob=thr_prob,
            thr_pos=thr_pos,
            save_dir=save_dir,
            fn_df_pred=fn_pred)
        cms.append(cm)
        cm_norms.append(cm_norm)
        reports.append(report)

        ## ROC curves
        roc_stat = roc_all(
            cls_task=cls_task,
            level=level,
            bootstrap=bootstrap,
            color='blue',
            save_dir=save_dir,
            fn_df_pred=fn_pred)
        roc_stats.append(roc_stat)

        ## PRC curves
        prc_auc = prc_all(
            cls_task=cls_task,
            level=level,
            color='red',
            save_dir=save_dir,
            fn_df_pred=fn_pred)
        prc_aucs.append(prc_auc)

    ### save validation results to txt
    test_logger(
        cls_task=cls_task,
        run_type=run_type,
        save_dir=save_dir,
        cms=cms,
        cm_norms=cm_norms,
        reports=reports,
        prc_aucs=prc_aucs,
        roc_stats=roc_stats,
        cnn_model=cnn_model,
        saved_model=saved_model,
        epoch=epoch,
        batch_size=batch_size,
        lr=lr)

    print('saved model as:', saved_model)

