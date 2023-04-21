import os
import numpy as np
import pandas as pd
import pickle
from time import gmtime, strftime
from datetime import datetime
import timeit
from utils.cm_all import cm_all
from utils.roc_all import roc_all
from utils.prc_all import prc_all
from utils.acc_loss import acc_loss
from utils.write_txt import write_txt

#--------------------------------------------------------------------------
# make plots
#-------------------------------------------------------------------------

def make_plots(run_type, thr_img, thr_prob, thr_pos, bootstrap, pro_data_dir, 
               save_dir, loss, acc, run_model, saved_model, epoch, batch_size, lr):

    ### determine if this is train or test
    if run_type == 'val':
        fn_df_pred = 'val_img_pred.csv'
        save_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/val'
    elif run_type == 'test':
        fn_df_pred = 'test_img_pred.csv'
        save_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/test'
    elif run_type == 'exval':
        fn_df_pred = 'exval_img_pred.csv'
        save_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/exval'
    elif run_type == 'exval2':
        fn_df_pred = 'rtog_img_pred.csv'
        save_dir = '/mnt/aertslab/USERS/Zezhong/contrast_detection/exval2'

    cms = []
    cm_norms = []
    reports = []
    stats = []
    prc_aucs = []
    levels = ['img', 'patient_mean_prob', 'patient_mean_pos']

    for level in levels:
        
        ## confusion matrix
        cm, cm_norm, report = cm_all(
            run_type=run_type,
            level=level,
            thr_img=thr_img,
            thr_prob=thr_prob,
            thr_pos=thr_pos,
            pro_data_dir=pro_data_dir,
            save_dir=save_dir,
            fn_df_pred=fn_df_pred
            )
        cms.append(cm)
        cm_norms.append(cm_norm)
        reports.append(report)

        ## ROC curves
        stat = roc_all(
            run_type=run_type,
            level=level,
            thr_prob=thr_prob,
            thr_pos=thr_pos,
            bootstrap=bootstrap,
            color='blue',
            pro_data_dir=pro_data_dir,
            save_dir=save_dir,
            fn_df_pred=fn_df_pred
            )
        stats.append(stat)

        ## PRC curves
        prc_auc = prc_all(
            run_type=run_type,
            level=level,
            thr_prob=thr_prob,
            thr_pos=thr_pos,
            color='red',
            pro_data_dir=pro_data_dir,
            save_dir=save_dir,
            fn_df_pred=fn_df_pred
            )
        prc_aucs.append(prc_auc)

    ### save validation results to txt
    write_txt(
        run_type=run_type,
        save_dir=save_dir,
        loss=loss,
        acc=acc,
        cm1=cms[0],
        cm2=cms[1],
        cm3=cms[2],
        cm_norm1=cm_norms[0],
        cm_norm2=cm_norms[1],
        cm_norm3=cm_norms[2],
        report1=reports[0],
        report2=reports[1],
        report3=reports[2],
        prc_auc1=prc_aucs[0],
        prc_auc2=prc_aucs[1],
        prc_auc3=prc_aucs[2],
        stat1=stats[0],
        stat2=stats[1],
        stat3=stats[2],
        run_model=run_model,
        saved_model=saved_model,
        epoch=epoch,
        batch_size=batch_size,
        lr=lr
        )
    print('saved model as:', saved_model)
    print('session end!!!')
    print('\n')

if __name__ == '__main__':

    output_dir      = '/media/bhkann/HN_RES1/HN_CONTRAST/output'
    train_img_dir   = '/media/bhkann/HN_RES1/HN_CONTRAST/train_img_dir'
    val_img_dir     = '/media/bhkann/HN_RES1/HN_CONTRAST/val_img_dir'
    test_img_dir    = '/media/bhkann/HN_RES1/HN_CONTRAST/test_img_dir'
    test_save_dir   = '/mnt/aertslab/USERS/Zezhong/constrast_detection/test'
    val_save_dir    = '/mnt/aertslab/USERS/Zezhong/constrast_detection/val'
    bootstrap       = 1000
    threshold       = 0.5
    color_roc       = 'blue'
    color_prc       = 'red'
    input_channel   = 3
    crop            = True
    thr_img         = 0.5
    thr_img         = 0.5
    thr_prob        = 0.5
    thr_pos         = 0.5
    run_type        = 'exval'
    run_model       = 'ResNet'
    saved_model     = 'ResNet_2021_06_20_03_27_27'
    epoch           = 500
    batch_size      = 34
    
    start = timeit.default_timer()
    
    make_plots()
    
    stop = timeit.default_timer()
    print('Run Time:', np.around((stop - start)/60, 0), 'mins')


