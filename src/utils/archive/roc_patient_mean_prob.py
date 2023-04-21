#----------------------------------------------------------------------
# Deep learning for classification for contrast CT;
# Transfer learning using Google Inception V3;
#-----------------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import pickle
from utils.mean_CI import mean_CI
from utils.plot_roc import plot_roc
from utils.roc_bootstrap import roc_bootstrap

# ----------------------------------------------------------------------------------
# plot ROI
# ----------------------------------------------------------------------------------
def roc_patient_mean_prob(run_type, output_dir, roc_fn, color, bootstrap, save_dir):
    
    ### determine if this is train or test
    if run_type == 'train' or run_type == 'val':
        df_sum = pd.read_pickle(os.path.join(save_dir, 'df_val_pred.p'))
    elif run_type == 'test':
        df_sum = pd.read_pickle(os.path.join(save_dir, 'df_test_pred.p'))
    ### determine if use mean values for patient-level prob scores
    df_mean = df_sum.groupby(['ID']).mean()
    y_true = df_mean['label'].to_numpy()
    y_pred = df_mean['y_pred'].to_numpy()
    pd.options.display.max_columns = 100
    pd.set_option('display.max_rows', 500)
    #print(df_sum[600:800])
    #print('y true:', y_true[0:25])
    ### plot roc curve
    auc2 = plot_roc(
        save_dir=save_dir,
        y_true=y_true,
        y_pred=y_pred,
        roc_fn=roc_fn,
        color=color
        )
    ### calculate roc, tpr, tnr with 1000 bootstrap
    stat2 = roc_bootstrap(
        bootstrap=bootstrap,
        y_true=y_true,
        y_pred=y_pred
        )

    print('roc patient mean prob:')
    print(auc2)
    print(stat2)

    return auc2, stat2


    

   
