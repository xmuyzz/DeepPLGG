#----------------------------------------------------------------------
# Deep learning for classification for contrast CT;
# Transfer learning using Google Inception V3;
#-----------------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import pickle
from utils.plot_roc import plot_roc
from utils.roc_bootstrap import roc_bootstrap

# ----------------------------------------------------------------------------------
# plot ROI
# ----------------------------------------------------------------------------------
def roc_patient_median_prob(run_type, output_dir, roc_fn, color, bootstrap, save_dir):
    
    ### determine if this is train or test
    if run_type == 'train' or run_type == 'val':
        df_sum = pd.read_pickle(os.path.join(save_dir, 'df_val_pred.p'))
    elif run_type == 'test':
        df_sum = pd.read_pickle(os.path.join(save_dir, 'df_test_pred.p'))
    ### determine if use mean values for patient-level prob scores
    df_median = df_sum.groupby(['ID']).median()
    y_true = df_median['label'].to_numpy()
    y_pred = df_median['y_pred'].to_numpy()
    
    ### plot roc curve
    auc3 = plot_roc(
        save_dir=save_dir,
        y_true=y_true,
        y_pred=y_pred,
        roc_fn=roc_fn,
        color=color
        )
    ### calculate roc, tpr, tnr with 1000 bootstrap
    stat3 = roc_bootstrap(
        bootstrap=bootstrap,
        y_true=y_true,
        y_pred=y_pred
        )

    print('roc patient median prob:')
    print(auc3)
    print(stat3)

    return auc3, stat3


    

   
