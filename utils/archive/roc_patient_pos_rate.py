#----------------------------------------------------------------------
# Deep learning for classification for contrast CT;
# Transfer learning using Google Inception V3;
#-------------------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import pickle
from utils.plot_roc import plot_roc
from utils.roc_bootstrap import roc_bootstrap

# ----------------------------------------------------------------------------------
# plot ROI
# ----------------------------------------------------------------------------------
def roc_patient_pos_rate(run_type, output_dir, roc_fn, color, bootstrap, save_dir):
    
    ### determine if this is train or test
    if run_type == 'train' or run_type == 'val':
        df_sum = pd.read_pickle(os.path.join(save_dir, 'df_val_pred.p'))
    if run_type == 'test':
        df_sum = pd.read_pickle(os.path.join(save_dir, 'df_test_pred.p'))
    
    ### use patient-average scores and labels to calcualte ROC
    df_mean = df_sum.groupby(['ID']).mean()
    y_true = df_mean['label'].to_numpy()
    ### pos_rate = n_predicted_class1 / n_img
    y_pred = df_mean['y_pred_class'].to_numpy()
    
    ### plot roc curve
    auc4 = plot_roc(
        save_dir=save_dir,
        y_true=y_true,
        y_pred=y_pred,
        roc_fn=roc_fn,
        color=color
        )

    ### calculate roc, tpr, tnr with 1000 bootstrap
    stat4 = roc_bootstrap(
        bootstrap=bootstrap,
        y_true=y_true,
        y_pred=y_pred
        )

    print('roc patient pos rate:')
    print(auc4)
    print(stat4)

    return auc4, stat4







    

   
