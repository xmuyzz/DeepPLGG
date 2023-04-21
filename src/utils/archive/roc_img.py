#--------------------------------------------------------------------------------------
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
# ROC and AUC on image level
# ----------------------------------------------------------------------------------
def roc_img(run_type, output_dir, roc_fn, color, bootstrap, save_dir):
    
    ### determine if this is train or test
    if run_type == 'train' or run_type == 'val':
        df_sum = pd.read_pickle(os.path.join(save_dir, 'df_val_pred.p'))
    if run_type == 'test':
        df_sum = pd.read_pickle(os.path.join(save_dir, 'df_test_pred.p'))
    y_true = df_sum['label'].to_numpy()
    y_pred = df_sum['y_pred'].to_numpy()
    
    ### plot roc curve
    auc1 = plot_roc(
        save_dir=save_dir,
        y_true=y_true,
        y_pred=y_pred,
        roc_fn=roc_fn,
        color=color
        )

    ### calculate roc, tpr, tnr with 1000 bootstrap
    stat1 = roc_bootstrap(
        bootstrap=bootstrap,
        y_true=y_true,
        y_pred=y_pred
        )
    print('roc img:')
    print(auc1)
    print(stat1)

    return auc1, stat1


    

    
