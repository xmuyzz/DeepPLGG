#----------------------------------------------------------------------
# Deep learning for classification for contrast CT;
# Transfer learning using Google Inception V3;
#-----------------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
import pickle
from sklearn.metrics import auc, roc_auc_score
from sklearn.metrics import precision_recall_curve
from utils.plot_prc import plot_prc

# ----------------------------------------------------------------------------------
# precision recall curve
# ----------------------------------------------------------------------------------
def prc_img(run_type, output_dir, prc_fn, color, save_dir):

    ### determine if this is train or test
    if run_type == 'val' or run_type == 'train':
        df_sum = pd.read_pickle(os.path.join(save_dir, 'df_val_pred.p'))
    if run_type == 'test':
        df_sum = pd.read_pickle(os.path.join(save_dir, 'df_test_pred.p'))
    y_true = df_sum['label'].to_numpy()
    y_pred = df_sum['y_pred'].to_numpy()
    
    ### plot roc curve
    prc_auc = plot_prc(
        save_dir=save_dir,
        y_true=y_true,
        y_pred=y_pred,
        prc_fn=prc_fn,
        color=color
        )

    print('prc img:')
    print(prc_auc)
    
    return prc_auc



    

    
