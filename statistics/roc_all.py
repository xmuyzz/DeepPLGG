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

def roc_all(run_type, level, thr_prob, thr_pos, bootstrap, color, pro_data_dir, save_dir,
            fn_df_pred):

    df_sum = pd.read_csv(os.path.join(pro_data_dir, fn_df_pred))

    if level == 'img':
        y_true = df_sum['label'].to_numpy()
        y_pred = df_sum['y_pred'].to_numpy()
        print_info = 'roc image:'
    elif level == 'patient_mean_prob':
        df_mean = df_sum.groupby(['ID']).mean()
        y_true = df_mean['label'].to_numpy()
        y_pred = df_mean['y_pred'].to_numpy()
        print_info = 'roc patient prob:'
    elif level == 'patient_mean_pos':
        df_mean = df_sum.groupby(['ID']).mean()
        y_true = df_mean['label'].to_numpy()
        y_pred = df_mean['y_pred_class'].to_numpy()
        print_info = 'roc patient pos:'
   
    auc = plot_roc(
        save_dir=save_dir,
        y_true=y_true,
        y_pred=y_pred,
        level=level,
        color='blue'
        )
    ### calculate roc, tpr, tnr with 1000 bootstrap
    stat = roc_bootstrap(
        bootstrap=bootstrap,
        y_true=y_true,
        y_pred=y_pred
        )

    print(print_info)
    print(stat)

    return stat

