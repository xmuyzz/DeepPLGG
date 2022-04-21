#----------------------------------------------------------------------
# Deep learning for classification for contrast CT;
# Transfer learning using Google Inception V3;
#-----------------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import glob
from time import gmtime, strftime
from datetime import datetime
import timeit
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from utils.plot_cm import plot_cm

# ----------------------------------------------------------------------------------
# plot ROI
# ----------------------------------------------------------------------------------
def cm_patient_prob(run_type, threshold, save_dir):
    
    ### determine if this is train or test
    if run_type == 'val':
        df_sum = pd.read_pickle(os.path.join(save_dir, 'df_val_pred.p'))
    if run_type == 'test':
        df_sum = pd.read_pickle(os.path.join(save_dir, 'df_test_pred.p'))
    
    ### use patient-average scores and labels to calcualte ROC
    df_mean = df_sum.groupby(['ID']).mean()
    y_true = df_mean['label'].to_numpy()
    preds = df_mean['y_pred'].to_numpy()
    
    ### using threshold to determine predicted class for patient
    y_pred = []
    for pred in preds:
        if pred > threshold:
            pred = 1
        else:
            pred = 0
        y_pred.append(pred)
    y_pred = np.asarray(y_pred)

    ### using confusion matrix to calculate AUC
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float64') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.around(cm_norm, 2)
    
    ## plot cm
    for cm, cm_type in zip([cm, cm_norm], ['raw', 'norm']):
        plot_cm(
            cm=cm,
            cm_type='raw',
            level='patient',
            save_dir=save_dir
            )

    ## classification report
    report = classification_report(y_true, y_pred, digits=3)

    # statistics
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    tn = cm[0][0]
    acc = (tp + tn)/(tp + fp + fn + tn)
    tpr = tp/(tp + fn)
    tnr = tn/(tn + fp)
    tpr = np.around(tpr, 3)
    tnr = np.around(tnr, 3)
    auc5 = (tpr + tnr)/2
    auc5 = np.around(auc5, 3)
    
    print('cm patient prob:')
    print(cm)
    print(cm_norm)
    print(report)

    return cm, cm_norm, report





    

   
