import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc, roc_curve



def main(curation_dir, pro_data_dir):

    pred = pd.read_csv(os.path.join(pro_data_dir, 'test_pred_2yr.csv'), index_col=0)
    print(pred)
    pred.columns = ['Subject_ID', 'Slice_ID', 'label', 'y_pred', 'y_pred_class']
    pred = pred.groupby(['Subject_ID']).mean().reset_index()
    df = pd.read_csv(os.path.join(curation_dir, 'master_csv.csv'))
    pred = pred.merge(df, how='left', on='Subject_ID')
    pred = pred[~pred['Extent of Tumor Resection'].isin(['Partial resection', 'Gross/Near total resection'])]
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    threshold = dict()
    fpr, tpr, threshold = roc_curve(pred['label'], pred['y_pred'])
    roc_auc = auc(fpr, tpr)
    roc_auc = np.around(roc_auc, 3)
    print('ROC AUC:', roc_auc)

if __name__ == '__main__':
    
    curation_dir = '/mnt/aertslab/USERS/Zezhong/pLGG/curation'
    pro_data_dir = '/mnt/aertslab/USERS/Zezhong/pLGG/pro_data'

    main(curation_dir, pro_data_dir)
