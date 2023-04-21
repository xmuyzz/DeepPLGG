import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def main(pro_data_dir):

    df0 = pd.read_csv(os.path.join(pro_data_dir, 'pat_pred_BRAF.csv'))
    df1 = pd.read_csv(os.path.join(pro_data_dir, 'pat_pred_fusion.csv'))
    preds = []
    for pred0, pred1 in zip(df0['y_pred_class'], df1['y_pred_class']):
        if pred0 == 0 and pred1 == 0:
            pred = 1
        else:
            pred = 0
        preds.append(pred)
    labels = []
    for label0, label1 in zip(df0['label'], df1['label']):
        if label0 == 0 and label1 == 0:
            label = 1
        else:
            label = 0
        labels.append(label)

    # classification report
    report = classification_report(labels, preds, digits=3)

    # using confusion matrix to calculate AUC
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype('float64') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.around(cm_norm, 2)

    print(report)
    print(cm)
    print(cm_norm)


if __name__ == '__main__':

    pro_data_dir = '/mnt/aertslab/USERS/Zezhong/pLGG/pro_data'
    
    main(pro_data_dir)
