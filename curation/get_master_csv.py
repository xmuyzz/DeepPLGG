import os
import numpy as np
import pandas as pd


def get_master_csv(curation_dir):

    df1 = pd.read_csv(os.path.join(curation_dir, 'BRAF_master.csv'))
    df2 = pd.read_csv(os.path.join(curation_dir, 'cbtn-all-LGG-BRAF.csv'))
    df2.drop_duplicates('Subject_ID', keep='first', inplace=True)
    df0 = df1.merge(df2, on='Subject_ID', how='left')
    print(df0)
    print(df0.shape[0])
    df0.to_csv(os.path.join(curation_dir, 'master_csv.csv'), index=False)

if __name__ == '__main__':

    curation_dir = '/mnt/aertslab/USERS/Zezhong/pLGG/curation'

    get_master_csv(curation_dir)
