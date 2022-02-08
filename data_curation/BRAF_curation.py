import os
import numpy as np
import pandas as pd


def BRAF_curation(proj_dir):
    
    curation_dir = os.path.join(proj_dir, 'curation')
    if not os.path.exists(curation_dir): os.mkdir(curation_dir)

    df1 = pd.read_csv(os.path.join(curation_dir, 'master.csv'))
    df2 = pd.read_csv(os.path.join(curation_dir, 'cbtn-all-LGG-BRAF.csv'))
    df2 = df2[['Subject_ID', 'BRAF-Status', 'Overall Survival', 'Progression Free Survival']]
    df = pd.merge(df1, df2, on='Subject_ID', how='left')
    df.drop_duplicates('Subject_ID', keep='first', inplace=True)
    df = df[['Subject_ID', 'Genomic_Data', 'BRAF-Status', 'Overall Survival', 
             'Progression Free Survival', 'T1W', 'T1W_Pre', 'T1W_Post', 'T2W', 
             'FLAIR', 'ADC', 'FA', 'seq', 'path']]
    
    df.to_csv(os.path.join(curation_dir, 'BRAF_master.csv'), index=False)
    print('saved data to csv!')


