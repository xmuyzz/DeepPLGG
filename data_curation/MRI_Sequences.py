import os
import numpy as np
import pandas as pd


def MRI_Sequences(proj_dir):

    curation_dir = os.path.join(proj_dir, 'curation')
    if not os.path.exists(curation_dir): os.mkdir(curation_dir)

    df0 = pd.read_csv(os.path.join(curation_dir, 'curation_data.csv'))
    df1 = pd.read_csv(os.path.join(curation_dir, 'pLGG_sum.csv'))
   
    df0['seq'] = df0['seq_id'] + df0['contrast']
    df = df0[['pat_id', 'scan_type', 'seq', 'path']]
    print(df)
    df2 = df.groupby(['pat_id'], as_index=True)['seq'].apply(list).reset_index()
    df3 = df.groupby(['pat_id'], as_index=True)['path'].apply(list).reset_index()
    df2['path'] = df3['path'].to_list()
    print(df2)
    print(df2.shape)
    df2.rename(columns={'pat_id': 'Subject_ID'}, inplace=True)
    df = pd.merge(df2, df1, on='Subject_ID')
    print(df)
    
    T1Wpres = []
    T1Wposts = []
    T1Ws = []
    T2Ws = []
    FLAIRs = []
    ADCs = []
    FAs = []
    for seq in df['seq']:
        if 'T1Wpre' in seq:
            a = 1
        else:
            a = 0
        if 'T1Wpost' in seq:
            b = 1
        else:
            b = 0
        if 'T1W ' in seq:
            c = 1
        else:
            c = 0
        if 'T2W ' in seq:
            d = 1
        else:
            d = 0
        if 'FLAIR ' in seq:
            e = 1
        else:
            e = 0
        if 'ADC ' in seq:
            f = 1
        else:
            f = 0
        if 'FA ' in seq:
            f = 1
        else:
            f = 0
        T1Wpres.append(a)
        T1Wposts.append(b)
        T1Ws.append(c)
        T2Ws.append(d)
        FLAIRs.append(e)
        ADCs.append(f)
        FAs.append(f)

    df['T1W_Pre'], df['T1W_Post'], df['T1W'], df['T2W'], df['FLAIR'], \
    df['ADC'], df['FA'] = [T1Wpres, T1Wposts, T1Ws, T2Ws, FLAIRs, ADCs, FAs]
    
    df = df[['Subject_ID', 'Clinical_Data', 'Genomic_Data', 'MRI_Data',  
             'T1W_Pre', 'T1W_Post', 'T1W', 'T2W', 'FLAIR', 'ADC', 'FA', 'seq', 'path']]
    #print(df)
    df.to_csv(os.path.join(curation_dir, 'master.csv'), index=False)


