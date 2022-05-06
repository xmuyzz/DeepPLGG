import os
import numpy as np
import pandas as pd
from opts import parse_opts



def survival_label(curation_dir):

    df = pd.read_csv(os.path.join(curation_dir, 'cbtn-all-LGG-BRAF.csv'))
    df = df[~df['Age at Diagnosis'].isin(['Not Reported'])]
    df = df[~df['Progression Free Survival'].isin(['Not Reported'])]
    df['fu'] = df['Age at Last Known Clinical Status'].astype(float) - df['Age at Diagnosis'].astype(float)
    #print(df['fu'])
    df = df[['Subject_ID', 'fu', 'Overall Survival', 'Progression Free Survival', 
            'Last Known Clinical Status']]
    #print(df)
    df.drop_duplicates('Subject_ID', keep='first', inplace=True)
    print(df)
    df.columns = ['Subject_ID', 'FU', 'OS', 'PFS', 'Death']
    df['PFS'] = df['PFS'].astype(float)
    # 3 yr survival
    events_3yr = []
    for pfs in df['PFS']:
        if pfs < 1095:
            event = 1
        else:
            event = 0
        events_3yr.append(event)
    df['3yr_event'] = events_3yr
    
    # 2 yr survival
    events_2yr = []
    for pfs in df['PFS']:
        if pfs < 730:
            event = 1
        else:
            event = 0
        events_2yr.append(event)
    df['2yr_event'] = events_2yr
    print(df) 
    df2 = pd.read_csv(os.path.join(curation_dir, 'BRAF_slice.csv'))
    df0 = df2.merge(df, on='Subject_ID', how='left')
    print(df0)
    print(df0.shape[0])
    df0.to_csv(os.path.join(curation_dir, 'BRAF_survival_slice.csv'), index=False)
   


if __name__ == '__main__':
 
    opt = parse_opts()
    if opt.root_dir is not None:
        opt.curation_dir = os.path.join(opt.root_dir, opt.curation)
        opt.pro_data_dir = os.path.join(opt.root_dir, opt.pro_data)
        if not os.path.exists(opt.pro_data_dir):
            os.makedirs(opt.pro_data_dir)
        if not os.path.exists(opt.curation_dir):
            os.makedirs(opt.curation_dir)
    else:
        print('provide root dir to start!')

    survival_label(curation_dir=opt.curation_dir)    






