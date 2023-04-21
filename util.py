import numpy as np
import os
import glob
import pandas as pd

IDs = []
csv_dir = '/mnt/kannlab_rfa/Zezhong/pLGG/BRAF/csv_file'
df = pd.read_csv(csv_dir + '/CBTN.csv')
for ID in df['pat_id']:
    ID = ID.split('_')[0]
    print(ID)
    IDs.append(ID)
df['pat_id'] = IDs
df.to_csv(csv_dir + '/CBTN.csv')

