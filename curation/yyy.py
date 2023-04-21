import os
import numpy as np
import pandas as pd
import glob


csv_dir = '/mnt/kannlab_rfa/Zezhong/pLGG/data/clinical_data'
df0 = pd.read_csv(csv_dir + '/BRAF_more2.csv')
IDs = df0['BCH MRN'].to_list()

proj_dir = '/mnt/kannlab_rfa/Zezhong/pLGG/data/BCH_raw'
ids = []
paths = []
for file in os.listdir(proj_dir + '/BCH_dcm_2023/T2W_one'):
    #print(id)
    if not file.startswith('.'):
        id = file.split('.')[0]
        #print(id)
        id = float(id)
        ids.append(id)  
        path = proj_dir + '/' + file
        paths.append(path)
df = pd.DataFrame({'BCH MRN': ids, 'T2W Path': paths})
df = df.merge(df0, how='left', on='BCH MRN')
df.to_csv(csv_dir + '/BCH_T2W_BRAF_2023.csv', index=False)

#print('ID:', IDs)
#print('id:', ids)

overlap = list(set(IDs) & set(ids))
print(overlap)
print(len(overlap))
print('done!')