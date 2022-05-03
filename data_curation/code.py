import numpy as np
import pandas as pd
import os



proj_dir = '/mnt/aertslab/USERS/Zezhong/pLGG/curation'
df1 = pd.read_csv(os.path.join(proj_dir, 'BRAF_slice.csv'))
df2 = pd.read_csv(os.path.join(proj_dir, 'BRAF_results.csv'))
list1 = df1['Subject_ID'].to_list()
list2 = df2['Subject_ID'].to_list()
print(set(list1))
print(set(list2))
print(len(set(list1)))
print(len(set(list2)))
a = list(set(list2) - set(list1))
print(a)

MRIs = []
for pat in list2:
    if pat in list1:
        MRI = 'Yes'
    else:
        MRI = 'No'
    MRIs.append(MRI)
print(MRIs)
df2['MRI'] = MRIs
print(df2)
df2.to_csv(os.path.join(proj_dir, 'BRAF_MRI.csv'), index=False)





