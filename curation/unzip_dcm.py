import os
import pandas as pd
import numpy as np
#import zipfile
import shutil
import glob


proj_dir = '/mnt/kannlab_rfa/Zezhong/pLGG/data/BCH_raw/BCH_dcm_2023'
data_dir = proj_dir + '/TOT_zip'
save_dir = proj_dir + '/TOT_dcm'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
bad_data = []
for count, fn in enumerate(glob.glob(data_dir + '/*.zip')):
    ID = fn.split('/')[-1].split('.')[0]
    save_path = save_dir + '/' + ID 
    if os.path.exists(save_path):
        print('data already exists!!!')
    else:
        print(count, ID)
        try:
            shutil.unpack_archive(
                filename=fn, 
                extract_dir=save_dir, 
                format='zip')
        except Exception as e:
            print('problematic data:', ID, e)
            bad_data.append(ID)
print(bad_data)