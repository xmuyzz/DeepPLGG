import sys
import os
import glob
import SimpleITK as sitk
import pydicom
import numpy as np
import pandas as pd



proj_dir = '/mnt/aertslab/USERS/Zezhong/pLGG'
T2W_dir = os.path.join(proj_dir, 'BCH_T2W')
count = 0
IDs = []
for img_dir in sorted(glob.glob(T2W_dir + '/*.nii.gz')):
    count += 1
    ID = img_dir.split('/')[-1].split('.')[0]
    try:
        moving_img = sitk.ReadImage(img_dir, sitk.sitkFloat32)
        print(count)
        print(ID)
    except Exception as e:
        print(count, ID)
        print(e)
        IDs.append(ID)
print(IDs)
