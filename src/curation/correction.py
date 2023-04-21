import os
import glob
import pandas as pd
import numpy as np
import SimpleITK as sitk



def correction(brain_dir, correction_dir):

    for img_dir in sorted(glob.glob(brain_dir + '/*.nii.gz')):
        ID = img_dir.split('/')[-1].split('.')[0]
        if ID[-1] == 'k':
            continue
        else:
            print(ID)
            img = sitk.ReadImage(img_dir, sitk.sitkFloat32)
            img = sitk.N4BiasFieldCorrection(img)
            ID = img_dir.split('/')[-1].split('.')[0]
            fn = ID + '_corrected.nii.gz'
            sitk.WriteImage(img, os.path.join(correction_dir, fn))
    print('bias field correction complete!')


if __name__ == '__main__':

    proj_dir = '/mnt/aertslab/USERS/Zezhong/pLGG'
    brain_dir = os.path.join(proj_dir, 'BCH_T2W_brain')
    correction_dir = os.path.join(proj_dir, 'BCH_T2W_correction')
    if not os.path.exists(correction_dir):
        os.makedirs(correction_dir)

    correction(brain_dir, correction_dir)
