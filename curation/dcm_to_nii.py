import os
import pandas as pd
import numpy as np
import shutil
import glob
import dicom2nifti
from glob import iglob
from pathlib import Path
import dicom2nifti.settings as settings

def dcm_to_nii():
    settings.disable_validate_slice_increment()
    settings.enable_resampling()
    settings.set_resample_spline_interpolation_order(1)
    settings.set_resample_padding(-1000)
    # single slice
    settings.disable_validate_slicecount()

    proj_dir = '/mnt/kannlab_rfa/Zezhong/pLGG/data/BCH_raw/BCH_dcm_2023/TOT_dcm'
    IDs = []
    level2 = iglob(proj_dir + '/*/*')
    l2_dirs = [x for x in level2 if os.path.isdir(x)]
    for path in l2_dirs:
        print(path)
        ID = path.split('/')[-1]
        path = Path(path)
        print(ID)
        level_up = 1
        save_dir = path.parents[level_up - 1]
        print(save_dir)
        try:
            dicom2nifti.convert_directory(
                dicom_directory=path, 
                output_folder=save_dir, 
                compression=True, 
                reorient=True)
        except Exception as e:
            print('problematic data:', ID, e)
            IDs.append(ID)
    print('all problematic data:', IDs)

if __name__ == '__main__':
    dcm_to_nii()