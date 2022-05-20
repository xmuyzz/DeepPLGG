import os
import pandas as pd
import numpy as np
#import zipfile
import shutil
import glob
import dicom2nifti
from glob import iglob
from pathlib import Path
import dicom2nifti.settings as settings



def main(data_dir, curation_dir, rename, unzip, rename_folder, get_nii):
    
    if rename:
        for fn in glob.glob(data_dir + '/*.zip'):
            os.rename(fn, fn.replace(' ', '_'))
            os.rename(fn, fn.replace('(1)', '1'))
            print(fn)
    
    if unzip:
        for count, fn in enumerate(glob.glob(data_dir + '/*.zip')):
            print(count)
            print(fn)
            shutil.unpack_archive(
                filename=fn, extract_dir=curation_dir, format='zip')

    if rename_folder:
        for fn in os.listdir(curation_dir):
            os.rename(
                os.path.join(curation_dir, fn),
                os.path.join(curation_dir, fn.replace(' ', '_')))
            print(fn)
        level2 = iglob(os.path.join(curation_dir, '*/*/*'))
        l2_dirs = [x for x in level2 if os.path.isdir(x)]
        for path in l2_dirs:
            os.rename(path, path.replace(' ', '_'))
            print(path)
    
    if get_nii:
        # Inconsistent slice incremement
        settings.disable_validate_slice_increment()
        settings.enable_resampling()
        settings.set_resample_spline_interpolation_order(1)
        settings.set_resample_padding(-1000)
        # single slice
        settings.disable_validate_slicecount()
        
        level2 = iglob(os.path.join(curation_dir, '*/*'))
        l2_dirs = [x for x in level2 if os.path.isdir(x)]
        for path in l2_dirs:
            print(path)
            path = Path(path)
            level_up = 1
            save_dir = path.parents[level_up - 1]
            print(save_dir)
            dicom2nifti.convert_directory(
                dicom_directory=path, 
                output_folder=save_dir, 
                compression=True, 
                reorient=True)



if __name__ == '__main__':

    data_dir = '/mnt/aertslab/USERS/Zezhong/pLGG/BCH/data/zip_data'
    curation_dir = '/mnt/aertslab/USERS/Zezhong/pLGG/BCH/BCH_curated/Girard_Michael_A_4228140'

    rename = False
    unzip = False
    rename_folder = False
    get_nii = True

    main(data_dir, curation_dir, rename, unzip, rename_folder, get_nii)




