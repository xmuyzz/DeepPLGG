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


def rename_dir(data_dir):
    print('start...')
    count = 0
    if rename:
        print(data_dir)
        for fn in glob.glob(data_dir + '/*.zip'):
            count += 1
            os.rename(fn, fn.replace(' ', '_'))
            #os.rename(fn, fn.replace('(1)', '1'))
            print(count, fn)


def unzip():
    """unzip data
    """
    proj_dir = '/mnt/kannlab_rfa/Zezhong/pLGG/data/BCH_raw'
    data_dir = proj_dir + '/2022_1212_zip'
    curation_dir = proj_dir + '/BCH_curation202212'
    if not os.path.exists(curation_dir):
        os.makedirs(curation_dir)
    bad_data = []
    for count, fn in enumerate(glob.glob(data_dir + '/*.zip')):
        ID = fn.split('/')[-1].split('.')[0]
        print(count, ID)
        try:
            shutil.unpack_archive(
                filename=fn, 
                extract_dir=curation_dir, 
                format='zip')
        except Exception as e:
            print('problematic data:', ID, e)
            bad_data.append(ID)
    print(bad_data)


def rename_folder(curation_dir):
    """ repace space with hyphen in patient folder sequence subfolders
    """
    for fn in os.listdir(curation_dir):
        path = curation_dir + '/' + fn
        os.rename(path, path.replace(' ', '_'))
        print(path)
    level2 = iglob(os.path.join(curation_dir, '*/*/*'))
    l2_dirs = [x for x in level2 if os.path.isdir(x)]
    for path in l2_dirs:
        os.rename(path, path.replace(' ', '_'))
        print(path)
    

def dcm_to_nii(curation_dir):
    """ convert dcm file to nii and save nii files in main folder;
    """
    # Inconsistent slice incremement
    settings.disable_validate_slice_increment()
    settings.enable_resampling()
    settings.set_resample_spline_interpolation_order(1)
    settings.set_resample_padding(-1000)
    # single slice
    settings.disable_validate_slicecount()
    # convert 
    IDs = []
    level2 = iglob(curation_dir + '/*/*')
    l2_dirs = [x for x in level2 if os.path.isdir(x)]
    proj_dir = '/mnt/kannlab_rfa/Zezhong/pLGG/data/BCH_raw/BCH_TOT/Girard_Michael_A_4228140'
    l2_dirs = [proj_dir]
    for path in l2_dirs:
        print(path)
        ID = path.split('/')[-1]
        path = Path(path)
        print(ID)
        level_up = 1
        save_dir = path.parents[level_up - 1]
        #print(save_dir)
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

    #data_dir = '/mnt/aertslab/USERS/Zezhong/pLGG/CBTN_BCH_Data/BCH/data/zip_data3'
    #curation_dir = '/mnt/aertslab/USERS/Zezhong/pLGG/CBTN_BCH_Data/BCH/BCH_curated2'
    #curation_dir = '/mnt/kannlab_rfa/Zezhong/pLGG/data/BCH_raw/BCH_curated2'
    proj_dir = '/mnt/kannlab_rfa/Zezhong/pLGG/data/BCH_raw'
    data_dir = proj_dir + '/2022_1212_zip'
    curation_dir = proj_dir + '/BCH_curation202212' 
     
    step = 'dcm_to_nii'
    
    if step == 'unzip':
        unzip()
    elif step == 'rename_folder':
        rename_folder(curation_dir)
    elif step == 'dcm_to_nii':
        dcm_to_nii(curation_dir)





