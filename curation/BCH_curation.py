import os
import pandas as pd
import numpy as np
#import zipfile
import shutil
import glob
import dicom2nifti
from glob import iglob



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
        for fn in l2_dirs:
            os.rename(fn, fn.replace(' ', '_'))
            #print(fn)

    if get_nii:
        for subdir, dirs, files in os.walk(curation_dir):
            #print('subdir:', subdir)
            #print('dirs:', dirs)
            print(files)            
#            dicom2nifti.convert_directory(
#                dicom_directory, 
#                output_folder, 
#                compression=True, 
#                reorient=True)



if __name__ == '__main__':

    data_dir = '/mnt/aertslab/USERS/Zezhong/pLGG/BCH/data/zip_data'
    curation_dir = '/mnt/aertslab/USERS/Zezhong/pLGG/BCH/BCH_curated'

    rename = False
    unzip = False
    rename_folder = False
    get_nii = True

    main(data_dir, curation_dir, rename, unzip, rename_folder, get_nii)




