import os
import numpy as np
import pandas as pd
import shutil
import glob
import dicom2nifti
from glob import iglob
from pathlib import Path
import dicom2nifti.settings as settings
import nibabel as nib



def main(curation_dir):

    
    level2 = iglob(os.path.join(curation_dir, '*/*'))
    l2_dirs = [x for x in level2]
    count = 0
    for path in l2_dirs:
        pat_id = path.split('/')[-2].split('_')[-1].strip()
        if path.split('.')[-1] == 'gz':
            a = path.split('/')[-1].split('.')[0].split('_')
            #print('a:', a)
            if not list(set(a) & set(['cor', 'sag', 'flair'])):
                if list(set(a) & set(['t2', 't2w', 't2-weighted', 't2-br'])):
                    count += 1
                    fn_dir = BCH_T2W_dir + '/' + pat_id + '.nii.gz'
                    img = nib.load(path)
                    nib.save(img, fn_dir)
                    print(count)
    print('T2W curation complete')



if __name__ == '__main__':

    curation_dir = '/mnt/aertslab/USERS/Zezhong/pLGG/BCH/BCH_curated'
    BCH_T2W_dir = '/mnt/aertslab/USERS/Zezhong/pLGG/BCH_T2W'
    
    main(curation_dir)



