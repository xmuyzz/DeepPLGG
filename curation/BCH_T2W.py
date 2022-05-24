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



def main(curation_dir, BCH_T2W_dir, pro_data_dir):
    
    level2 = iglob(os.path.join(curation_dir, '*/*'))
    l2_dirs = [x for x in level2]
    count = 0
    img_dirs = []
    pat_ids = []
    n_slices = []
    for path in l2_dirs:
        pat_id = path.split('/')[-2].split('_')[-1]
        #pat_id = path.split('/')[-2]
        if path.split('.')[-1] == 'gz':
            a = path.split('/')[-1].split('.')[0].split('_')
            #print('a:', a)
            if not list(set(a) & set(['cor', 'sag', 'flair'])):
                if list(set(a) & set(['t2', 't2w', 't2-weighted', 't2-br'])):
                    count += 1
                    #fn_dir = BCH_T2W_dir + '/' + pat_id + '.nii.gz'
                    img = nib.load(path)
                    n_slice = img.shape[2]
                    print(count)
                    print(pat_id)
                    img_dirs.append(path)
                    pat_ids.append(pat_id)
                    n_slices.append(n_slice)
    df = pd.DataFrame({'pat_id': pat_ids, 'img_dir': img_dirs, 'n_slice': n_slices})
    pd.set_option('display.max_rows', None)
    #print(df)
    #df = df.sort_values('n_slice', ascending=False).drop_duplicates(['pat_id'])
    df = df.loc[df.groupby(['pat_id'])['n_slice'].idxmax()].reset_index()
    #print(df)
    
    # save nii data
    for pat_id, img_dir in zip(df['pat_id'], df['img_dir']):
        fn_dir = BCH_T2W_dir + '/' + pat_id + '.nii.gz'
        img = nib.load(img_dir)
        nib.save(img, fn_dir)
    print('save T2W complete!')

    img_dirs = [i for i in sorted(glob.glob(BCH_T2W_dir + '/*.nii.gz'))]
    #print(img_dirs)
    df['img_dir'] = img_dirs
    df.columns = ['index', 'bch_mrn', 'img_dir', 'n_slice']
    meta = pd.read_csv(os.path.join(pro_data_dir, 'BCH_clinical_meta.csv'), on_bad_lines='skip')
    meta['bch_mrn'] = meta['bch_mrn'].astype('float64')
    df['bch_mrn'] = df['bch_mrn'].astype('float64')
    df = df.merge(meta, how='left', on='bch_mrn')
    #print(df)
    df.to_csv(os.path.join(pro_data_dir, 'BCH_master.csv'), index=False)



if __name__ == '__main__':

    curation_dir = '/mnt/aertslab/USERS/Zezhong/pLGG/BCH/BCH_curated'
    BCH_T2W_dir = '/mnt/aertslab/USERS/Zezhong/pLGG/BCH_T2W'
    pro_data_dir = '/mnt/aertslab/USERS/Zezhong/pLGG/pro_data'

    main(curation_dir, BCH_T2W_dir, pro_data_dir)



