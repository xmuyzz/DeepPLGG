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



def BCH_T2W(curation_dir, BCH_T2W_dir, BCH_T2W_all_dir, pro_data_dir, save_t2w):

    """
    get BCH T2W images with largest slice numbers
    """
    
    level2 = iglob(os.path.join(curation_dir, '*/*'))
    l2_dirs = [x for x in level2]
    count = 0
    img_dirs = []
    pat_ids = []
    n_slices = []
    t2w_seqs = []
    for path in l2_dirs:
        pat_id = path.split('/')[-2].split('_')[-1]
        #pat_id = path.split('/')[-2]
        if path.split('.')[-1] == 'gz':
            t2w_seq = path.split('/')[-1].split('.')[0]
            a = path.split('/')[-1].split('.')[0].split('_')
            #print('a:', a)
            if not list(set(a) & set(['cor', 'sag', 'flair'])):
                if list(set(a) & set(['t2', 't2w', 't2-weighted', 't2-br'])):
                    count += 1
                    #fn_dir = BCH_T2W_dir + '/' + pat_id + '.nii.gz'
                    img = nib.load(path)
                    #meta = str(img.header.extensions)
                    #print(meta)
                    n_slice = img.shape[2]
                    print(count)
                    print(pat_id)
                    img_dirs.append(path)
                    pat_ids.append(pat_id)
                    n_slices.append(n_slice)
                    t2w_seqs.append(t2w_seq)
    df = pd.DataFrame({
        'pat_id': pat_ids, 'img_dir': img_dirs, 'n_slice': n_slices, 't2w_seq': t2w_seqs})
    pd.set_option('display.max_rows', None)
    #print(df)
    
    if save_t2w == 'one_t2w':
        # get the df with largest slice numbers
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
        df.columns = ['index', 'bch_mrn', 'img_dir', 'n_slice', 't2w_seq']
        meta = pd.read_csv(os.path.join(pro_data_dir, 'BCH_clinical_meta.csv'), on_bad_lines='skip')
        meta['bch_mrn'] = meta['bch_mrn'].astype('float64')
        df['bch_mrn'] = df['bch_mrn'].astype('float64')
        df = df.merge(meta, how='left', on='bch_mrn')
        #print(df)
        df.to_csv(os.path.join(pro_data_dir, 'BCH_master.csv'), index=False)
    elif save_t2w == 'all_t2w':
        counts = {}
        pat_ids = df['pat_id'].to_list()
        for i, pat_id in enumerate(pat_ids):
            if pat_id in counts:
                counts[pat_id] += 1
                pat_ids[i] = f'{pat_id}_{counts[pat_id]}'
            else:
                counts[pat_id] = 1
        print(pat_ids)
        df['img_id'] = pat_ids
        for img_id, img_dir in zip(df['img_id'], df['img_dir']):
            fn_dir = BCH_T2W_all_dir + '/' + img_id + '.nii.gz'
            img = nib.load(img_dir)
            nib.save(img, fn_dir)


if __name__ == '__main__':

    curation_dir = '/mnt/aertslab/USERS/Zezhong/pLGG/BCH/BCH_curated'
    BCH_T2W_dir = '/mnt/aertslab/USERS/Zezhong/pLGG/BCH_T2W'
    BCH_T2W_all_dir = '/mnt/aertslab/USERS/Zezhong/pLGG/BCH_T2W_all'
    pro_data_dir = '/mnt/aertslab/USERS/Zezhong/pLGG/pro_data'

    BCH_T2W(
        curation_dir=curation_dir, 
        BCH_T2W_dir=BCH_T2W_dir, 
        BCH_T2W_all_dir=BCH_T2W_all_dir, 
        pro_data_dir=pro_data_dir, 
        save_t2w='all_t2w'
        )






