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
import SimpleITK as sitk



def BCH_T2W(proj_dir, save_t2w):
    """
    get BCH T2W images with largest slice numbers
    """
    curation_dir = proj_dir + '/BCH_raw/BCH_TOT'
    clinical_dir = proj_dir + '/clinical_data'
    BCH_T2W_dir = proj_dir + '/BCH_raw/BCH_T2W_one'
    BCH_T2W_all_dir = proj_dir + '/BCH_raw/BCH_T2W_all'
    if not os.path.exists(BCH_T2W_dir):
        os.makedirs(BCH_T2W_dir)
    if not os.path.exists(BCH_T2W_all_dir):
        os.makedirs(BCH_T2W_all_dir)
    l2_dirs = [x for x in iglob(curation_dir + '/*/*')]
    count = 0
    img_dirs = []
    pat_ids = []
    n_slices = []
    t2w_seqs = []
    no_t2w = []
    all_pat_names = []
    pat_names = []
    for path in l2_dirs:
        pat_name = path.split('/')[-2]
        pat_id = path.split('/')[-2].split('_')[-1]
        all_pat_names.append(pat_name)
        #pat_id = path.split('/')[-2]
        if path.split('.')[-1] == 'gz':
            t2w_seq = path.split('/')[-1].split('.')[0]
            #a = path.split('/')[-1].split('.')[0].split('_')
            name = path.split('/')[-1]
            #print('a:', a)
            if 't2' in name and 'cor' not in name and 'sag' not in name and 'flair' not in name:
            #if not list(set(a) & set(['cor', 'sag', 'flair'])) and \
            #    list(set(a) & set(['t2', 't2w', 't2-weighted', 't2-br'])):
                    count += 1
                    #fn_dir = BCH_T2W_dir + '/' + pat_id + '.nii.gz'
                    img = nib.load(path)
                    #meta = str(img.header.extensions)
                    #print(meta)
                    n_slice = img.shape[2]
                    print(count, pat_id)
                    img_dirs.append(path)
                    pat_ids.append(pat_id)
                    n_slices.append(n_slice)
                    t2w_seqs.append(t2w_seq)
                    pat_names.append(pat_name)
    no_t2w = set(all_pat_names) - set(pat_names)
    print('no T2W:', no_t2w)
    df = pd.DataFrame({'pat_id': pat_ids, 
                       'img_dir': img_dirs, 
                       'n_slice': n_slices, 
                       't2w_seq': t2w_seqs})
    pd.set_option('display.max_rows', None)
    #print(df)
    
    if save_t2w == 'one_t2w':
        # get the df with largest slice numbers
        #df = df.sort_values('n_slice', ascending=False).drop_duplicates(['pat_id'])
        df = df.loc[df.groupby(['pat_id'])['n_slice'].idxmax()].reset_index()
        print(df)
        # save nii data
        bad_data = []
        count = 0
        for pat_id, img_dir in zip(df['pat_id'], df['img_dir']):
            try:
                count += 1
                print(count, pat_id)
                fn_dir = BCH_T2W_dir + '/' + pat_id + '.nii.gz'
                img = sitk.ReadImage(img_dir)
                sitk.WriteImage(img, fn_dir)
            except Exception as e:
                print(e)
                bad_data.append(pat_id)
        print('bad data:', bad_data)
        print('save T2W complete!')
        ## get meta data spreadsheet
        img_dirs = [i for i in sorted(glob.glob(BCH_T2W_dir + '/*.nii.gz'))]
        #print(img_dirs)
        df['img_dir'] = img_dirs
        df.columns = ['index', 'bch_mrn', 'img_dir', 'n_slice', 't2w_seq']
        #meta = pd.read_csv(os.path.join(clinical_dir, 'BCH_clinical_meta.csv'), on_bad_lines='skip')
        #meta['bch_mrn'] = meta['bch_mrn'].astype('float64')
        #df['bch_mrn'] = df['bch_mrn'].astype('float64')
        #df = df.merge(meta, how='left', on='bch_mrn')
        #print(df)
        df.to_csv(clinical_dir + '/T2W_one.csv', index=False)
    
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
        bad_data = []
        df['img_id'] = pat_ids
        for img_id, img_dir in zip(df['img_id'], df['img_dir']):
            try:
                fn_dir = BCH_T2W_all_dir + '/' + img_id + '.nii.gz'
                img = sitk.ReadImage(img_dir)
                sitk.WriteImage(img, fn_dir)
            except Exception as e:
                print(e)
                bad_data.append(img_id)
        print(bad_data)


if __name__ == '__main__':
    
    proj_dir = '/mnt/kannlab_rfa/Zezhong/pLGG/data'

    BCH_T2W(proj_dir=proj_dir, save_t2w='one_t2w')






