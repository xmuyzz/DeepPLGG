import os
import numpy as np
import pandas as pd
import nibabel as nib
import glob


def get_slice(data_dir, curation_dir):

    imgs = []
    segs = []
    for path in sorted(glob.glob(data_dir + '/*nii.gz')):
        if path.split('_')[-1] == 'T2W.nii.gz':
            img_dirs.append(path)
        else:
            seg_dirs.append(path)
    wmins = []
    wmaxs = []
    IDs = []
    img_dirs = []
    for seg, img in zip(segs, imgs):
        img_ID = seg.split('/')[-1].split('.')[0]
        seg_ID = seg.split('/')[-1].split('.')[0]
        if seg_ID = img_ID:
            print(seg_ID)
            seg = nib.load(seg)
            seg = img.get_fdata()
            w = np.any(seg, axis=(0, 1))
            wmin, wmax = np.where(w)[0][[0, -1]]
            print('wmin:', wmin)
            print('wmax:', wmax)
            wmins.append(wmin)
            wmaxs.append(wmax)
            IDs.append(seg_ID)
            img_dirs.append(img_dir)
    df_slice = pd.DataFrame({'Subject_ID': IDs, 'wmin': wmins, 'wmax': wmaxs. 'img_dir': img_dirs})

    df_braf = pd.read_csv(os.path.join(curation_dir, 'BRAF_master.csv'))
    df_braf = df_brad[['Subject_ID'], ['BRAF-Status'], ['overall_survival'], ['Progression Free']]
    df_braf.join(df_slice, how='outer')
    df_braf.to_csv(os.path.join(curation_dir, 'BRAF_slice.csv'), index=False)

if __name__ == '__main__':

    proj_dir = '/mnt/aertslab/USERS/Zezhong/pLGG'
    data_dir = os.path.join(proj_dir, 'T2W')
    curation_dir = os.path.join(proj_dir, 'curation')

    get_slice(
        data_dir=data_dir,
        curation_dir=curation_dir
        )
~            
