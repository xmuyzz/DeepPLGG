import os
import numpy as np
import pandas as pd
import nibabel as nib
import glob


def main(data_dir, curation_dir):

    seg_dirs = []
    img_dirs = []
    for path in sorted(glob.glob(data_dir + '/*nii.gz')):
        if path.split('_')[-1] == 'T2W.nii.gz':
            img_dirs.append(path)
        else:
            seg_dirs.append(path)
    wmins = []
    wmaxs = []
    IDs = []
    for seg_dir in seg_dirs:
        seg_ID = seg_dir.split('/')[-1].split('.')[0]
        #print(seg_ID)
        #print(img_ID)
        print(seg_ID)
        seg = nib.load(seg_dir)
        seg = seg.get_fdata()
        w = np.any(seg, axis=(0, 1))
        wmin, wmax = np.where(w)[0][[0, -1]]
        print('wmin:', wmin)
        print('wmax:', wmax)
        wmins.append(wmin)
        wmaxs.append(wmax)
        IDs.append(seg_ID)
    print(len(IDs))
    img0_dirs = []
    img_IDs = []
    for img_dir in img_dirs:
        img_ID = img_dir.split('/')[-1].split('.')[0].split('_')[0]
        if img_ID in IDs:
            img_IDs.append(img_ID)
            img0_dirs.append(img_dir)
        else:
            print(img_ID)
    print(list(set(IDs) - set(img_IDs)))
    print(len(img0_dirs))
    df_slice = pd.DataFrame({'Subject_ID': IDs, 'wmin': wmins, 'wmax': wmaxs, 'img_dir': img0_dirs})
    print(df_slice)
    df_braf = pd.read_csv(os.path.join(curation_dir, 'BRAF_master.csv'))
    df_braf = df_braf[['Subject_ID', 'BRAF-Status', 'Overall_Survival', 'Progression_Free_Survival']]
    df = df_braf.merge(df_slice, on='Subject_ID', how='inner')
    df.to_csv(os.path.join(curation_dir, 'BRAF_slice.csv'), index=False)
    print(df)
    print('complete!!!')


if __name__ == '__main__':

    proj_dir = '/mnt/aertslab/USERS/Zezhong/pLGG'
    data_dir = os.path.join(proj_dir, 'T2W')
    curation_dir = os.path.join(proj_dir, 'curation')

    main(data_dir=data_dir, curation_dir=curation_dir)
 
