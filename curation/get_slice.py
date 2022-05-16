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
            imgs.append(path)
        else:
            segs.append(path)
    wmins = []
    wmaxs = []
    IDs = []
    paths = []
    for seg in segs:
        ID = seg.split('/')[-1].split('.')[0]
        print(ID)
        img = nib.load(seg)
        img = img.get_fdata()
        w = np.any(img, axis=(0, 1))
        wmin, wmax = np.where(w)[0][[0, -1]]
        print('wmin:', wmin)
        print('wmax:', wmax)
        wmins.append(wmin)
        wmaxs.append(wmax)
        IDs.append(ID)
        df = pd.DataFrame({'ID': IDs, 'wmin': wmins, 'wmax': wmaxs})
        df.to_csv(os.path.join(curation_dir, 'slice.csv'), index=False) 

if __name__ == '__main__':
    
    proj_dir = '/mnt/aertslab/USERS/Zezhong/pLGG'
    data_dir = os.path.join(proj_dir, 'T2W')
    curation_dir = os.path.join(proj_dir, 'curation')
    
    get_slice(
        data_dir=data_dir,
        curation_dir=curation_dir
        )
