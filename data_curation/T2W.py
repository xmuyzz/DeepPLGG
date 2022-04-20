import os
import pandas as pd
import numpy as np
import nibabel as nib



def main(proj_dir):

    curation_dir = os.path.join(proj_dir, 'curation')
    data_dir = os.path.join(proj_dir, 'MRI_curation')
    T2W_dir = os.path.join(proj_dir, 'T2W')
    T2W0_dir = os.path.join(proj_dir, 'T2W0')
    if not os.path.exists(T2W0_dir):
        os.makedirs(T2W0_dir)

    df = pd.read_pickle(os.path.join(curation_dir, 'master.pkl'))
    no_T2Ws = []
    for seqs, img_dirs, pat_id in zip(df['seq'], df['path'], df['Subject_ID']):
        count = 0
        for seq, img_dir in zip(seqs, img_dirs):
            if 'T2W' not in seqs:
                no_T2Ws.append(pat_id)
            else:
                if seq == 'T2W':
                    count += 1
                    print(pat_id)
                    print(count)
                    fn = pat_id + '_' + str(count) + '_T2W.nii.gz'
                    img = nib.load(img_dir) 
                    nib.save(img, os.path.join(T2W0_dir, fn))
    print('patients without T2W:', no_T2Ws)


if __name__ == '__main__':

    proj_dir = '/mnt/aertslab/USERS/Zezhong/pLGG'

    main(proj_dir)
