import sys
import os
import glob
import SimpleITK as sitk
import pydicom
import numpy as np
import pandas as pd
from HDBET.HD_BET.hd_bet import hd_bet



def brain_extraction(input_dir, output_dir):

    """
    Brain extraction using HDBET package (UNet based DL method)
    Args:
        T2W_dir {path} -- input dir;
        brain_dir {path} -- output dir;
    Returns:
        Brain images
    """

    hd_bet(inputs=input_dir, outputs=output_dir, device='cpu', mode='fast', tta=0)
    print('brain extraction complete!')


if __name__ == '__main__':

    temp_img = 'temp_head.nii.gz'
    proj_dir = '/mnt/aertslab/USERS/Zezhong/pLGG'
    reg_dir = os.path.join(proj_dir, 'BCH_T2W_reg2')
    brain_dir = os.path.join(proj_dir, 'BCH_T2W_brain2')

    brain_extraction(input_dir=reg_dir, output_dir=brain_dir)
