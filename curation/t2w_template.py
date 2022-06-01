import sys
import os
import glob
import SimpleITK as sitk
import pydicom
import numpy as np
import pandas as pd
import nibabel as nib
from HDBET.HD_BET.hd_bet import hd_bet



def t2w_template(pro_data_dir, interp_type='linear', brain_extraction=False):

    if brain_extraction:
        # brain extration
        input_img = os.path.join(pro_data_dir, 't2w_template.nii.gz')
        output_img = os.path.join(pro_data_dir, 't2w_brain.nii.gz')
        hd_bet(inputs=input_img, outputs=output_img, device='cpu', mode='fast', tta=0)

    # respace to (1, 1, 3)
    img = sitk.ReadImage(
        os.path.join(pro_data_dir, 't2w_brain.nii.gz'), sitk.sitkFloat32)
    mask = sitk.ReadImage(
        os.path.join(pro_data_dir, 't2w_brain_mask.nii.gz'), sitk.sitkFloat32)    
    # bounding box to delete white space
    arr = sitk.GetArrayFromImage(img)
    #arr_mask = sitk.GetArrayFromImage(mask)
    #arr = arr * arr_mask
    arr = np.transpose(arr, (2, 1, 0))
    xs, ys, zs = np.where(arr!=0)
    arr = arr[min(xs):max(xs)+1, min(ys):max(ys)+1, min(zs):max(zs)+1]
    nii = nib.Nifti1Image(arr, affine=np.eye(4))
    nib.save(nii, os.path.join(pro_data_dir, 'brain_test.nii.gz'))

    # respace fixed img on z-direction
    #new_size = img.GetSize()
    #print(new_size)
    old_size = arr.shape
    #print(new_size)
    old_spacing = img.GetSpacing()
    new_spacing = (1, 1, 1)
    #new_size = [old_size[0], old_size[1], int(round((old_size[2] * 1) / float(z_spacing)))]
    new_size = [
        int(round((old_size[0] * old_spacing[0]) / float(new_spacing[0]))),
        int(round((old_size[1] * old_spacing[1]) / float(new_spacing[1]))),
        int(round((old_size[2] * old_spacing[2]) / float(new_spacing[2])))
        ]
    print(new_size)
    old_origin = img.GetOrigin()
    new_origin = [old_origin[0] + min(xs), old_origin[1] - min(ys), old_origin[2] + min(zs)]
    if interp_type == 'linear':
        interp_type = sitk.sitkLinear
    elif interp_type == 'bspline':
        interp_type = sitk.sitkBSpline
    elif interp_type == 'nearest_neighbor':
        interp_type = sitk.sitkNearestNeighbor
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    #resample.SetOutputOrigin(img.GetOrigin())
    resample.SetOutputOrigin(new_origin)
    resample.SetOutputDirection(img.GetDirection())
    resample.SetInterpolator(interp_type)
    resample.SetDefaultPixelValue(img.GetPixelIDValue())
    resample.SetOutputPixelType(sitk.sitkFloat32)
    img = resample.Execute(img)
    sitk.WriteImage(img, os.path.join(pro_data_dir, 'T2W_brain_template.nii.gz'))
    print('complete!')


    

if __name__ == '__main__':

    proj_dir = '/mnt/aertslab/USERS/Zezhong/pLGG'
    reg_dir = os.path.join(proj_dir, 'BCH_T2W_reg')
    brain_dir = os.path.join(proj_dir, 'BCH_T2W_brain')
    correction_dir = os.path.join(proj_dir, 'BCH_T2W_correction')
    pro_data_dir = os.path.join(proj_dir, 'pro_data')

    t2w_template(pro_data_dir)







