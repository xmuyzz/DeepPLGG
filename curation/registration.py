import sys
import os
import glob
import SimpleITK as sitk
import pydicom
import numpy as np
import pandas as pd
from HD-BET.HD_BET import hd-bet



def registration(pro_data_dir, output_dir, temp_img, save_tfm=False):

    """
    MRI registration with SimpleITK
    Args:
        pro_data_dir {path} -- Name of dataset
        temp_img {str} -- registration image template
        output_dir {path} -- Path to folder where the registered nrrds will be saved.
    Returns:
        The sitk image object -- nii.gz
    Raises:
        Exception if an error occurs.
    """
    
    # Actually read the data based on the user's selection.
    fixed_image = sitk.ReadImage(os.path.join(pro_data_dir, temp_img), sitk.sitkFloat32)
    df = pd.read_csv(os.path.join(pro_data_dir, 'BCH_master.csv'))
    count = 0
    for img_dir, pat_id in zip(df['img_dir'], df['bch_mrn']):
        count += 1
        print(count)
        try:
            moving_image = sitk.ReadImage(img_dir, sitk.sitkFloat32)
            transform = sitk.CenteredTransformInitializer(
                fixed_image, 
                moving_image, 
                sitk.Euler3DTransform(), 
                sitk.CenteredTransformInitializerFilter.GEOMETRY)
            
            # multi-resolution rigid registration using Mutual Information
            registration_method = sitk.ImageRegistrationMethod()
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
            registration_method.SetMetricSamplingPercentage(0.01)
            registration_method.SetInterpolator(sitk.sitkLinear)
            registration_method.SetOptimizerAsGradientDescent(
                learningRate=1.0, 
                numberOfIterations=100, 
                convergenceMinimumValue=1e-6, 
                convergenceWindowSize=10)
            registration_method.SetOptimizerScalesFromPhysicalShift()
            registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4, 2, 1])
            registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
            registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            registration_method.SetInitialTransform(transform)
            final_transform = registration_method.Execute(
                fixed_image, 
                moving_image)                               
            moving_image_resampled = sitk.Resample(
                moving_image, 
                fixed_image, 
                final_transform, 
                sitk.sitkLinear, 
                0.0, 
                moving_image.GetPixelID())
            
            sitk.WriteImage(
                moving_image_resampled, os.path.join(output_dir, str(int(pat_id)) + '_registered.nii.gz'))
            if save_tfm:
                sitk.WriteTransform(final_transform, os.path.join(output_dir, str(int(pat_id)) + '_T2.tfm'))
        except:
            print('problematic data:', count, pat_id)


def correction(input_dir, output_dir):

    """
    Bias field correction with SimpleITK
    Args:
        input_dir {path} -- input directory
        output_dir {path} -- output directory
    Returns:
        Images in nii.gz format
    """

    for img_dir in sorted(glob.glob(brain_dir + '/*.nii.gz')):
        ID = img_dir.split('/')[-1].split('.')[0]
        if ID[-1] == 'k':
            continue
        else:
            print(ID)
            img = sitk.ReadImage(img_dir, sitk.sitkFloat32)
            img = sitk.N4BiasFieldCorrection(img)
            ID = img_dir.split('/')[-1].split('.')[0]
            fn = ID + '_corrected.nii.gz'
            sitk.WriteImage(img, os.path.join(correction_dir, fn))
    print('bias field correction complete!')


def brain_extraction(i, o, device, mode, tta):
    
    """
    brain extraction - skull stripping with HD-BET
    works for T2W, pre-T1W, post-T2W, FLAIR
    Args:
        input {path} -- input files or folders
        output {path} -- output files or folders
        mode {str} -- (fast | accurate)
        tta {int} -- (0 | 1)
    Returns:
        img in nii.gz format
    """

    hd-bet(i=input_dir, o=output_dir, device='cpu', mode='fast', tta=0)



if __name__ == '__main__':
    
    temp_img = 't2w_temp.nii.gz'
    proj_dir = '/mnt/aertslab/USERS/Zezhong/pLGG'
    output_dir = os.path.join(proj_dir, 'BCH_T2W_reg')
    pro_data_dir = os.path.join(proj_dir, 'pro_data')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    registration(pro_data_dir, output_dir, temp_img)




