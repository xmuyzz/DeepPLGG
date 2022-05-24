import sys
import os
import glob
import SimpleITK as sitk
import pydicom
import numpy as np
import pandas as pd


def main(pro_data_dir, output_dir, temp_img, save_tfm=False):

    """
    Registers two CTs together
    Args:
        dataset (str): Name of dataset.
        patient_id (str): Unique patient id.
        data_type (str): Type of data (e.g., ct, pet, mri..)
        input_dir (str): Path to folder initial nrrd image files
        output_dir (str): Path to folder where the registered nrrds will be saved.
    Returns:
        The sitk image object.
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
    

if __name__ == '__main__':
    
    temp_img = 't2w_head.nii.gz'
    proj_dir = '/mnt/aertslab/USERS/Zezhong/pLGG'
    output_dir = os.path.join(proj_dir, 'BCH_T2W_reg')
    pro_data_dir = os.path.join(proj_dir, 'pro_data')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    main(pro_data_dir, output_dir, temp_img)




