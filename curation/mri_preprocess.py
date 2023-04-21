import sys
import os
import glob
import SimpleITK as sitk
import pydicom
import numpy as np
import pandas as pd
import itk
import nibabel as nib
from HDBET.HD_BET.hd_bet import hd_bet
from tqdm import tqdm


def bf_correction(input_dir, output_dir):

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



def registration(cohort, pro_data_dir, input_dir, output_dir, temp_img, run_list, 
                 interp_type='linear', save_tfm=False):

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
    #fixed_image = itk.imread(os.path.join(temp_dir, temp_img), itk.F)
    fixed_img = sitk.ReadImage(os.path.join(temp_dir, temp_img), sitk.sitkFloat32)
    IDs = []
    for img_dir in sorted(glob.glob(T2W_dir + '/*.nii.gz')):
        ID = img_dir.split('/')[-1].split('.')[0]
        try:
            moving_img = sitk.ReadImage(img_dir, sitk.sitkFloat32)
        except Exception as e:
            IDs.append(ID)
    print('problematic data:', IDs)
    count = 0
    for img_dir in tqdm(sorted(glob.glob(input_dir + '/*.nii.gz'))):
        ID = img_dir.split('/')[-1].split('.')[0]
        if ID in IDs:
            print('problematic data!')
        else:
            if 'T2W' in ID:
                count += 1
                print(count)
                #print(ID)
                #try:
                moving_img = sitk.ReadImage(img_dir, sitk.sitkFloat32)
                # bias filed correction
                moving_img = sitk.N4BiasFieldCorrection(moving_img)
                if cohort == 'BCH':
                    pat_id = img_dir.split('/')[-1].split('.')[0]
                elif cohort == 'CBTN':
                    pat_id = img_dir.split('/')[-1].split('.')[0].split('_')[0][1:]
                print(pat_id)
                #print('moving image:', moving_image.shape)
                # respace fixed img on z-direction
                z_spacing = moving_img.GetSpacing()[2]
                old_size = fixed_img.GetSize()
                old_spacing = fixed_img.GetSpacing()
                new_spacing = (1, 1, z_spacing)
                new_size = [
                    int(round((old_size[0] * old_spacing[0]) / float(new_spacing[0]))),
                    int(round((old_size[1] * old_spacing[1]) / float(new_spacing[1]))),
                    int(round((old_size[2] * old_spacing[2]) / float(new_spacing[2])))
                    ]
                #new_size = [old_size[0], old_size[1], int(round((old_size[2] * 1) / float(z_spacing)))]
                #new_size = [old_size[0], old_size[1], old_size[2]]
                if interp_type == 'linear':
                    interp_type = sitk.sitkLinear
                elif interp_type == 'bspline':
                    interp_type = sitk.sitkBSpline
                elif interp_type == 'nearest_neighbor':
                    interp_type = sitk.sitkNearestNeighbor
                resample = sitk.ResampleImageFilter()
                resample.SetOutputSpacing(new_spacing)
                resample.SetSize(new_size)
                resample.SetOutputOrigin(fixed_img.GetOrigin())
                resample.SetOutputDirection(fixed_img.GetDirection())
                resample.SetInterpolator(interp_type)
                resample.SetDefaultPixelValue(fixed_img.GetPixelIDValue())
                resample.SetOutputPixelType(sitk.sitkFloat32)
                fixed_img = resample.Execute(fixed_img)
                #print(fixed_img.shape)
                transform = sitk.CenteredTransformInitializer(
                    fixed_img, 
                    moving_img, 
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
                final_transform = registration_method.Execute(fixed_img, moving_img)                               
                moving_img_resampled = sitk.Resample(
                    moving_img, 
                    fixed_img, 
                    final_transform, 
                    sitk.sitkLinear, 
                    0.0, 
                    moving_img.GetPixelID())
                sitk.WriteImage(
                    moving_img_resampled, os.path.join(output_dir, str(int(pat_id)) + '_reg.nii.gz'))
                if save_tfm:
                    sitk.WriteTransform(final_transform, os.path.join(output_dir, str(int(pat_id)) + '_T2.tfm'))
                #except Exception as e:
                #    print(e)


def registration_elastix(input_image_path, output_path, fixed_image_path):
    
    """
    Registration with ITK-Elastix
    https://github.com/InsightSoftwareConsortium/ITKElastix
    """

    fixed_image = itk.imread(fixed_image_path, itk.F)
    # Import Parameter Map
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile(proj_dir + '/reg_parameters/Parameters_Rigid.txt')
    for img_dir in tqdm(sorted(glob.glob(input_dir + '/*.nii.gz'))):
        ID = img_dir.split('/')[-1].split('.')[0]
        if 'T2W' in ID:
            count += 1
            print(count)
            #print(ID)
            #try:
            moving_img = itk.imread(img_dir, sitk.sitkFloat32)
            # bias filed correction
            #moving_img = sitk.N4BiasFieldCorrection(moving_img)
            if cohort == 'BCH':
                pat_id = img_dir.split('/')[-1].split('.')[0]
            elif cohort == 'CBTN':
                pat_id = img_dir.split('/')[-1].split('.')[0].split('_')[0][1:]
            print(pat_id)

    if "nii" in input_image_path:
        moving_image = itk.imread(input_image_path, itk.F)
        # Call registration function
        try:
            result_image, result_transform_parameters = itk.elastix_registration_method(
                fixed_image, moving_image,
                parameter_object=parameter_object,
                log_to_console=False)
            image_id = input_image_path.split("/")[-1]
            new_dir = output_path+image_id.split(".")[0]
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)
            itk.imwrite(result_image, new_dir+"/"+image_id)
            print("Registered ", image_id)
        except:
            print("Cannot transform", input_image_path.split("/")[-1])


if __name__ == '__main__':
    
    temp_img = 'temp_head.nii.gz'
    proj_dir = '/mnt/aertslab/USERS/Zezhong/pLGG/CBTN_BCH_Data'
    T2W_dir = os.path.join(proj_dir, 'BCH_T2W')
    reg_dir = os.path.join(proj_dir, 'BCH_T2W_reg3')
    brain_dir = os.path.join(proj_dir, 'BCH_T2W_brain2')
    correction_dir = os.path.join(proj_dir, 'BCH_T2W_correction')
    pro_data_dir = os.path.join(proj_dir, 'pro_data')
    temp_dir = os.path.join(proj_dir, 'reg_temp') 
    CBTN_T2W_dir = os.path.join(proj_dir, 'T2W')
    CBTN_reg_dir = os.path.join(proj_dir, 'CBTN_T2W_reg')
    CBTN_brain_dir = os.path.join(proj_dir, 'CBTN_T2W_brain')

    run_list = ['1053918', '1136384', '2271088', '2293568', '2296306', '4040081', 
                '4365249', '4514729', '4615639', '5048067']
    register = True
    extraction = False


    if register:
        registration(
            cohort='CBTN',
            pro_data_dir=pro_data_dir, 
            input_dir=CBTN_T2W_dir, 
            output_dir=CBTN_reg_dir, 
            temp_img=temp_img,
            run_list=run_list)

    if extraction:
        brain_extraction(input_dir=reg_dir, output_dir=brain_dir)
    




