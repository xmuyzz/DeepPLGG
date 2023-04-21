import SimpleITK as sitk
import sys
import os
import matplotlib.pyplot as plt



def resize_3d(img_dir, interp_type, resize_shape): 

    """
    rescale to a common "more compact" size (either downsample or upsample)
    """

    # calculate new spacing
    image = sitk.ReadImage(img_dir)
    input_size = image.GetSize()
    input_spacing = image.GetSpacing()
    # keep the original size and spacing in z direction
    output_size = (resize_shape[0], resize_shape[1], input_size[2])
    output_spacing = (
        (input_size[0] * input_spacing[0]) / output_size[0],
        (input_size[1] * input_spacing[1]) / output_size[1],
        input_spacing[2])
    
    if interp_type == 'linear':
        interp_type = sitk.sitkLinear
    elif interp_type == 'bspline':
        interp_type = sitk.sitkBSpline
    elif interp_type == 'nearest_neighbor':
        interp_type = sitk.sitkNearestNeighbor
    
    resample = sitk.ResampleImageFilter()
    resample.SetSize(output_size)
    resample.SetOutputSpacing(output_spacing)
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetInterpolator(interp_type)
    img_nrrd = resample.Execute(image) 
    
    img_arr = sitk.GetArrayFromImage(img_nrrd)

    return img_arr



