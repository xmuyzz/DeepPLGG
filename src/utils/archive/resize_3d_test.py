#--------------------------------------------------------------------------
# rescale to a common "more compact" size (either downsample or upsample)
#--------------------------------------------------------------------------

import SimpleITK as sitk
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import nrrd


def resize_3d(nrrd_image, interp_type, output_size):
    
    """
    Rescale a given nrrd file to a given size - either downsample or upsample.
    Args:
        path_to_nrrd (str): Path to nrrd file.
        interpolation_type (str): Either 'linear' (for images with continuous values),
                                  'bspline' (also for images but will mess up the range of the values),
                                  or 'nearest_neighbor' (for masks with discrete values).
        new_size (tuple): Tuple containing 3 values for shape to resample to. (x,y,z)
        return_type (str): Either 'sitk_object' or 'numpy_array'.
        output_dir (str): Optional. If provided, nrrd file will be saved there. If not provided, file will not be saved.
    Returns:
        Either a sitk image object or a numpy array derived from it (depending on 'return_type').
    Raises:
        Exception if an error occurs.
    """
    
    ### calculate new spacing
    image = sitk.ReadImage(nrrd_image)
    input_size = image.GetSize()
    input_spacing = image.GetSpacing()
    output_spacing = (
        (input_size[0] * input_spacing[0]) / output_size[0],
        (input_size[1] * input_spacing[1]) / output_size[1],
        (input_size[2] * input_spacing[2]) / output_size[2]
        )
    print('{} {}'.format('input spacing: ', input_spacing))
    print('{} {}'.format('output spacing: ', output_spacing))
    ### choose interpolation algorithm
    if interp_type == 'linear':
        interp_type = sitk.sitkLinear
    elif interp_type == 'bspline':
        interp_type = sitk.sitkBSpline
    elif interp_type == 'nearest_neighbor':
        interp_type = sitk.sitkNearestNeighbor
    ### interpolate
    resample = sitk.ResampleImageFilter()
    resample.SetSize(output_size)
    resample.SetOutputSpacing(output_spacing)
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetInterpolator(interp_type)
    output_image = resample.Execute(image)
    #print(output_image.GetSize())
    ### convert img to np array
    resized_arr = sitk.GetArrayFromImage(output_image)
    ### plot image
    data = resized_arr
    data[data <= -1024] = -1024
    data = np.interp(data, [-1024, 3017], [0, 1])
    print(data.shape)
    data = data[4:36, :, :]
    print(resized_arr.shape)
    print(data.shape)
    #print(resized_arr[2, 30, 10])
    #print(data[2, 30, 10])
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(resized_arr[2, :, :], cmap='gray')
    ax2.imshow(data[31, :, :], cmap='gray')
    plt.show()

    return resized_arr

nrrd_image = '/Volumes/YZZ_HDD/contrast_detection/test/PMH/PMH_OPC-00049_CT-SIM_raw_raw_raw_xx.nrrd'

np_array = resize_3d(
    nrrd_image=nrrd_image,
    interp_type='linear',
    output_size=(64, 64, 36)
    )

##img, header = nrrd.read(nrrd_image)
##fig, ax = plt.subplots(1, 1)
##ax.imshow(img[:, :, 2], cmap='gray')
##plt.show()


