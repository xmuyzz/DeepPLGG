import numpy as np
import SimpleITK as sitk
from interpolate import sitk_interpolation
import os

def rescale(dataset, patient_id, data_type, path_to_nrrd, interpolation_type, new_size, return_type, output_dir = ""):
    """
    Rescale a given nrrd file to a given size - either downsample or upsample.
    Args:
        dataset (str): Name of dataset.
        patient_id (str): Unique patient id.
        data_type (str): Type of data (e.g., ct, pet, mri, lung(mask), heart(mask)..)
        path_to_nrrd (str): Path to nrrd file.
        interpolation_type (str): Either 'linear' (for images with continuous values), 'bspline' (also for images but will mess up the range of the values), or 'nearest_neighbor' (for masks with discrete values).
        new_size (tuple): Tuple containing 3 values for shape to resample to. (x,y,z)
        return_type (str): Either 'sitk_object' or 'numpy_array'.
        output_dir (str): Optional. If provided, nrrd file will be saved there. If not provided, file will not be saved.
    Returns:
        Either a sitk image object or a numpy array derived from it (depending on 'return_type').
    Raises:
        Exception if an error occurs.
    """
    try:
        # calculate new spacing
        img = sitk.ReadImage(path_to_nrrd)
        old_size = img.GetSize()
        old_spacing = img.GetSpacing()
        new_spacing = ((old_size[0]*old_spacing[0])/new_size[0],
                        (old_size[1]*old_spacing[1])/new_size[1],
                        (old_size[2]*old_spacing[2])/new_size[2])
        print('{} {}'.format('new spacing: ', new_spacing))
        # interpolate
        new_sitk_object = sitk_interpolation(path_to_nrrd, interpolation_type, new_spacing)
        # check if the result shape matches the target
        assert new_sitk_object.GetSize() == new_size, "oops.. The shape of the returned array does not match your requested target shape."
        # save and return
        if output_dir != "":
            # write new nrrd
            writer = sitk.ImageFileWriter()
            writer.SetFileName(os.path.join(output_dir, "{}_{}_{}_interpolated_resized_rescaled_xx.nrrd".format(dataset, patient_id, data_type)))
            writer.SetUseCompression(True)
            writer.Execute(new_sitk_object)

        if return_type == "sitk_object":
            return new_sitk_object
        elif return_type == "numpy_array":
            return sitk.GetArrayFromImage(new_sitk_object)

    except Exception as e:
        print ("Error in {}_{}, {}".format(dataset, patient_id, e))



def downsample_img(dataset, patient_id, data_type, path_to_nrrd, target_shape, return_type, output_dir=""):
    """
    Not used. Resizes a given nrrd file to a given size in all three dimensions. This is an older downsampling function that uses block reduce. It works well for images, but needs to have a flag for working with masks (to ensure a binary output is acheived). Finally, reusing the interpolate function to downsample/upsample was found to be the best approach.
    Args:
        dataset (str): Name of dataset.
        patient_id (str): Unique patient id.
        data_type (str): Type of data (e.g., ct, pet, mri, lung(mask), heart(mask)..)
        path_to_nrrd (str): Path to nrrd file.
        target_shape (str): Tuple containing 3 values for size to downsample to: (x,y,z).
        return_type (str): Either 'sitk_object' or 'numpy_array'.
        output_dir (str): Optional. If provided, nrrd file will be saved there. If not provided, file will not be saved.
    Returns:
        Either a sitk image object or a numpy array derived from it (depending on 'return_type').
    Raises:
        Exception if an error occurs.
    """
    try:
        # load image and convert to array
        img = sitk.ReadImage(path_to_nrrd)
        arr = sitk.GetArrayFromImage(img)
        # figure our downsampling block size
        block = (arr.shape[0]/target_shape[2], arr.shape[1]/target_shape[1], arr.shape[2]/target_shape[0])
        # check if all are integers
        assert block[0].is_integer(), "Dividing your input Z shape and target Z shape should produce a whole number."
        assert block[1].is_integer(), "Dividing your input Y shape and target Y shape should produce a whole number."
        assert block[2].is_integer(), "Dividing your input X shape and target X shape should produce a whole number."
        # downsample. Since we asserted above already, we can safely convert the block values to integers.
        new_arr = block_reduce(arr, tuple([int(x) for x in block]), np.average)
        # calculate new spacing based on original_spacing and target_shape. Since we are downsampling, the assumption here is that the new spacing values will be greater than original spacing values.
        original_spacing = img.GetSpacing()
        new_spacing = (original_spacing[0]*block[2], original_spacing[1]*block[1], original_spacing[2]*block[0])
        # create SITK image
        new_sitk_object = sitk.GetImageFromArray(new_arr)
        new_sitk_object.SetSpacing(new_spacing)
        new_sitk_object.SetOrigin(img.GetOrigin())
        # print
        print('Orginal shape: {}'.format(img.GetSize()))
        print('Target shape: {}'.format(target_shape))
        print('Block (Z-first) : {}'.format(block))
        print('Original spacing: {}'.format(original_spacing))
        print('Target spacing: {}'.format(new_spacing))
        print('Shape after downsample: {}'.format(new_sitk_object.GetSize()))
        # check if the result shape matches the target
        assert new_sitk_object.GetSize() == target_shape, "oops.. The shape of the returned array does not match your requested target shape."
        # save and return
        if output_dir != "":
            writer = sitk.ImageFileWriter()
            writer.SetFileName(os.path.join(output_dir, "{}_{}_{}_interpolated_resized_downsampled_xx.nrrd".format(dataset, patient_id, data_type)))
            writer.SetUseCompression(True)
            writer.Execute(new_sitk_object)
        if return_type == "sitk_object":
            return new_sitk_object
        elif return_type == "numpy_array":
            return data
    except Exception as e:
        print ("Error in {}_{}, {}".format(dataset, patient_id, e))
