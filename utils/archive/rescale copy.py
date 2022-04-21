
import SimpleITK as sitk
import sys
import os

# 9. rescale to a common "more compact" size (either downsample or upsample)
from rescale import rescale
dataset = "rtog"
patient_id = "200"
data_type = "ct"
path_to_nrrd = "/data/output/dataset1_123_ct_interpolated_resized_raw_xx.nrrd"
interpolation_type = "linear" #"nearest_neighbor"
new_size = (108, 108, 128) # x,y,z
return_type = "numpy_array"
output_dir = "/data/output"
rescaled_nrrd = rescale(dataset, patient_id, data_type, path_to_nrrd, interpolation_type,
                        new_size, return_type, output_dir)


def rescale(dataset, patient_id, data_type, path_to_nrrd, interpolation_type,
            new_size, return_type, output_dir = ""):
    """
    Rescale a given nrrd file to a given size - either downsample or upsample.
    Args:
        dataset (str): Name of dataset.
        patient_id (str): Unique patient id.
        data_type (str): Type of data (e.g., ct, pet, mri, lung(mask), heart(mask)..)
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
    try:
        # calculate new spacing
        img = sitk.ReadImage(path_to_nrrd)
        old_size = img.GetSize()
        old_spacing = img.GetSpacing()
        new_spacing = (
            (old_size[0]*old_spacing[0])/new_size[0],
            (old_size[1]*old_spacing[1])/new_size[1],
            (old_size[2]*old_spacing[2])/new_size[2]
            )
        print('{} {}'.format('new spacing: ', new_spacing))
        # interpolate
        new_sitk_object = sitk_interpolation(path_to_nrrd, interpolation_type, new_spacing)
        # check if the result shape matches the target
        assert new_sitk_object.GetSize() == new_size, "oops.. The shape of the returned array does not match your requested target shape."
        # save and return
        if output_dir != "":
            # write new nrrd
            writer = sitk.ImageFileWriter()
            writer.SetFileName(os.path.join(output_dir,
                "{}_{}_{}_interpolated_resized_rescaled_xx.nrrd".format(dataset, patient_id, data_type)))
            writer.SetUseCompression(True)
            writer.Execute(new_sitk_object)

        if return_type == "sitk_object":
            return new_sitk_object
        elif return_type == "numpy_array":
            return sitk.GetArrayFromImage(new_sitk_object)

    except Exception as e:
        print ("Error in {}_{}, {}".format(dataset, patient_id, e))

def sitk_interpolation(path_to_nrrd, interpolation_type, new_spacing):

    data = sitk.ReadImage(path_to_nrrd)
    original_spacing = data.GetSpacing()
    original_size = data.GetSize()
    print('{} {}'.format('original size: ', original_size))
    print('{} {}'.format('original spacing: ', original_spacing))

    new_size = [
        int(round((original_size[0]*original_spacing[0])/float(new_spacing[0]))),
        int(round((original_size[1]*original_spacing[1])/float(new_spacing[1]))),
        int(round((original_size[2]*original_spacing[2])/float(new_spacing[2])))
        ]

    print('{} {}'.format('new size: ', new_size))

    # http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/20_Expand_With_Interpolators.html
    if interpolation_type == 'linear':
        interpolation_type = sitk.sitkLinear
    elif interpolation_type == 'bspline':
        interpolation_type = sitk.sitkBSpline
    elif interpolation_type == 'nearest_neighbor':
        interpolation_type = sitk.sitkNearestNeighbor

    resampleImageFilter = sitk.ResampleImageFilter()
    new_image = resampleImageFilter.Execute(
        data,
        new_size,
        sitk.Transform(),
        interpolation_type,
        data.GetOrigin(),
        [float(x) for x in new_spacing],
        data.GetDirection(),
        0,
        data.GetPixelIDValue()
        )
    new_image.SetSpacing(new_spacing)

    return new_image

    
