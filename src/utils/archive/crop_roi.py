import os
import itertools
import operator
import numpy as np
import SimpleITK as sitk
from util import get_arr_from_nrrd, generate_sitk_obj_from_npy_array
from scipy import ndimage

def get_spacing(sitk_obj):
    """
    flip spacing from sitk (x,y,z) to numpy (z,y,x)
    """
    spacing = sitk_obj.GetSpacing()
    
    return (spacing[2], spacing[1], spacing[0])

def get_arr_from_nrrd(link_to_nrrd, type):
    '''
    Used for images or labels.
    '''
    sitk_obj = sitk.ReadImage(link_to_nrrd)
    spacing = get_spacing(sitk_obj)
    origin = sitk_obj.GetOrigin()
    arr = sitk.GetArrayFromImage(sitk_obj)
    
    if type=="label":
        arr = threshold(arr)
        assert arr.min() == 0, "minimum value is not 0"
        assert arr.max() == 1, "minimum value is not 1"
        assert len(np.unique(arr)) == 2, "arr does not contain 2 unique values"
    
    return sitk_obj, arr, spacing, origin


def generate_sitk_obj_from_npy_array(image_sitk_obj, pred_arr, resize=True, output_dir=""):

    """
    When resize==True: Used for saving predictions where padding needs to be added to increase the size of the prediction and match that of input to model. This function matches the size of the array in image_sitk_obj with the size of pred_arr, and saves it. This is done equally on all sides as the input to model and model output have different dims to allow for shift data augmentation.
    When resize==False: the image_sitk_obj is only used as a reference for spacing and origin. The numpy array is not resized.
    image_sitk_obj: sitk object of input to model
    pred_arr: returned prediction from model - should be squeezed.
    NOTE: image_arr.shape will always be equal or larger than pred_arr.shape, but never smaller given that
    we are always cropping in data.py
    """
    if resize==True:
        # get array from sitk object
        image_arr = sitk.GetArrayFromImage(image_sitk_obj)
        # change pred_arr.shape to match image_arr.shape
        # getting amount of padding needed on each side
        z_diff = int((image_arr.shape[0] - pred_arr.shape[0]) / 2)
        y_diff = int((image_arr.shape[1] - pred_arr.shape[1]) / 2)
        x_diff = int((image_arr.shape[2] - pred_arr.shape[2]) / 2)
        # pad, defaults to 0
        pred_arr = np.pad(pred_arr, ((z_diff, z_diff), (y_diff, y_diff), (x_diff, x_diff)), 'constant')
        assert image_arr.shape == pred_arr.shape, "returned array shape does not match your requested shape."

    # save sitk obj
    new_sitk_object = sitk.GetImageFromArray(pred_arr)
    new_sitk_object.SetSpacing(image_sitk_obj.GetSpacing())
    new_sitk_object.SetOrigin(image_sitk_obj.GetOrigin())

    if output_dir != "":
        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_dir)
        writer.SetUseCompression(True)
        writer.Execute(new_sitk_object)
    
    return new_sitk_object

def crop_top_image_only(dataset, patient_id, path_to_image_nrrd, crop_shape, return_type, output_folder_image):
    
    """
    Will center the image and crop top of image after it has been registered.
    Args:
        dataset (str): Name of dataset.
        patient_id (str): Unique patient id.
        path_to_image_nrrd (str): Path to image nrrd file.
        path_to_label_nrrd (str): Path to label nrrd file.
        crop_shape (list) shape to save cropped image  (x, y, z)
        return_type (str): Either 'sitk_object' or 'numpy_array'.
        output_folder_image (str) path to folder to save image nrrd
        output_folder_label (str) path to folder to save label nrrd
    Returns:
        Either a sitk image object or a numpy array derived from it 
        (depending on 'return_type') of both image and label.
    Raises:
        Exception if an error occurs.
    """
    
    try:
        # get image, arr, and spacing
        image_obj, image_arr, image_spacing, image_origin = get_arr_from_nrrd(path_to_image_nrrd, "image")
        
        ## Return top 25 rows of 3D volume, centered in x-y space / start at anterior (y=0)?
        print("image_arr shape: ", image_arr.shape)
        c, y, x = image_arr.shape
        
        ## Get center of mass to center the crop in Y plane
        mask_arr = np.copy(image_arr) 
        mask_arr[mask_arr > -500] = 1
        mask_arr[mask_arr <= -500] = 0
        mask_arr[mask_arr >= -500] = 1 
        print("mask_arr min and max:", np.amin(mask_arr), np.amax(mask_arr))
        centermass = ndimage.measurements.center_of_mass(mask_arr) # z,x,y   
        cpoint = c - crop_shape[2]//2
        print("cpoint, ", cpoint)
        centermass = ndimage.measurements.center_of_mass(mask_arr[cpoint, :, :])   
        print("center of mass: ", centermass)
        startx = int(centermass[0] - crop_shape[0]//2)
        starty = int(centermass[1] - crop_shape[1]//2)      
        #startx = x//2 - crop_shape[0]//2       
        #starty = y//2 - crop_shape[1]//2
        startz = int(c - crop_shape[2])
        print("start X,Y,Z: ", startx, starty, startz)
        if startz < 0:
            image_arr = np.pad(
                image_arr,
                ((abs(startz)//2, abs(startz)//2), (0, 0), (0, 0)), 
                'constant', 
                constant_values=-1024
                )
            image_arr_crop = image_arr[
                0:crop_shape[2], 
                starty:starty + crop_shape[1], 
                startx:startx + crop_shape[0]
                ]
        else:
            image_arr_crop = image_arr[
                0:crop_shape[2], 
                starty:starty + crop_shape[1], 
                startx:startx + crop_shape[0]
                ]
        if image_arr_crop.shape[0] < crop_shape[2]:
            print("initial cropped image shape too small:", image_arr_crop.shape)
            print(crop_shape[2], image_arr_crop.shape[0])
            image_arr_crop = np.pad(
                image_arr_crop,
                ((int(crop_shape[2] - image_arr_crop.shape[0]), 0), (0, 0), (0, 0)),
                'constant',
                constant_values=-1024
                )
            print("padded size: ", image_arr_crop.shape)
        print('Returning bottom rows')
        
        ### save nrrd
        output_path_image = os.path.join(output_folder_image, "{}_{}_image_interpolated_roi_raw_gt.nrrd".format(dataset, patient_id))
        image_crop_sitk = generate_sitk_obj_from_npy_array(
            image_obj, 
            image_arr_crop, 
            resize=False, 
            output_dir=output_path_image
            )
        print("Saving image cropped")
        
        if return_type == "sitk_object":
            return image_crop_sitk
        elif return_type == "numpy_array":
            return image_arr_crop
    except Exception as e:
        print ("Error in {}_{}, {}".format(dataset, patient_id, e))
