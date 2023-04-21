import nibabel as nb
from deepbrain import Extractor


def skull_strip():

    # Load a nifti as 3d numpy image [H, W, D]
    img = nib.load(img_path).get_fdata()

    ext = Extractor()

    # `prob` will be a 3d numpy image containing probability 
    # of being brain tissue for each of the voxels in `img`
    prob = ext.run(img) 

    # mask can be obtained as:
    mask = prob > 0.5
