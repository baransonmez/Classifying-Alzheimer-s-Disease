import os
from glob import glob
import nibabel as nib

def_size = (176, 256, 256)


def find_path_pairs(raw_path):
    """" Due to memory issues images will be opened in feature extraction part """
    mri_paths = []
    for root in glob(raw_path + "/*/"):
        type = os.path.basename(os.path.normpath(root))
        if type == "AD-Brain":
            cls = 1
        else:
            cls = 0
        # read nii.gz
        for full_path in glob(root + "*.gz"):
            mri_paths.append((full_path, [cls], cls))

    return mri_paths


def print_different_sizes(raw_path):
    """" Eliminate the data with different size """
    for full_path in glob(raw_path + "/*/*/*/*.gz"):
        x = nib.load(full_path)
        if x.shape != def_size:
            print(full_path)
            print(x.shape)

