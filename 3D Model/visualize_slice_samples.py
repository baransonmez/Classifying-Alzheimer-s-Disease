import nibabel as nib
import matplotlib.pyplot as plt


def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


def show_3_slice(path):
    """ Show sample 3 slices from 3D image"""
    x = nib.load(path)
    x = x.get_fdata()

    print("Size of the volume is " + str(x.shape))

    slice_1 = x[75, :, :]
    slice_0 = x[:, 75, :]
    slice_2 = x[:, :, 75]

    show_slices([slice_1, slice_0, slice_2])
    plt.suptitle("Central slices from x,y,z axis")
    plt.show()

