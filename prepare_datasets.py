# ==========================================================
#
#  This prepare the hdf5 datasets of the DRIVE database
#
# ============================================================

import os
import h5py
import numpy as np
from PIL import Image


def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


# ------------Path of the images --------------------------------------------------------------
# train
original_imgs_train = "/mnt/nas/DRIVE_liot/training/images/"
groundTruth_imgs_train = "/mnt/nas/DRIVE_liot/training/1st_manual/"
borderMasks_imgs_train = "/mnt/nas/DRIVE_liot/training/mask/"
print("test")
print("test")

# test
original_imgs_test = "/mnt/nas/DRIVE/test/images/"
groundTruth_imgs_test = "/mnt/nas/DRIVE/test/1st_manual/"
borderMasks_imgs_test = "/mnt/nas/DRIVE/test/mask/"

# ---------------------------------------------------------------------------------------------

channels = 4
#DRIVE
height = 584
width = 565
# hdf5 path
dataset_path = "/mnt/nas/DRIVEliot_datasets_training_testing/"

def get_datasets(imgs_dir, groundTruth_dir, borderMasks_dir, train_test="null"):
    Nimgs = len(os.listdir(imgs_dir))
    print("Nimgs",Nimgs)
    imgs = np.empty((Nimgs, height, width, channels))
    groundTruth = np.empty((Nimgs, height, width))
    border_masks = np.empty((Nimgs, height, width))
    for path, subdirs, files in os.walk(imgs_dir):  # list all files, directories in the path
        for i in range(len(files)):
            # corresponding original image
            print("original image: " + files[i])
            img = Image.open(imgs_dir + files[i])
            imgs[i] = np.asarray(img)

            # corresponding ground truth
            groundTruth_name = files[i][0:2] + "_manual1.gif"#DRIVE
            g_truth = Image.open(groundTruth_dir + groundTruth_name)
            groundTruth[i] = np.asarray(g_truth)

            # corresponding border masks
            border_masks_name = ""
            if train_test == "train":
                border_masks_name = files[i][0:2] + "_training_mask.gif"#DRIVE

            elif train_test == "test":
                border_masks_name = files[i][0:2] + "_test_mask.gif"#DRIVE

            else:
                print("specify if train or test!!")
                exit()
            b_mask = Image.open(borderMasks_dir + border_masks_name)
            border_masks[i] = np.asarray(b_mask)

    assert (np.max(groundTruth) == 255 and np.max(border_masks) == 255)
    assert (np.min(groundTruth) == 0 and np.min(border_masks) == 0)
    print("ground truth and border masks are correctly withih pixel value range 0-255 (black-white)")
    # reshaping for my standard tensors
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    assert (imgs.shape == (Nimgs, channels, height, width))
    groundTruth = np.reshape(groundTruth, (Nimgs, 1, height, width))
    border_masks = np.reshape(border_masks, (Nimgs, 1, height, width))
    assert (groundTruth.shape == (Nimgs, 1, height, width))
    assert (border_masks.shape == (Nimgs, 1, height, width))
    return imgs, groundTruth, border_masks


if not os.path.exists(dataset_path):
   os.makedirs(dataset_path)

# getting the training datasets
imgs_train, groundTruth_train, border_masks_train = get_datasets(original_imgs_train, groundTruth_imgs_train,
                                                                 borderMasks_imgs_train, "train")
print("saving train datasets")
write_hdf5(imgs_train, dataset_path + "DRIVE_dataset_imgs_train.hdf5")
write_hdf5(groundTruth_train, dataset_path + "DRIVE_dataset_groundTruth_train.hdf5")
write_hdf5(border_masks_train, dataset_path + "DRIVE_dataset_borderMasks_train.hdf5")




# getting the testing datasets
imgs_test, groundTruth_test, border_masks_test = get_datasets(original_imgs_test, groundTruth_imgs_test,
                                                              borderMasks_imgs_test, "test")
print("saving test datasets")
write_hdf5(imgs_test, dataset_path + "DRIVE_dataset_imgs_test.hdf5")
write_hdf5(groundTruth_test, dataset_path + "DRIVE_dataset_groundTruth_test.hdf5")
write_hdf5(border_masks_test, dataset_path + "DRIVE_dataset_borderMasks_test.hdf5")
