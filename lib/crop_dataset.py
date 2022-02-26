import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

import os
import json
import numpy as np
from PIL import Image
from help_functions import load_hdf5
from pre_processing import my_augment
from extract_patches import extract_random_img

class CurvilinearDataset(Dataset):
    def __init__(self, data_train_imgs_loader, data_train_mask_loader, repeat = 30, batch_size=32, patch_h=128, patch_w=128, joint_augment=None, augment=None, target_augment=None):
        self.train_imgs_original = load_hdf5(data_train_imgs_loader)
        self.train_masks = load_hdf5(data_train_mask_loader)
        self.train_imgs, self.train_masks = my_augment(self.train_imgs_original, self.train_masks)
        if data_train_imgs_loader.find("CHASEDB1") or data_train_imgs_loader.find("STARE"):
            self.train_masks[self.train_masks >= 0.5] = 1
            self.train_masks[self.train_masks < 0.5] = 0
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.joint_augment = joint_augment
        self.augment = augment
        self.target_augment = target_augment
        self.batch_size = batch_size
        self.repeat = repeat
        self.data_len = self.train_imgs_original.shape[0] * self.repeat

    def __len__(self):
        return max(self.data_len, self.batch_size)

    def __getitem__(self, index):
        index = index % (self.data_len//self.repeat)
        image = self.train_imgs[index]
        gt = self.train_masks[index]
        patch_image, patch_gt = extract_random_img(image, gt,self.patch_h,self.patch_w)

        return patch_image, patch_gt


class CurvilinearDataset_val(Dataset):
    def __init__(self, data_test_imgs_loader, data_test_mask_loader, repeat = 2, batch_size=32, patch_h=128, patch_w=128, joint_augment=None, augment=None, target_augment=None):
        self.test_imgs_original = load_hdf5(data_test_imgs_loader)
        self.test_masks = load_hdf5(data_test_mask_loader)
        self.test_imgs, self.test_masks = my_augment(self.test_imgs_original, self.test_masks)
        if data_test_imgs_loader.find("CHASEDB1") or data_test_imgs_loader.find("STARE"):
            self.test_masks[self.test_masks >= 0.5] = 1
            self.test_masks[self.test_masks < 0.5] = 0
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.joint_augment = joint_augment
        self.augment = augment
        self.target_augment = target_augment
        self.batch_size = batch_size
        self.repeat = repeat
        self.data_len = self.test_imgs_original.shape[0] * self.repeat

    def __len__(self):
        return max(self.data_len, self.batch_size)

    def __getitem__(self, index):
        index = index % (self.data_len//self.repeat)
        image = self.test_imgs[index]
        gt = self.test_masks[index]
        patch_image, patch_gt = extract_random_img(image, gt,self.patch_h,self.patch_w)
        return patch_image, patch_gt


if __name__ == "__main__":
    image_dir = "/mnt/nas/DRIVEliot_datasets_training_testing/DRIVE_dataset_imgs_train.hdf5"
    gt_dir = "/mnt/nas/DRIVEliot_datasets_training_testing/DRIVE_dataset_imgs_train.hdf5"
    val_image_dir = "/mnt/nas/DRIVEliot_datasets_training_testing/DRIVE_dataset_imgs_train.hdf5"
    val_gt_dir = "/mnt/nas/DRIVEliot_datasets_training_testing/DRIVE_dataset_imgs_train.hdf5"
    CurvilinearData = CurvilinearDataset(image_dir, gt_dir)
    CurvilinearDataset_val(val_image_dir,val_gt_dir)
