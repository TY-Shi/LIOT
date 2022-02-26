import numpy as np
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from scipy.misc import imresize
from scipy.ndimage.interpolation import rotate

def random_flip(img, mask, u=1):
    if np.random.random() < u:
        img = torch.flip(img, [3])
        mask = torch.flip(mask, [3])
    if np.random.random() < u:
        img = torch.flip(img, [2])
        mask = torch.flip(mask, [2])
    return img, mask

def random_rotate(img, mask, rotate_limit=(-20, 20), u=1):
    if np.random.random() < u:
        theta = np.random.uniform(rotate_limit[0], rotate_limit[1])
        img = F.rotate(img, theta)
        mask = F.rotate(mask, theta)
    return img, mask

def shift(x, wshift, hshift, row_axis=0, col_axis=1):
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = hshift * h
    ty = wshift * w
    x = image.apply_affine_transform(x, ty=ty, tx=tx)
    return x


def random_shift(img, mask, w_limit=(-0.1, 0.1), h_limit=(-0.1, 0.1), u=1):
    if np.random.random() < u:
        wshift = np.random.uniform(w_limit[0], w_limit[1])
        hshift = np.random.uniform(h_limit[0], h_limit[1])
        img = F.affine(img,angle=0,translate=(wshift,hshift), scale=1.0,shear=0)
        mask = F.affine(mask,angle=0,translate=(wshift,hshift), scale=1.0,shear=0)
    return img, mask


def random_zoom(img, mask, zoom_range=(0.8, 1), u=1):
    if np.random.random() < u:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
        img = F.affine(img,angle=0,translate=(0,0), scale=zx,shear=0)
        mask = F.affine(mask,angle=0,translate=(0,0), scale=zx,shear=0)
    return img, mask


def random_shear(img, mask, intensity_range=(-0.5, 0.5), u=1):
    if np.random.random() < u:
        sh = np.random.uniform(intensity_range[0], intensity_range[1])
        img = F.affine(img,angle=0,translate=(0,0), scale=1.0,shear=sh)
        mask = F.affine(mask,angle=0,translate=(0,0), scale=1.0,shear=sh)
    return img, mask


def random_gray(img, u=0.5):
    if np.random.random() < u:
        coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
        gray = np.sum(img * coef, axis=2)
        img = np.dstack((gray, gray, gray))
    return img


def random_contrast(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        img = F.adjust_contrast(img, alpha)
    return img


def random_brightness(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        img = F.adjust_brightness(img,alpha)
    return img


def random_saturation(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        img = F.adjust_saturation(img,alpha)
    return img


def random_channel_shift(x, limit, channel_axis=2):
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_ch + np.random.uniform(-limit, limit), min_x, max_x) for x_ch in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def random_augmentation(img, mask):
    #(N,C,H,W)
    img = torch.tensor(img, dtype=torch.float32)
    mask = torch.tensor(mask, dtype=torch.long)
    img, mask = random_rotate(img, mask, rotate_limit=(-180, 180), u=0.5)
    img, mask = random_shear(img, mask, intensity_range=(-5, 5), u=0.05)
    img, mask = random_flip(img, mask, u=0.5)
    img, mask = random_shift(img, mask, w_limit=(-0.1, 0.1), h_limit=(-0.1, 0.1), u=0.05)
    img, mask = random_zoom(img, mask, zoom_range=(0.8, 1.2), u=0.05)
    return img, mask

if __name__ == "__main__":
    im = Image.open('./21_training.tif')
    im = np.array(im)
    im = torch.tensor(im)
    im = torch.unsqueeze(im,0)
    im = im.permute(0, 3, 1, 2)
    print(im.shape)
    mask = im
    plt.title("Orgin")
    plt.imshow(im[0].permute(1, 2, 0))
    plt.show()
    img_bright = random_contrast(im, limit=(-0.3, 0.3), u=1)
    plt.title("Bright")
    plt.imshow(img_bright[0].permute(1, 2, 0))
    plt.show()



