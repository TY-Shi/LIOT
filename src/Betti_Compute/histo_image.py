"""
This is a module for svs files used for the analysis of histology images
"""

import numpy as np

import skimage.feature
import skimage.io
import skimage.measure
import skimage.color
from skimage.filters import sobel
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.util.shape import view_as_windows

import scipy.ndimage
from scipy.stats import gaussian_kde

import os

from sklearn.feature_extraction import image
from sklearn.neighbors import KDTree
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



def load_png(path):
    """
        A simple wrapper for the skimage library

        :param path: path to the image
        :type path: string

        :returns: np.array containing the loaded image
    """
    return np.array(skimage.io.imread(path))

def cut_svs(path, bf_folder, corner=[0,0],size=[256,256], name="patch.jpg"):

    """
        A method for cutting a patch from an svs file. The method uses the bfconver tool from here.
        https://docs.openmicroscopy.org/bio-formats/5.7.1/users/comlinetools/index.html

        :param path: path to the image
        :type path: string

        :param corner: top left corner of the patch

        :param size: width and height of the desired patch

        :returns: np.array containing the patch
    """
    exec_string = "echo y | "+bf_folder+"./bfconvert "+path+" -crop "+str(corner[0])+","+str(corner[1])+","+str(size[0])+","+str(size[1])+" "+name+" >/dev/null"
    os.system(exec_string)
    try:
        array = np.array(skimage.io.imread(name))
        if name == "patch.jpg":
            os.remove(name)
        return array
    except:
        if name == "patch.jpg":
            os.remove(name)
        return []



def compute_density_function_points(points,invert=False,method='scott'):

    """
        A method for computing a Gaussian Kernel Density Function (KDF) on a point cloud.
        Values of the function evaluated are returned for each point.

        :param points: list of points
        :type points: list of pairs

        :param invert: if True the inverse of the KDE estimated is returned

        :param method: method used for the gaussian_kde function

        :returns: list of values where the i-th value is the KDF value estimated for the i-th point
    """

    #invert=True: swap maxima and minima for the persistence diagram
    if len(points) > 0:
        x,y = zip(*points)
        others=np.array([np.array(x),np.array(y)])
        Z = gaussian_kde(others, bw_method=method)
        Z = Z(others)*pow(10,6)
        if invert:
            -Z
        return Z
    else:
        return []

def compute_density_function_grid(points,grid=[256,256],invert=False,method='scott'):

    """
        A method for computing a Gaussian Kernel Density Function (KDF) on a point cloud.
        Values of the function evaluated are returned on a grid.

        :param points: list of points
        :type points: list of pairs

        :param grid: dimension of the grid provided in output

        :param invert: if True the inverse of the KDE estimated is returned

        :param method: method used for the gaussian_kde function

        :returns: np.array encoding the evaluated KDF
    """

    x,y = zip(*points)
    pts=np.array([np.array(x),np.array(y)])
    xmin = np.min(y)
    xmax = np.max(y)
    ymin = np.min(x)
    ymax = np.max(x)
    X, Y = np.mgrid[xmin:xmax:grid[0]*1j, ymin:ymax:grid[1]*1j]
    positions = np.vstack([Y.ravel(), X.ravel()])
    kernel = gaussian_kde(pts,bw_method=method)
    Z = np.reshape(kernel(positions).T, X.shape)
    Z = Z*pow(10,6)
    if invert:
        return np.max(Z)-Z
    return Z

def compute_distance_to_a_measure(points, image_size, nn=10, scale=1):
    """
        A method for computing a distance function from a point cloud.
        Values of the function evaluated are returned on a grid.
        This simulate the construction of a RIPS complex for a 2D domain

        :param points: list of points
        :type points: np.array (number of points * 2)

        :param image_size: dimension of the image from which the points have been extracted

        :returns: np.array encoding the evaluated distance
    """
    real_size = image_size
    real_size[0] = int(real_size[0]*scale)
    real_size[1] = int(real_size[1]*scale)

    points2 = points*scale

    tree = KDTree(points2)

    print(str(real_size[0])+" "+str(real_size[1]))
    x,y = np.meshgrid(range(real_size[0]),range(real_size[1]))
    x = np.reshape(x,[real_size[0]*real_size[1]])
    y = np.reshape(y,[real_size[0]*real_size[1]])
    qr_pts = list(zip(x,y))
    print(str(len(qr_pts)))

    dist, ind = tree.query(qr_pts, k=nn)

    dist = np.sum(dist,axis=1)/nn
    image = np.zeros([real_size[0],real_size[1]])

    for i in range(real_size[0]*real_size[1]):
        image[qr_pts[i]] = dist[i]

    return image

def save_points(points, name='points.txt'):
    """
        A method for saving a point cloud extracted from an histopatology image

        :param points: point cloud to be saved
        :type points: np.array
        :param name: name of the output file
        :type name: String
    """
    np.savetxt(name,points);

def load_points(name):
    """
        A method for loading a point cloud from a txt file

        :param name: name of the input file
        :type name: String

        :returns: np.array containing the point cloud
    """
    return np.loadtxt(name);

def patches_from_annotation(image,annotation,min_area=500000,max_area=10000000,buffer=100):

    """
        A method for creating a set of patches from an annotation masks.
        For each blob in the annotation mask a new patch will be returned as an np.array.
        This has been implemented based on the IvyGap annotations.

        :param image: path to the image to split
        :type image: String

        :param annotation: path to the annotation mask
        :type annotation: String

        :param min_area: minimum area for an object to be used as a patch
        :param buffer: additional buffer around the patch (number of pixels per side)

        :returns: list of np.array each one containing a distinct patch
    """

    tissue = skimage.io.imread(image)
    ann_image = skimage.io.imread(annotation, as_grey=True)

    ann_image[ann_image[:,:] > 0.1] = 1.0
    labels = skimage.measure.label(ann_image < 1.0)
    objProps = skimage.measure.regionprops(labels)

    patches = []
    for obj in objProps:
        if(obj.area > min_area+(buffer*buffer) and obj.area < max_area):

            min_r = obj.bbox[0]-buffer
            if min_r < 0:
                min_r = 0

            max_r = obj.bbox[2]+buffer
            if max_r > ann_image.shape[0]:
                max_r = ann_image.shape[0]


            min_c = obj.bbox[1]-buffer
            if min_c < 0:
                min_c = 0

            max_c = obj.bbox[3]+buffer
            if max_c > ann_image.shape[1]:
                max_c = ann_image.shape[1]

            patch = tissue[min_r:max_r,min_c:max_c,:]
            patches.append(patch)


    return patches
