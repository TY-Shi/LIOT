import sys
sys.path.append('./source/')
import numpy as np
import matplotlib.image as mpimg
import os
import ext_libs.Gudhi as gdh
import pd
import histo_image as hi

imagely_copy = mpimg.imread('output_IMG_1.png')
imagely = imagely_copy.copy()
width,height = imagely.shape
imagely[width - 1, :] = 0
imagely[:, height - 1] = 0
imagely[0, :] = 0
imagely[:, 0] = 0
temp = gdh.compute_persistence_diagram(imagely, i = 1)
betti_number = len(temp)
print (betti_number)