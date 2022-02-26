import sys
import numpy as np
import matplotlib.image as mpimg
import os
sys.path.append('/home/sty/retinal_liot_topo/src/Betti_Compute/')
#sys.path.insert(0, './Betti_Compute/')
print("sys_path",sys.path)
import ext_libs.Gudhi as gdh

def betti_number(imagely):
	# imagely_copy = mpimg.imread('output_IMG_1.png')
	imagely = imagely.detach().cpu().clone().numpy()
	#print("imagely.shape",imagely.shape)
	width,height = imagely.shape
	imagely[width - 1, :] = 0
	imagely[:, height - 1] = 0
	imagely[0, :] = 0
	imagely[:, 0] = 0
	temp = gdh.compute_persistence_diagram(imagely, i = 1)
	betti_number = len(temp)
	# print (betti_number)
	return betti_number