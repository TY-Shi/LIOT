import torch
import tifffile
from PIL import Image
import numpy as np
import os
import cv2
import numba
import time
import math

Img_dir = "./"+ "DRIVE_RGBimage/"
Save_dir  ="./"+ "DRIVE_liot/"
if not os.path.exists(Save_dir):
    os.makedirs(Save_dir)

files = os.listdir(Img_dir)

@numba.jit()
def LIOT_example(img):
	'''
	This funtion is a simple example but not efficient.
	'''
	#padding 8
	img = np.asarray(img)
	gray_img= img[:,:,1]#convert to gray; if not retinal dataset, you can use standard grayscale api

	pad_img = np.pad(gray_img, ((8,8)), 'constant')
	original_gray = img
	Weight = pad_img.shape[0]
	Height = pad_img.shape[1]
	Output_array = np.zeros((original_gray.shape[0],original_gray.shape[1],4)).astype(np.uint8)#four_direction_img
	mult= np.array([1, 2, 4, 8, 16, 32, 64,128])

	for w in range(8,Weight-8):
		for h in range(8,Height-8):
			orgin_value =np.array([1,1,1,1,1,1,1,1])*pad_img[w,h]
			Right_binary_code = orgin_value-pad_img[w+1:w+9,h]
			Right_binary_code[Right_binary_code>0] = 1
			Right_binary_code[Right_binary_code<=0] = 0

			Left_binary_code = orgin_value-pad_img[w-8:w,h]
			Left_binary_code[Left_binary_code>0] = 1
			Left_binary_code[Left_binary_code<=0] = 0

			Up_binary_code = orgin_value-pad_img[w,h+1:h+9].T
			Up_binary_code[Up_binary_code>0] = 1
			Up_binary_code[Up_binary_code<=0] = 0

			Down_binary_code = orgin_value-pad_img[w,h-8:h].T
			Down_binary_code[Down_binary_code>0] = 1
			Down_binary_code[Down_binary_code<=0] = 0

			Sum_Right = np.sum(mult*Right_binary_code,0)
			Sum_Left = np.sum(mult*Left_binary_code,0)
			Sum_Up = np.sum(mult*Up_binary_code,0)
			Sum_Down = np.sum(mult*Down_binary_code,0)

			Output_array[w-8,h-8][0] = Sum_Right
			Output_array[w-8,h-8][1] = Sum_Left
			Output_array[w-8,h-8][2] = Sum_Up
			Output_array[w-8,h-8][3] = Sum_Down

	return Output_array


@numba.jit()
def distance_weight_binary_pattern_faster(img):
	'''
	This function is faster than LIOT_example.py;
	More efficient LIOT will be continuously updated;
	'''
	img = np.asarray(img)#input image H*W*C
	gray_img= img[:,:,1]#convert to gray; if not retinal dataset, you can use standard grayscale api
	pad_img = np.pad(gray_img, ((8,8)), 'constant')
	Weight = pad_img.shape[0]
	Height = pad_img.shape[1]
	sum_map = np.zeros((gray_img.shape[0], gray_img.shape[1], 4)).astype(np.uint8)
	directon_map = np.zeros((gray_img.shape[0], gray_img.shape[1], 8)).astype(np.uint8)

	for direction in range(0,4):
		for postion in range(0,8):
			if direction == 0:#Right
				new_pad = pad_img[postion + 9: Weight - 7 + postion, 8:-8]  # from low to high
				#new_pad = pad_img[16-postion: Weight - postion, 8:-8]  	# from high to low
			elif direction==1:#Left
				#new_pad = pad_img[7 - postion:-1 * (9 + postion), 8:-8]  	#from low to high
				new_pad = pad_img[postion:-1 * (16 - postion), 8:-8]  	  	#from high to low
			elif direction==2:#Up
				new_pad = pad_img[8:-8, postion + 9:Height - 7 + postion]  	# from low to high
				#new_pad = pad_img[8:-8, 16 - postion: Height - postion]   	#from high to low
			elif direction==3:#Down
				#new_pad = pad_img[8:-8, 7 - postion:-1 * (9 + postion)]  	# from low to high
				new_pad = pad_img[8:-8, postion:-1 * (16 - postion)]  		#from high to low

			tmp_map = gray_img.astype(np.int64) - new_pad.astype(np.int64)
			tmp_map[tmp_map > 0] = 1
			tmp_map[tmp_map <= 0] = 0
			directon_map[:,:,postion] = tmp_map * math.pow( 2, postion)
		sum_direction = np.sum(directon_map,2)
		sum_map[:,:,direction] = sum_direction

	return sum_map


for i in range(len(files)):
	image_path = Img_dir+files[i]	#jpg image
	save_path = Save_dir + files[i]
	Gray_image = Image.open(image_path)
	Deal_img = distance_weight_binary_pattern_faster(Gray_image)
	#deal_filename = save_path[:-4] + ".tiff"
	#tifffile.imsave(deal_filename,Deal_img)
