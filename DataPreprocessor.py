import numpy as np
from glob import glob
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from PIL import Image
import gc
import keras
from scipy.io import loadmat, savemat
from keras.models import model_from_json
import csv
import itertools
from keras.callbacks import History
from keras.layers.core import Activation
import os
import glob
import subprocess
import ntpath
import scipy

def convertheatmap(hm): # Resize the heatmap to be the correct output size
    hma = hm[7:595, 7:695] #[588,688]
    hmb = Image.fromarray(hma)
    hmc = hmb.resize((344, 294), Image.ANTIALIAS) #[294,344]
    hmd = np.asarray(hmc)
    hme = hmd[7:287, 7:337] #[280,330]
    hmf = Image.fromarray(hme)
    hmg = hmf.resize((165, 140), Image.ANTIALIAS) #[140,165]
    hmh = np.asarray(hmg)
    hmi = hmh[7:133, 7:158] #[126,151]
    hmj = hmi[4:122, 4:147] #[118,143]
    hmk = hmj[4:114, 4:139] #[110,135]
    return hmk

def save_array(imagetube, output_dir, folder_index, count):
    os.chdir(output_dir)
    np.save(str(folder_index) + str(count) + ".npy", imagetube)

def convert_MOD_images_to_npy(input_dir,output_dir, b_reduce_to_valid_heatmap, output_size):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir);
	
	file_number = os.popen('find ' + input_dir + ' -mindepth 1 -type d | wc -l').read()

	for im_index in range(int(file_number)):
	    folder_index = im_index + 1
	    print "Folder Number: " + str(folder_index)
	    frame1 = input_dir + str(folder_index) + "/_01_01.png"
	    frame2 = input_dir + str(folder_index) + "/_01_02.png"
	    frame3 = input_dir + str(folder_index) + "/_02_01.png"
	    frame4 = input_dir + str(folder_index) + "/_02_02.png"
	    s1 = Image.open(frame1).resize(output_size[1:3], Image.ANTIALIAS)
	    s2 = Image.open(frame2).resize(output_size[1:3], Image.ANTIALIAS)
	    s3 = Image.open(frame3).resize(output_size[1:3], Image.ANTIALIAS)
	    s4 = Image.open(frame4).resize(output_size[1:3], Image.ANTIALIAS)
	    
	    if b_reduce_to_valid_heatmap:
	    	s1a = convertheatmap(np.asarray(s1))
	    	s2a = convertheatmap(np.asarray(s2))
	    	s3a = convertheatmap(np.asarray(s3))
	    	s4a = convertheatmap(np.asarray(s4))
	    else:
	    	s1a = np.asarray(s1)
	    	s2a = np.asarray(s2)
	    	s3a = np.asarray(s3)
	    	s4a = np.asarray(s4)
	
	    s1a = s1a[:,:,0]
	    s2a = s2a[:,:,0]
	    s3a = s3a[:,:,0]
	    s4a = s4a[:,:,0]

	    if s1:
	       del s1#.close()
	    if s2:
	       del s2#.close(); 
	    if s3:
		del s3#.close(); 
	    if s4:
		del s4#.close();
	    save_array(s1a, output_dir, folder_index, 1)
	    save_array(s2a, output_dir, folder_index, 2)
	    save_array(s3a, output_dir, folder_index, 3)
	    save_array(s4a, output_dir, folder_index, 4)

def convert_CC_images_to_npy(input_dir,output_dir, b_is_heatmap,output_size):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir);
	print input_dir
	image_paths = glob.glob(input_dir + "/*")
	image_index = 0
	for image_path in image_paths:
	    image_index = image_index + 1
	    print "Image Number: " + str(image_index)
	    im = Image.open(image_path)
    	    im_output = np.asarray(im.resize(output_size[1:3], Image.ANTIALIAS))
	    if im:
	       del im
	    os.chdir(output_dir)
	    img_file_name = ntpath.basename(image_path)
            np.save(img_file_name + ".npy", im_output)

def generate_data_CSV(data_dir, CSV_conf_path):
	with open(CSV_conf_path, 'wb') as fp:
    	    a = csv.writer(fp, dialect='excel')
	    data_tiles = glob.glob(data_dir + "/*")
	    for data in data_tiles:
		data_file_name = ntpath.basename(data)
		line = [data_file_name]
		a.writerow(line)

def generate_CSV(image_npy_array_dir, heatmap_npy_array_dir, CSV_conf_path):
	with open(CSV_conf_path, 'wb') as fp:
    	    a = csv.writer(fp, dialect='excel')
	    
	    image_tiles = glob.glob(image_npy_array_dir + "/*")
	    
	    heatmap_tiles = glob.glob(heatmap_npy_array_dir + "/*")
	
	    for img, hm in zip(image_tiles, heatmap_tiles):
		img_file_name = ntpath.basename(img)
		hm_file_name = ntpath.basename(hm)
		line = [img_file_name, hm_file_name]
		a.writerow(line)

def slice_images(heatmap_sliced_dir, original_heatmap_dir):
	file_number = os.popen('find ' + original_heatmap_dir + ' -mindepth 1 -type f | wc -l').read()
	for num in range(int(file_number)):
	    slice_directory = heatmap_sliced_dir + str(num+1)
	    if not os.path.exists(slice_directory):
		os.makedirs(slice_directory)
	    tiles = image_slicer.slice(original_heatmap_dir+ str(num+1) + ".png", 4, save=False)
	    image_slicer.save_tiles(tiles, directory=slice_directory)
	    num += 1
