#import theano
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, ZeroPadding2D,Flatten, add
from keras.models import Sequential, Model
from keras.layers.advanced_activations import PReLU, ELU
from PIL import Image
from scipy.io import loadmat, savemat
import scipy.ndimage
from keras.models import model_from_json
import csv
import itertools
from keras.callbacks import History
from keras.layers.core import Activation
import os
#os.environ["THEANO_FLAGS"] = "blas.ldflags=-lblas -lgfortran"
from keras.optimizers import Adam
from keras.optimizers import SGD
#from sklearn.cross_validation import train_test_split
from datetime import datetime
from keras.layers.normalization import BatchNormalization
import DataPreprocessor
#import DataPostprocessor
import DeepModels
import random
import glob
import ntpath
import math
import scipy.misc
import scipy.ndimage.interpolation as spyi
import tensorflow as tf

def load_model(json_path):
    model = model_from_json(open(json_path).read())
    return model

def load_weights(model, weight_path):
    dict2 = loadmat(weight_path)
    dict = conv_dict(dict2)
    i = 0
    for layer in model.layers:
        weights = dict[str(i)]
        layer.set_weights(weights)
        i += 1
    return model

def conv_dict(dict2): # Helper function to save the model
    i = 0
    dict = {}
    for i in range(len(dict2)):
        if str(i) in dict2:
            if dict2[str(i)].shape == (0, 0):
                dict[str(i)] = dict2[str(i)]
            else:
                weights = dict2[str(i)][0]
                weights2 = []
                for weight in weights:
                    if weight.shape in [(1, x) for x in range(0, 5000)]:
                        weights2.append(weight[0])
                    else:
                        weights2.append(weight)
                dict[str(i)] = weights2
    return dict

def save_model(model, json_path, weight_path):
    json_string = model.to_json()
    open(json_path, 'w').write(json_string)
    dict = {}
    i = 0
    for layer in model.layers:
        weights = layer.get_weights()
        my_list = np.zeros(len(weights), dtype=np.object)
        my_list[:] = weights
        dict[str(i)] = my_list
        i += 1
    savemat(weight_path, dict)

def meansubtraction(data):
    X_train = data
    mean = np.mean(X_train)
    for tube in range(batch):
        for img in range(channels):
            for row in range(img_rows):
                for col in range(img_cols):
                    X_train[tube,img,row,col] = X_train[tube,img,row,col] - mean
    return X_train

def generate_arrays_from_file(image_npy_dir, heatmap_npy_dir, CSV_conf_path, input_size, output_size, exp_batch_size):
    epoch_index = 0
    while 1:
	lines = open(CSV_conf_path).readlines()
	random.shuffle(lines)
	open(CSV_conf_path, 'w').writelines(lines)
        if epoch_index != 0:
            print "Epoch Index: " + str(epoch_index)
            os.system("sort -R -o " + CSV_conf_path + " " + CSV_conf_path)
        imagebatch, hmbatch = None, None
        imagebatch, hmbatch = [], []
        f = open(CSV_conf_path,'rb')
        csv_f = csv.reader(f)
        image_count = 0
        for row in csv_f:
            if image_count == exp_batch_size:
                imagebatch = np.asarray(imagebatch)
                imagebatch = imagebatch.reshape(imagebatch.shape[0], input_size[0], input_size[1], input_size[2]) #batch_size x 1 x Nrow x Ncol
                hmbatch = np.asarray(hmbatch)
                hmbatch = hmbatch.reshape(hmbatch.shape[0], output_size[0], output_size[1], output_size[2])
                imagebatch = imagebatch.astype(np.float32)        
                imagebatch /= 255
                hmbatch = hmbatch.astype(np.float32)
                hmbatch /= 255
                image_count = 0
                yield imagebatch, hmbatch
                imagebatch, hmbatch = None, None
                imagebatch, hmbatch = [], []            
            image = np.load(image_npy_dir + "" + row[0])
	    if len(image.shape)==2 and input_size[0] >1:
		image_tmp = np.zeros((image.shape[0],image.shape[1],input_size[0]))
		for channel_index in range(input_size[0]):
			image_tmp[:,:,channel_index] = image
		image = image_tmp
            hm = np.load(heatmap_npy_dir + "" + row[1])
            imagebatch.append(image)
            hmbatch.append(hm)
            image_count += 1
        epoch_index += 1
	f.close()


def parse_image_to_scaled_sub_images(im):
	image = np.copy(im)
	print im.shape
	image_shape_norm = (512,512,3) # Change it to three (third axis)
	images = []
	image_confs = []
	scale = 0
	(images_scaled,image_confs_scaled) = parse_image_to_sub_images(image,image_shape_norm,scale)		
	if len(images)==0:
		images = images_scaled
		image_confs = image_confs_scaled
	else:
		images = np.concatenate((images,images_scaled),axis=0)
		image_confs += image_confs_scaled
	return (images,image_confs)

def parse_image_to_sub_images(image,image_shape_norm,scale):
	image_shape = image.shape
	I = int(math.ceil(float(image_shape[0])/float(image_shape_norm[0])))
	J = int(math.ceil(float(image_shape[1])/float(image_shape_norm[1])))
	images = []
	image_confs = []
	for i in range(I):
 		for j in range(J):
			i_start = i*image_shape_norm[0]
			if i<(I-1):
				i_end = (i+1)*image_shape_norm[0]-1
			else:
				i_end = image_shape[0]-1
			j_start = j*image_shape_norm[1]
			if j<(J-1):
				j_end = (j+1)*image_shape_norm[1]-1
			else:
				j_end = image_shape[1]-1
			image_sub = np.zeros(image_shape_norm)
			image_temp = np.zeros((image.shape[0], image.shape[1],3))
			if len(image.shape) < 3:
				image_temp[:,:,0] = image
				image_temp[:,:,1] = image
				image_temp[:,:,2] = image
				image = image_temp
				
			image_sub[0:i_end-i_start+1,0:j_end-j_start+1,:] = image[i_start:i_end+1,j_start:j_end+1,:] 
			image_sub = image_sub.reshape((image_shape_norm[0],image_shape_norm[1],3)) 
			print image_sub.shape
			images.append(image_sub)
			conf = (scale,i_start,i_end,j_start,j_end)
			image_confs.append(conf)
	images = np.asarray(images)
	images = images.astype(np.float32)
        images /= 255.0
	return (images,image_confs)

def parse_sub_responses_to_responses(predictions,image_confs):
	predictions *= 255.0
	scale_size_dicts = {}
	for conf in image_confs:
		scale = conf[0]
		if scale not in scale_size_dicts.keys():
			scale_size_dicts[scale]=[0,0]
		if conf[2]>scale_size_dicts[scale][0]:
				scale_size_dicts[scale][0] = conf[2]+1
		if conf[4]>scale_size_dicts[scale][1]:
				scale_size_dicts[scale][1] = conf[4]+1
	responses = {}
	responses_2 = {}
	for scale in scale_size_dicts.keys():
		responses[scale] = np.zeros(scale_size_dicts[scale])
		responses_2[scale] = np.zeros(scale_size_dicts[scale])
	for i in range(len(predictions)):
		scale = image_confs[i][0]
		i_start = image_confs[i][1]
		i_end = image_confs[i][2]
		j_start = image_confs[i][3]
		j_end = image_confs[i][4]
		prediction = predictions[i]
		prediction = prediction.reshape((prediction.shape[0],prediction.shape[1]))# made changes here, change the numbers
		
		prediction = scipy.ndimage.zoom(prediction,4,order=0)
		
		print prediction.shape
		responses[scale][i_start:i_end+1,j_start:j_end+1] = prediction[0:i_end-i_start+1,0:j_end-j_start+1]
	return responses

def CC_Test_scales():
	# Parameters
	stage = 'Test'
	input_dir = '/Test/'
	output_root_dir = 'output/' 
	output_dir = output_root_dir + "heatmap_predictions"+stage+"/"
	model_path = output_root_dir + 'model.json'
	weights_path = output_root_dir + 'weights_280000.mat'	
	if not os.path.exists(output_dir):
		os.makedirs(output_dir);
	model = DeepModels.load_CC_ResNet()

	model.load_weights('weights.094-0.00286159.hdf5')
		
	model.compile(loss='mae', optimizer='Adam')
	image_paths = glob.glob(input_dir+ "/*.jpg")
	for image_index in range(len(image_paths)):
		print str(image_index)+"/"+str(len(image_paths))
		image_path = image_paths[image_index]
		image_name = ntpath.basename(image_path)
		print image_name
		
		image_ori = Image.open(image_path) 
		img_shape = image_ori.size	
		
		image_tmp = np.array(image_ori)
		image = np.zeros((image_tmp.shape[0],image_tmp.shape[1],3))
		image = np.copy(image_tmp)
		(images,image_confs) = parse_image_to_scaled_sub_images(image)
		input_size = images[0].shape
		print input_size
		
		batch_size = 64
		batch_number = int(math.ceil(float(images.shape[0])/batch_size))
		for batch_index in range(batch_number):
			batch_start = batch_index*batch_size
			if batch_index<(batch_number-1):
				batch_end = (batch_index+1)*batch_size-1
			else:
				batch_end = images.shape[0]-1
			images_batch = images[batch_start:batch_end+1,:,:,:]
			predictions_batch = model.predict_on_batch(images_batch)
			if batch_index==0:
				predictions = predictions_batch
			else:
				predictions = np.concatenate((predictions,predictions_batch),axis=0)
		responses = parse_sub_responses_to_responses(predictions,image_confs)	
		
		for scale in responses.keys():
			
			output_subdir = output_dir + os.path.basename(image_name)
			if not os.path.exists(output_subdir):
				os.makedirs(output_subdir)
			print "Max value is: " + str(np.max(responses[scale]))
			print "Min value is: " + str(np.min(responses[scale]))
			result_mat = scipy.misc.toimage(responses[scale],high = np.max(responses[scale]),low = np.min(responses[scale]), mode = 'F')
			result_mat.save(output_subdir+"/scale_"+str(scale)+".tiff")
		image_ori.save(output_subdir+"/image.png")
	return

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    session = tf.Session(config=config)
    CC_Test_scales()
