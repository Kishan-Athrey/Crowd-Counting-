
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, ZeroPadding2D,Flatten, Input,add, Dense, Lambda
from keras.layers.advanced_activations import PReLU, ELU
from PIL import Image
from scipy.io import loadmat, savemat
from keras.models import model_from_json
import csv
import itertools
from keras.callbacks import History, TensorBoard, ModelCheckpoint, LambdaCallback, LearningRateScheduler
from keras.layers.core import Activation
import os
from keras.optimizers import Adam, RMSprop
from keras.optimizers import SGD
from datetime import datetime
from keras.layers.normalization import BatchNormalization
import DataPreprocessor
import DeepModels
import random
import glob
import ntpath
import scipy.misc as misc
import scipy.ndimage.interpolation as spyi
from keras.models import load_model
import keras.backend as K
from keras.callbacks import ReduceLROnPlateau
import scipy.io
import scipy
from keras.utils import plot_model
import tensorflow as tf
import pydot
import graphviz

def meansubtraction(data):
    X_train = data
    mean = np.mean(X_train)
    for tube in range(batch):
        for img in range(channels):
            for row in range(img_rows):
                for col in range(img_cols):
                    X_train[tube,img,row,col] = X_train[tube,img,row,col] - mean
    return X_train

def get_batch(image_dir, heatmap_dir, CSV_conf_path, input_size, output_size, exp_batch_size,npy_presave):
	exp_batch_size-=2
	if npy_presave:
		return generate_arrays_from_npy(image_dir, heatmap_dir, CSV_conf_path, input_size, output_size, exp_batch_size)
	else:
		return generate_arrays_from_images(image_dir, heatmap_dir, CSV_conf_path, input_size, output_size, exp_batch_size)

def generate_arrays_from_images(image_dir, heatmap_dir, CSV_conf_path, input_size, output_size, exp_batch_size):
    epoch_index = 0
    while 1:
	lines = open(CSV_conf_path).readlines()
	random.shuffle(lines)
	open(CSV_conf_path, 'w').writelines(lines)

        if epoch_index != 0:
            print "Epoch Index: " + str(epoch_index)
            os.system("sort -R -o " + CSV_conf_path + " " + CSV_conf_path)
        image_batch, heatmap_batch = [], []
        f = open(CSV_conf_path,'rb')
        csv_f = csv.reader(f)
        image_count = 0
        for row in csv_f:
            if image_count == exp_batch_size:
                image_batch = np.array(image_batch)
                image_batch = image_batch.reshape(image_batch.shape[0], input_size[0], input_size[1], input_size[2]) #batch_size x 1 x Nrow x Ncol
                heatmap_batch = np.asarray(heatmap_batch)
                heatmap_batch = heatmap_batch.reshape(heatmap_batch.shape[0], output_size[0], output_size[1], output_size[2])
                image_batch = image_batch.astype(np.float32)
                image_batch /= 255.0
                heatmap_batch = heatmap_batch.astype(np.float32)
            
                heatmap_batch = heatmap_batch	#for single scale /160.0
                image_count = 0
                yield image_batch, heatmap_batch
                image_batch, heatmap_batch = [], []  
                 
	    image = Image.open(image_dir + "" + row[0])
	 
	    image = np.array(image)
	 
	    dirs = row[0]
	    
	    heatmap_file = scipy.io.loadmat(heatmap_dir + "" + dirs[:-4] + '.mat')
	    heatmap = heatmap_file['dm_patch']/160.0
	    original_sum = np.sum(heatmap)
    	    heatmap = spyi.zoom(heatmap,0.25)
    	    heat_rz_sum = np.sum(heatmap)
    	    if (original_sum or heat_rz_sum) == 0:
    	    	heatmap = heatmap*0.0
    	    else:
    	    	heatmap = heatmap*(original_sum/heat_rz_sum)  	
    	    
    	    heatmap = heatmap.clip(min = 0)
    	    heatmap = heatmap/16.0
            image_batch.append(image)
            heatmap_batch.append(heatmap)
            image_count += 1
        epoch_index += 1
	f.close()


def generate_CSV(image_dir, heatmap_dir, CSV_conf_path):
	with open(CSV_conf_path, 'wb') as fp:
    	    a = csv.writer(fp, dialect='excel')
	    
	    image_tiles = glob.glob(image_dir + "/*")
	    
	    heatmap_tiles = glob.glob(heatmap_dir + "/*")
	    for img, hm in zip(image_tiles, heatmap_tiles):
		img_file_name = ntpath.basename(img)
		hm_file_name = ntpath.basename(hm)
		line = [img_file_name, hm_file_name]
		a.writerow(line)

		    		
def read_eval_input_dm_patches(root_dir,image_density_paths,image_size):
	image_paths = image_density_paths[1000:5000]
	print len(image_paths)
	image_array = np.zeros((len(image_paths),image_size[0],image_size[1],3)) # change size here (3)
	density_array = np.zeros((len(image_paths),128,128,1))
	print "Reading validation images"
	for i in range(len(image_paths)):
		
		
		image_path = image_paths[i]
		dm_path = image_path.replace("validation_image_patches_512","validation_dm_patches_512")
		dm_path = dm_path[:-4] + ".mat"
		
		dm_mat = scipy.io.loadmat(dm_path)
		dm = dm_mat['dm_patch']
		
		original_count = np.sum(dm)
		
		dm_resized = spyi.zoom(dm,0.25)
		resized_count = np.sum(dm_resized)
		
		if (original_count or resized_count)== 0:
			dm_resized = dm_resized*0.0
		else:
			dm_resized = dm_resized*(original_count/resized_count)
		dm_scaled = dm_resized.clip(min=0)
		dm_scaled = dm_scaled/16.0
		
		image = Image.open(image_path)
		
		image_array[i,:,:,:] = np.array(image)/255.0 
		density_array[i,:,:,0] = dm_scaled
	print "Max value of dm is: " + str(np.max(density_array))
	
	return image_array,density_array		
			 
	
def output_images(images,output_dir, image_names=[]):
	
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	os.chdir(output_dir)
	
	image_index = 0
	
	for i in range(images.shape[0]):
		image_index += 1
		image_current = images[i,:,:,:]
		image_current = image_current.reshape(image_current.shape[0],image_current.shape[1],image_current.shape[2])
		image_current = image_current[:,:,0]
		image_current = (image_current - np.min(image_current))/(np.max(image_current) - np.min(image_current))
		image_current = image_current*255.0
		image_current = image_current.astype(np.uint8)
		img = Image.fromarray(image_current)
		if image_names == []:
		    	img.save(output_dir + str(image_index) + ".png")
		else:
		    	print output_dir + os.path.basename(image_names[i])
		    	
		    	img.save(output_dir + os.path.basename(image_names[i]))	
	

def CC_train():
	# Parameters
	exp_batch_size = 14
	input_size = (512,512,3)
	output_size = (128,128,1)
	
	npy_presave = False
	root_dir = 'train/'
	image_dir = root_dir + "train_image_patches/"
	image_npy_dir = root_dir + "images_npy/"
	heatmap_dir = root_dir + "train_dm_patches/"
	heatmap_npy_dir = root_dir + "heatmaps_npy/"
	CSV_conf_path = root_dir + "data_conf_.csv"
	output_dir = root_dir + "output/"
	model_path = output_dir + 'model.json'
	weights_path = output_dir + 'weights_205000.mat'
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	
    	# Data preprocessing: 
	if npy_presave:
		DataPreprocessor.convert_CC_images_to_npy(image_dir,image_npy_dir, False,input_size)
		DataPreprocessor.convert_CC_images_to_npy(heatmap_dir,heatmap_npy_dir, True,output_size)
		DataPreprocessor.generate_CSV(image_npy_dir, heatmap_npy_dir, CSV_conf_path)	
		data_dir = image_npy_dir
		GT_dir = heatmap_npy_dir	
	else:
		DataPreprocessor.generate_CSV(image_dir, heatmap_dir, CSV_conf_path)
		
		data_dir = image_dir
		GT_dir = heatmap_dir
		
	# Build the model
	session = tf.Session(config = config)
	model = DeepModels.load_CC_ResNet()
	model.load_weights('weights.093-0.00284886.hdf5')
	#model.load_weights('weights.062-0.00326376.hdf5')
	model.summary()

	validate_set_dir = "val/"
	image_paths = glob.glob(validate_set_dir + "validation_image_patches_512/" + "/*.png")	
	im_eval,dm_eval = read_eval_input_dm_patches(validate_set_dir,image_paths,input_size)
	
	def testmodel(epoch, logs):
    		predx = im_eval[0:32,:,:,:]#get_batch(data_dir, GT_dir, CSV_conf_path, input_size, output_size, exp_batch_size,npy_presave)
    		predy = dm_eval[0:32,:,:,:]
    		predout = model.predict(predx,batch_size= 32)
    		output_images(predx,output_dir+"/images/")
		output_images(predy,output_dir+"/heatmaps/")		
		output_images(predout,output_dir+"/heatmap_predictions/")
		
    		
	def smoothL1(y_true, y_pred):
		HUBER_DELTA = 0.05
    		x = K.abs(y_true - y_pred)
        	x = tf.where(x < HUBER_DELTA, 0.5 * (x ** 2), HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
        	return  K.mean(x,axis = -1)
        
        def learning_rate_changer(epoch_index):
        	if epoch_index != 0 and epoch_index % 10 == 0:
        		print "Changing learning rate"
        		return float(K.set_value(adma.lr,0.9*K.get_value(adam.lr))) 

	adam = Adam(lr=0.000001,beta_1 = 0.9,beta_2 = 0.999, epsilon = 1e-08, decay = 0)
	
	model.compile(loss= 'mae', optimizer= adam, metrics = ['mse'])
	
	tensorboard = TensorBoard(log_dir='./logs_densenet_finetune', histogram_freq=0, batch_size= 16, write_graph=True)
	check_point = ModelCheckpoint("weights.{epoch:03d}-{val_loss:.8f}.hdf5",verbose=1)
	model.fit_generator(get_batch(data_dir, GT_dir, CSV_conf_path, input_size, output_size, exp_batch_size,npy_presave), steps_per_epoch = 500 ,epochs= 1000, verbose=1, callbacks= [tensorboard,check_point] ,validation_data = (im_eval,dm_eval),initial_epoch = 63)
	return

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    CC_train()
