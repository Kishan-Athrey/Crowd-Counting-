import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization

import tensorflow as tf
from keras import backend as K

import numpy as np
import json
np.random.seed(7) # 0bserver07 for reproducibility





def create_encoding_layers():
    kernel = 3
    filter_size = 32
    pad = 1
    pool_size = 2
    return [
        Conv2D(filter_size, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),
	#Dropout(0.5),
        #ZeroPadding2D(padding=(pad,pad)),
        Conv2D(filter_size*2, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),
	#Dropout(0.5),
        #ZeroPadding2D(padding=(pad,pad)),
        Conv2D(filter_size*3, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),
	#Dropout(0.5),
        #ZeroPadding2D(padding=(pad,pad)),
        Conv2D(filter_size*4, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu')
    ]

def create_decoding_layers():
    kernel = 3
    filter_size = 32
    pad = 1
    pool_size = 2
    return[
        #ZeroPadding2D(padding=(pad,pad)),
        Conv2D(filter_size*4, (kernel, kernel), padding='same'),
        BatchNormalization(),
	Activation('relu'),
        UpSampling2D(size=(pool_size,pool_size)),
        #Dropout(0.5),
        #ZeroPadding2D(padding=(pad,pad)),
        Conv2D(filter_size*3, (kernel, kernel), padding='same'),
        BatchNormalization(),
	Activation('relu'),
        UpSampling2D(size=(pool_size,pool_size)),
       	#Dropout(0.5),
        #ZeroPadding2D(padding=(pad,pad)),
        Conv2D(filter_size*2, (kernel, kernel), padding='same'),
        BatchNormalization(),
	Activation('relu'),
        UpSampling2D(size=(pool_size,pool_size)),
      	#Dropout(0.5),
        #ZeroPadding2D(padding=(pad,pad)),
        Conv2D(filter_size, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu')
    ]

def SegNet():
	with tf.device('/gpu:0'):
		segnet_basic = models.Sequential()

		segnet_basic.add(Layer(input_shape=(256,256,1)))



		segnet_basic.encoding_layers = create_encoding_layers()
		for l in segnet_basic.encoding_layers:
		    segnet_basic.add(l)

		# Note: it this looks weird, that is because of adding Each Layer using that for loop
		# instead of re-writting mode.add(somelayer+params) everytime.

		segnet_basic.decoding_layers = create_decoding_layers()
		for l in segnet_basic.decoding_layers:
		    segnet_basic.add(l)

		segnet_basic.add(Conv2D(1, (1, 1), padding='same'))
		return segnet_basic






# Save model to JSON
'''
with open('segNet_basic_model.json', 'w') as outfile:
    outfile.write(json.dumps(json.loads(segnet_basic.to_json()), indent=2))'''
