import os
from keras.layers.advanced_activations import PReLU, ELU
from keras.layers import PReLU
from keras.models import Model
from keras.layers import (Input,
    Activation,
    merge,
    Dense,
    Flatten,add,concatenate
)
from keras.layers.convolutional import (Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras.models import load_model


# Helper to build a conv -> BN -> relu block
def _conv_bn_relu(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        conv = Conv2D(nb_filter, (nb_row,nb_col),kernel_initializer="he_normal", padding="same",dilation_rate = 1)(input)
        norm = BatchNormalization(axis = 3)(conv)
        return Activation("relu")(norm)

    return f


# Helper to build a BN -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
def _bn_relu_conv(nb_filter, nb_row, nb_col, dilation,subsample=(1, 1)):
    def f(input):
        norm = input
        
        #activation = Activation("relu")(norm)
        #activation = PReLU()(norm)
        return Conv2D(nb_filter,(nb_row,nb_col),kernel_initializer="he_normal", padding="same",dilation_rate = dilation, activation = 'relu')(norm)

    return f


# Bottleneck architecture for > 34 layer resnet.
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
# Returns a final conv layer of nb_filters * 4
def _bottleneck(nb_filters, dilation,init_subsample=(1, 1)):
    def f(input):
        conv_1_1 = _bn_relu_conv(nb_filters, 1, 1, 1,subsample=init_subsample)(input)
        conv_3_3 = _bn_relu_conv(nb_filters, 3, 3,dilation)(conv_1_1)
        residual = _bn_relu_conv(nb_filters * 4, 1, 1,1)(conv_3_3)
        return _shortcut(input, residual)

    return f


# Basic 3 X 3 convolution blocks.
# Use for resnet with layers <= 34
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
def _basic_block(nb_filters, init_subsample=(1, 1)):
    def f(input):
        conv1 = _bn_relu_conv(nb_filters, 3, 3, subsample=init_subsample)(input)
        residual = _bn_relu_conv(nb_filters, 3, 3)(conv1)
        return _shortcut(input, residual)

    return f


# Adds a shortcut between input and residual block and merges them with "sum"
def _shortcut(input, residual):
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    stride_width = input._keras_shape[1] / residual._keras_shape[1] # made changes here
    stride_height = input._keras_shape[2] / residual._keras_shape[2] # made changes here
    equal_channels = residual._keras_shape[3] == input._keras_shape[3] # made changes here

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(residual._keras_shape[3], (1, 1), #made changes here change 3 to 1
                                 strides=(stride_width, stride_height),
                                 kernel_initializer="he_normal", padding="valid")(input)

    return add([shortcut, residual])


# Builds a residual block with repeating bottleneck blocks.
def _residual_block(block_function, nb_filters, repetations, is_first_layer=False):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2)
            input = block_function(nb_filters=nb_filters, init_subsample=init_subsample)(input)
        return input

    return f

# Builds a CC residual block with repeating bottleneck blocks.
def _CC_residual_block(block_function, nb_filters, repetations, dilation , is_first_layer=False):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            input = block_function(nb_filters=nb_filters, dilation = dilation ,init_subsample=init_subsample)(input)
        return input

    return f

def _bottleneck_2(nb_filters, init_subsample=(1, 1)):
    def f(input):
        conv_3_3 = _conv_bn_relu(nb_filters, 3, 3)(input)
        residual = _conv_bn_relu(nb_filters, 3, 3)(conv_3_3)
        return _shortcut(input, residual)

    return f
    
def CC_ResNet(input_shape):
    with tf.device('/gpu:0'):
	    input = Input(shape=input_shape)
	    conv1 = _conv_bn_relu(nb_filter=32, nb_row=7, nb_col=7, subsample=(1, 1))(input)
	    #conv2 = Conv2D(32,(3,3),strides = (4,4),padding = 'same', activation = 'relu',kernel_initializer="he_normal")(conv1)
	    pool1 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(conv1)
	    # Build residual blocks..
	    block_fn = _bottleneck
	    block1 = _CC_residual_block(block_fn, nb_filters=32, repetations=3, dilation = 1, is_first_layer=True)(pool1)
	    block2 = _CC_residual_block(block_fn, nb_filters=32, repetations=4, dilation = 1)(block1)
	    block3 = _CC_residual_block(block_fn, nb_filters=32, repetations=8, dilation = 1)(block2)
	    block4 = _CC_residual_block(block_fn, nb_filters=32, repetations=3, dilation = 1)(block3)
	    #block5 = _CC_residual_block(block_fn, nb_filters=64, repetations=3, dilation = 1)(block4)
	    #block6 = _CC_residual_block(block_fn, nb_filters=64, repetations=2, dilation = 1)(block5)
	    #block7 = _CC_residual_block(block_fn, nb_filters=32, repetations=6, dilation = 1)(block6) # added extra layers
	    block8 = Conv2D(1, (1, 1), padding='valid')(block4)
	    model = Model(inputs = [input], outputs = [block8])
    	    return model
#PRelu = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
def CC_athrey_net(input_shape):
	with tf.device('/gpu:0'):
		ip = Input(shape=input_shape)
		Conv1 = Conv2D(24,(11,11),padding = 'same', activation = 'relu',kernel_initializer='glorot_normal')(ip)
		Conv2 = Conv2D(32,(9,9),padding = 'same', activation = 'relu',kernel_initializer='glorot_normal')(Conv1)
		max_pool_1 = MaxPooling2D(pool_size=(2,2), strides = (2,2))(Conv2)
		Conv3 = Conv2D(48,(7,7),padding = 'same',activation = 'relu',kernel_initializer='glorot_normal')(max_pool_1)
		Conv4 = Conv2D(64,(5,5),padding = 'same',activation = 'relu', dilation_rate = (1,1),kernel_initializer='glorot_normal')(Conv3)
		max_pool_2 = MaxPooling2D(pool_size=(2,2), strides = (2,2))(Conv4)
		Conv5 = Conv2D(96,(3,3),padding = 'same', activation = 'relu',kernel_initializer='glorot_normal', dilation_rate = (1,1))(max_pool_2)
		Conv1_1_1 = Conv2D(32,(3,3), padding = 'valid', kernel_initializer = 'glorot_normal', strides = (4,4))(Conv2)
		Conv1_1_2 = Conv2D(64,(3,3), padding = 'valid', kernel_initializer = 'glorot_normal', strides = (2,2))(Conv4)	
		concat = concatenate([Conv1_1_1,Conv1_1_2,Conv5],axis = 3)
		#Conv6 = Conv2D(64,(3,3),padding = 'same', activation = 'relu',kernel_initializer='glorot_normal', dilation_rate = (1,1))(Conv5)
		Conv7 = Conv2D(1,(1,1),padding = 'valid')(concat)
		model = Model(inputs = [ip], outputs = [Conv7])
		return model

def CC_ResNet_2(input_shape):
    with tf.device('/gpu:0'):
	    input = Input(shape=input_shape)
	    conv1 = _conv_bn_relu(nb_filter=32, nb_row=11, nb_col=11, subsample=(1, 1))(input)
	    #conv2 = Conv2D(32,(3,3),strides = (4,4),padding = 'same', activation = 'relu',kernel_initializer="he_normal")(conv1)
	    pool1 = MaxPooling2D(pool_size=(4, 4), strides=(2, 2))(conv1)
	    # Build residual blocks..
	    block_fn = _bottleneck
	    block1 = _CC_residual_block(block_fn, nb_filters=16, repetations=2, dilation = 1, is_first_layer=True)(pool1)
	    block2 = _CC_residual_block(block_fn, nb_filters=32, repetations=3, dilation = 1)(block1)
	    block3 = _CC_residual_block(block_fn, nb_filters=32, repetations=4, dilation = 1)(block2)
	    #block4 = _CC_residual_block(block_fn, nb_filters=64, repetations=3, dilation = 1)(block3)
	    #block5 = _CC_residual_block(block_fn, nb_filters=32, repetations=3, dilation = 1)(block4)
	    #block6 = _CC_residual_block(block_fn, nb_filters=32, repetations=2, dilation = 1)(block5)
	    #block7 = _CC_residual_block(block_fn, nb_filters=32, repetations=6, dilation = 2)(block6) # added extra layers
	    block8 = Conv2D(1, (1, 1), padding='valid')(block3)
	    model = Model(inputs = [input], outputs = [block8])
    	    return model

def CC_MCNN(input_shape):
	with tf.device('/gpu:0'):
		ip = Input(shape=input_shape)
		
		# Column 1
		Conv1_1 = Conv2D(16,(5,5),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv1_1')(ip)
		max_pool_1_1 = MaxPooling2D(pool_size=(2,2), strides = (2,2),name='max_pool_1_1')(Conv1_1)
		Conv1_2 = Conv2D(32,(5,5),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv1_2')(max_pool_1_1)
		Conv1_3 = Conv2D(24*2,(3,3),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv1_3')(Conv1_2)
		Conv1_4 = Conv2D(24*2,(3,3),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv1_4')(Conv1_3)
		max_pool_1_2 = MaxPooling2D(pool_size=(2,2), strides = (2,2),name='max_pool_1_2')(Conv1_4)
		Conv1_5 = Conv2D(24*3,(3,3),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv1_5')(max_pool_1_2)
		Conv1_6 = Conv2D(24*4,(3,3),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv1_6')(Conv1_5)
		Conv1_7 = Conv2D(24*6,(3,3),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv1_7')(Conv1_6)
		Conv1_8 = Conv2D(1,(1,1),padding = 'same', kernel_initializer = 'glorot_uniform',name='Conv1_8')(Conv1_7)
		
		# Column 2
		Conv2_1 = Conv2D(16,(9,9),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv2_1')(ip)
		max_pool_2_1 = MaxPooling2D(pool_size=(2,2), strides = (2,2),name='max_pool_2_1')(Conv2_1)
		Conv2_2 = Conv2D(32,(9,9),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv2_2')(max_pool_2_1)
		Conv2_3 = Conv2D(24*2,(7,7),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv2_3')(Conv2_2)
		Conv2_4 = Conv2D(24*2,(7,7),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv2_4')(Conv2_3)
		max_pool_2_2 = MaxPooling2D(pool_size=(2,2), strides = (2,2,),name='max_pool_2_2')(Conv2_4)
		Conv2_5 = Conv2D(24*3,(3,3),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv2_5')(max_pool_2_2)
		Conv2_6 = Conv2D(24*4,(3,3),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv2_6')(Conv2_5)
		Conv2_7 = Conv2D(24*6,(3,3),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv2_7')(Conv2_6)
		Conv2_8 = Conv2D(1,(1,1),padding = 'same', kernel_initializer= 'glorot_uniform',name='Conv2_8')(Conv2_7)
		
		# Column 3
		Conv3_1 = Conv2D(16,(7,7),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv3_1')(ip)
		max_pool_3_1 = MaxPooling2D(pool_size=(2,2), strides = (2,2),name='max_pool_3_1')(Conv3_1)
		Conv3_2 = Conv2D(48,(5,5),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv3_2')(max_pool_3_1)
		Conv3_3 = Conv2D(48*2,(3,3),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv3_3')(Conv3_2)
		Conv3_4 = Conv2D(48*2,(3,3),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv3_4')(Conv3_3)
		max_pool_3_2 = MaxPooling2D(pool_size=(2,2), strides = (2,2),name='max_pool_3_2')(Conv3_3)
		Conv3_5 = Conv2D(48*3,(3,3),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv3_5')(max_pool_3_2)
		Conv3_6 = Conv2D(48*3,(3,3),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv3_6')(Conv3_5)
		Conv3_7 = Conv2D(48*5,(3,3),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv3_7')(Conv3_6)
		Conv3_8 = Conv2D(1,(1,1),padding = 'same', kernel_initializer= 'glorot_uniform',name='Conv3_8')(Conv3_7)
		
		concat = concatenate([Conv1_8,Conv2_8,Conv3_8],axis = 3)
		Conv4_1 = Conv2D(1,(1,1), padding = 'same', kernel_initializer='glorot_uniform')(concat)
		model = Model(inputs = [ip], outputs = [Conv4_1])
		return model

def CC_MCNN_old(input_shape):
	with tf.device('/gpu:1'):
		ip = Input(shape=input_shape)
		Conv1_1 = Conv2D(16,(11,11),padding = 'same', activation = 'relu',kernel_initializer='he_normal')(ip)
		max_pool_1_1 = MaxPooling2D(pool_size=(2,2), strides = (2,2))(Conv1_1)
		Conv1_2 = Conv2D(24,(9,9),padding = 'same', activation = 'relu',kernel_initializer='he_normal')(max_pool_1_1)
		Conv1_3 = Conv2D(24,(7,7),padding = 'same', activation = 'relu',kernel_initializer='he_normal')(Conv1_2)
		max_pool_1_2 = MaxPooling2D(pool_size=(2,2), strides = (2,2))(Conv1_3)
		Conv1_4 = Conv2D(24*2,(7,7),padding = 'same', activation = 'relu',kernel_initializer='he_normal')(max_pool_1_2)
		Conv1_5 = Conv2D(24*2,(5,5),padding = 'same', activation = 'relu',kernel_initializer='he_normal')(Conv1_4)
		
		# Column 2
		Conv2_1 = Conv2D(16,(9,9),padding = 'same', activation = 'relu',kernel_initializer='he_normal')(ip)
		max_pool_2_1 = MaxPooling2D(pool_size=(2,2), strides = (2,2))(Conv2_1)
		Conv2_2 = Conv2D(32,(7,7),padding = 'same', activation = 'relu',kernel_initializer='he_normal')(max_pool_2_1)
		Conv2_3 = Conv2D(32,(5,5),padding = 'same', activation = 'relu',kernel_initializer='he_normal')(Conv2_2)
		max_pool_2_2 = MaxPooling2D(pool_size=(2,2), strides = (2,2))(Conv2_3)
		Conv2_4 = Conv2D(32*2,(5,5),padding = 'same', activation = 'relu',kernel_initializer='he_normal')(max_pool_2_2)
		Conv2_5 = Conv2D(24*2,(3,3),padding = 'same', activation = 'relu',kernel_initializer='he_normal')(Conv2_4)
		
		# Column 3
		Conv3_1 = Conv2D(16,(7,7),padding = 'same', activation = 'relu',kernel_initializer='he_normal')(ip)
		max_pool_3_1 = MaxPooling2D(pool_size=(2,2), strides = (2,2))(Conv3_1)
		Conv3_2 = Conv2D(48,(5,5),padding = 'same', activation = 'relu',kernel_initializer='he_normal')(max_pool_3_1)
		Conv3_3 = Conv2D(48,(3,3),padding = 'same', activation = 'relu',kernel_initializer='he_normal')(Conv3_2)
		max_pool_3_2 = MaxPooling2D(pool_size=(2,2), strides = (2,2))(Conv3_3)
		Conv3_4 = Conv2D(48*2,(3,3),padding = 'same', activation = 'relu',kernel_initializer='he_normal')(max_pool_3_2)
		Conv3_5 = Conv2D(24*2,(3,3),padding = 'same', activation = 'relu',kernel_initializer='he_normal')(Conv3_4)
		
		concat = concatenate([Conv1_5,Conv2_5,Conv3_5],axis = 3)
		Conv4_1 = Conv2D(1,(1,1), padding = 'same', kernel_initializer='he_normal')(concat)
		model = Model(inputs = [ip], outputs = [Conv4_1])
		return model


def CC_MCNN_1(input_shape):
	with tf.device('/gpu:0'):
		ip = Input(shape=input_shape)
		# Column 1
		Conv1_1 = Conv2D(16,(5,5),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv1_1')(ip)
		max_pool_1_1 = MaxPooling2D(pool_size=(2,2), strides = (2,2),name='max_pool_1_1')(Conv1_1)
		Conv1_2 = Conv2D(32,(5,5),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv1_2')(max_pool_1_1)
		Conv1_3 = Conv2D(24*2,(3,3),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv1_3')(Conv1_2)
		Conv1_4 = Conv2D(24*2,(3,3),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv1_4')(Conv1_3)
		max_pool_1_2 = MaxPooling2D(pool_size=(2,2), strides = (2,2),name='max_pool_1_2')(Conv1_4)
		Conv1_5 = Conv2D(24*3,(3,3),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv1_5')(max_pool_1_2)
		Conv1_6 = Conv2D(24*4,(3,3),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv1_6')(Conv1_5)
		Conv1_7 = Conv2D(24*6,(3,3),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv1_7')(Conv1_6)
		Conv1_8 = Conv2D(1,(1,1),padding = 'same', kernel_initializer = 'glorot_uniform',name='Conv1_8')(Conv1_7)
		model =  Model(inputs = [ip], outputs = [Conv1_8])
		return model

def CC_MCNN_2(input_shape):
	with tf.device('/gpu:0'):
		ip = Input(shape=input_shape)
		# Column 2
		Conv2_1 = Conv2D(16,(9,9),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv2_1')(ip)
		max_pool_2_1 = MaxPooling2D(pool_size=(2,2), strides = (2,2),name='max_pool_2_1')(Conv2_1)
		Conv2_2 = Conv2D(32,(9,9),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv2_2')(max_pool_2_1)
		Conv2_3 = Conv2D(24*2,(7,7),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv2_3')(Conv2_2)
		Conv2_4 = Conv2D(24*2,(7,7),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv2_4')(Conv2_3)
		max_pool_2_2 = MaxPooling2D(pool_size=(2,2), strides = (2,2,),name='max_pool_2_2')(Conv2_4)
		Conv2_5 = Conv2D(24*3,(3,3),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv2_5')(max_pool_2_2)
		Conv2_6 = Conv2D(24*4,(3,3),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv2_6')(Conv2_5)
		Conv2_7 = Conv2D(24*6,(3,3),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv2_7')(Conv2_6)
		Conv2_8 = Conv2D(1,(1,1),padding = 'same', kernel_initializer= 'glorot_uniform',name='Conv2_8')(Conv2_7)
		model =  Model(inputs = [ip], outputs = [Conv2_8])
		return model

def CC_MCNN_3(input_shape):
	with tf.device('/gpu:1'):
		ip = Input(shape=input_shape)
		Conv3_1 = Conv2D(16,(7,7),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv3_1')(ip)
		max_pool_3_1 = MaxPooling2D(pool_size=(2,2), strides = (2,2),name='max_pool_3_1')(Conv3_1)
		Conv3_2 = Conv2D(48,(5,5),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv3_2')(max_pool_3_1)
		Conv3_3 = Conv2D(48*2,(3,3),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv3_3')(Conv3_2)
		Conv3_4 = Conv2D(48*2,(3,3),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv3_4')(Conv3_3)
		max_pool_3_2 = MaxPooling2D(pool_size=(2,2), strides = (2,2),name='max_pool_3_2')(Conv3_3)
		Conv3_5 = Conv2D(48*3,(3,3),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv3_5')(max_pool_3_2)
		Conv3_6 = Conv2D(48*3,(3,3),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv3_6')(Conv3_5)
		Conv3_7 = Conv2D(48*5,(3,3),padding = 'same', activation = 'relu',kernel_initializer='glorot_uniform',name='Conv3_7')(Conv3_6)
		Conv3_8 = Conv2D(1,(1,1),padding = 'same', kernel_initializer= 'glorot_uniform',name='Conv3_8')(Conv3_7)
		model =  Model(inputs = [ip], outputs = [Conv3_8])
		return model


def CC_MCNN_Combined(input_shape):
	with tf.device('/gpu:1'):
		ip = Input(shape=input_shape)
		column_1 = load_model('/home/kishan/Documents/Crowd_Counting_Project/Gauss_pred_local_context/logs_dm_pred_MCNN_net_final_dataset/MCNN_1.h5')
		column_1.load_weights('/home/kishan/Documents/Crowd_Counting_Project/Gauss_pred_local_context/logs_dm_pred_MCNN_net_final_dataset/weights.152-0.00396636MCNN_1.hdf5', by_name=True)
		column_2 = load_model('/home/kishan/Documents/Crowd_Counting_Project/Gauss_pred_local_context/logs_dm_pred_MCNN_net_final_dataset/MCNN_2.h5')
		column_2.load_weights('/home/kishan/Documents/Crowd_Counting_Project/Gauss_pred_local_context/logs_dm_pred_MCNN_net_final_dataset/weights.040-0.00461347MCNN_2.hdf5', by_name=True)
		column_3 = load_model('/home/kishan/Documents/Crowd_Counting_Project/Gauss_pred_local_context/logs_dm_pred_MCNN_net_final_dataset/MCNN_3_new.h5')
		column_3.load_weights('/home/kishan/Documents/Crowd_Counting_Project/Gauss_pred_local_context/logs_dm_pred_MCNN_net_final_dataset/weights.010-0.01028650MCNN_3.hdf5', by_name=True)
		#column_1_ip = column_1.get_layer(index = 0
		#column_2_ip =
		#column_3_ip =
		#print type(column_3)
		#raw_input('press here') 
		
		model_1 = Model(inputs = [column_1.input], outputs = [column_1.get_layer(name = 'Conv1_7').output])
		model_2 = Model(inputs = [column_2.input], outputs = [column_2.get_layer(name = 'Conv2_7').output])
		model_3 = Model(inputs = [column_3.input], outputs = [column_3.get_layer(name = 'Conv3_7').output])
		out_1 = model_1(ip)
		out_2 = model_2(ip)
		out_3 = model_3(ip)
		concat = concatenate([out_1,out_2,out_3],axis = 3)
		Conv4_1 = Conv2D(1,(1,1), padding = 'same', kernel_initializer='glorot_uniform')(concat)
		model = Model(inputs = [ip], outputs = [Conv4_1])
		return model
		 


def main():
    import time
    start = time.time()
    model = resnet()
    duration = time.time() - start
    print "{} s to make model".format(duration)

    start = time.time()
    model.output
    duration = time.time() - start
    print "{} s to get output".format(duration)

    start = time.time()
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    duration = time.time() - start
    print "{} s to get compile".format(duration)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(current_dir, "resnet_50.png")
    plot(model, to_file=model_path, show_shapes=True)


if __name__ == '__main__':
    main()

