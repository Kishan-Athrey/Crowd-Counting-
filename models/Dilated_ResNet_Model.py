from keras.layers.advanced_activations import PReLU, ELU
from keras.models import Model
from keras.layers import Input,Activation,merge, Dense,Flatten,add,concatenate
from keras.layers.convolutional import Conv2D,SeparableConv2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras import regularizers
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Lambda

l2 = 0.0

def _conv_bn_relu(nb_filter, nb_row, nb_col, dilation,subsample=(1, 1)):
    def f(input):
        conv = Conv2D(nb_filter, (nb_row,nb_col),kernel_initializer="he_normal", padding="same",dilation_rate = dilation, kernel_regularizer = regularizers.l2(l2))(input)
        norm = BatchNormalization(axis = 3)(conv)
        return Activation("relu")(norm)
    return f
    
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
    
def residual_units(nb_filters, dilation,init_subsample=(1, 1)):
    def f(input):
        conv_3_3 = _conv_bn_relu(nb_filters, 3, 3, dilation,subsample=init_subsample)(input)
        residual = _conv_bn_relu(nb_filters, 3, 3, dilation)(conv_3_3)   
        return _shortcut(input, residual)
    return f
  
  
def _bn_relu_conv(nb_filter, nb_row, nb_col, dilation,subsample=(1, 1)):
    def f(input):
        norm = input
        activation = Activation("relu")(norm)
        #activation = PReLU()(norm)
        return Conv2D(nb_filter,(nb_row,nb_col),kernel_initializer="he_normal", padding="same",dilation_rate = dilation, kernel_regularizer= regularizers.l2(l2))(activation)

    return f 
    
def residual_units_2(nb_filters, dilation,init_subsample=(1, 1)):
    def f(input):
        conv_1_1 = _bn_relu_conv(nb_filters, 1, 1, dilation,subsample=init_subsample)(input)
        conv_3_3 = _bn_relu_conv(nb_filters, 3, 3, dilation)(conv_1_1)
        residual = _bn_relu_conv(nb_filters * 2, 1, 1, dilation)(conv_3_3)   
        return _shortcut(input, residual)
    return f
    
def resize_like(input_tensor,H,W): # resizes input tensor wrt. ref_tensor
    #H, W = ref_tensor.get_shape()[1], ref.get_shape()[2]
    return tf.image.resize_images(input_tensor, [H,W])
    
def CC_multiscale_Resnet(input_shape):
	with tf.device('/gpu:0'):
		ip = Input(shape=input_shape)		
		conv9_9 = Conv2D(32,(3,3), dilation_rate = 4, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal',kernel_regularizer= regularizers.l2(l2))(ip)	
		conv3_3 = Conv2D(32,(3,3), dilation_rate = 1, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal',kernel_regularizer= regularizers.l2(l2))(conv9_9)
		conv5_5 = Conv2D(32,(3,3), dilation_rate = 2, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal',kernel_regularizer= regularizers.l2(l2))(conv9_9)
		conv7_7 = Conv2D(32,(3,3), dilation_rate = 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal',kernel_regularizer= regularizers.l2(l2))(conv9_9)	
		concat_1 = concatenate([conv3_3,conv5_5,conv7_7],axis = 3)
		dpt_conv = SeparableConv2D(128,(3,3),padding = 'same',activation = 'relu',kernel_initializer = 'he_normal')(concat_1)
		#conv_1 = _conv_bn_relu(64, 3, 3, subsample=(1, 1),dilation = 1)(dpt_conv) # tap here
		max_pool_1 = MaxPooling2D(pool_size=(2,2),strides =(2,2))(dpt_conv)
		res_blk_1 = residual_units_2(32,init_subsample=(1, 1),dilation = 1)(max_pool_1)
		res_blk_2 = residual_units_2(32,init_subsample=(1, 1),dilation = 1)(res_blk_1)
		res_blk_3 = residual_units_2(32,init_subsample=(1, 1),dilation = 1)(res_blk_2)
		res_blk_4 = residual_units_2(32,init_subsample=(1, 1),dilation = 1)(res_blk_3)
		#res_blk_5 = residual_units_2(32,init_subsample=(1, 1),dilation = 1)(res_blk_4) # tap here
		max_pool_2 = MaxPooling2D(pool_size=(2,2),strides = (2,2))(res_blk_4)
		res_blk_6 = residual_units_2(64,init_subsample=(1, 1),dilation = 2)(max_pool_2)
		res_blk_7 = residual_units_2(64,init_subsample=(1, 1),dilation = 2)(res_blk_6)
		res_blk_8 = residual_units_2(64,init_subsample=(1, 1),dilation = 2)(res_blk_7)
		res_blk_9 = residual_units_2(64,init_subsample=(1, 1),dilation = 2)(res_blk_8)
		res_blk_10 = residual_units_2(64,init_subsample=(1, 1),dilation = 2)(res_blk_9)
		#conv_2 = Conv2D(64,(3,3),padding = 'same', kernel_initializer = 'he_normal',activation = 'relu')(res_blk_10)
		#conv_2 = _conv_bn_relu(64, 3, 3, subsample=(1, 1),dilation = 1)()
		dm_out = Conv2D(1,(1,1),kernel_initializer = 'he_normal')(res_blk_10)
		# Density map estimation
		#dm_upsample_1 = tf.image.resize_images(res_blk_8,[150,150])
		#dm_upsample_1 = Lambda(resize_like,arguments={'H':150,'W':150},name = 'Lambda_1')(res_blk_8)
		#dm_1_conv_1_1 = Conv2D(128,(1,1),padding = 'valid', kernel_initializer = 'he_normal')(res_blk_4)
		#dm_add_1 = add([dm_upsample_1,dm_1_conv_1_1])
		#dm_res_blk_1 = residual_units_2(64,init_subsample=(1, 1),dilation = 1)(dm_add_1)
		#dm_upsample_2 = tf.image.resize_images(dm_res_blk_1,[300,300])
		#dm_upsample_2 = Lambda(resize_like,arguments={'H':300,'W':300}, name = 'Lambda_2')(dm_res_blk_1)
		#dm_2_conv_1_1 = Conv2D(128,(1,1),padding = 'valid', kernel_initializer = 'he_normal')(conv_1)
		#dm_add_2 = add([dm_upsample_2,dm_2_conv_1_1])
		#dm_res_blk_2 = residual_units_2(32,init_subsample=(1, 1),dilation = 1)(dm_add_2)
		#dm_out = Conv2D(1,(1,1),kernel_initializer = 'he_normal')(dm_res_blk_2)
		
		model = Model(inputs = [ip],outputs = [dm_out])
		return model
		
def CC_multiscale_Resnet_2(input_shape):
	with tf.device('/gpu:0'):
		ip = Input(shape=input_shape)		
		conv3_3 = Conv2D(32,(3,3), dilation_rate = 1, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal',kernel_regularizer= regularizers.l2(l2))(ip)
		conv5_5 = Conv2D(32,(3,3), dilation_rate = 2, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal',kernel_regularizer= regularizers.l2(l2))(ip)
		conv7_7 = Conv2D(32,(3,3), dilation_rate = 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal',kernel_regularizer= regularizers.l2(l2))(ip)
		conv9_9 = Conv2D(32,(3,3), dilation_rate = 4, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal',kernel_regularizer= regularizers.l2(l2))(ip)		
		concat_1 = concatenate([conv3_3,conv5_5,conv7_7,conv9_9],axis = 3)
		dpt_conv = SeparableConv2D(128,(3,3),padding = 'same',activation = 'relu',kernel_initializer = 'he_normal')(concat_1)
		conv_1 = _conv_bn_relu(32, 3, 3, subsample=(1, 1),dilation = 1)(dpt_conv) # tap here
		max_pool_1 = MaxPooling2D(pool_size=(2,2),strides =(2,2))(conv_1)
		res_blk_1 = residual_units(32,init_subsample=(1, 1),dilation = 1)(max_pool_1)
		res_blk_2 = residual_units(32,init_subsample=(1, 1),dilation = 1)(res_blk_1)
		#res_blk_3 = residual_units(64,init_subsample=(1, 1),dilation = 1)(res_blk_2)
		res_blk_4 = residual_units(32,init_subsample=(1, 1),dilation = 1)(res_blk_2) # tap here
		max_pool_2 = MaxPooling2D(pool_size=(2,2),strides = (2,2))(res_blk_4)
		res_blk_5 = residual_units(64,init_subsample=(1, 1),dilation = 2)(max_pool_2)
		res_blk_6 = residual_units(64,init_subsample=(1, 1),dilation = 2)(res_blk_5)
		res_blk_7 = residual_units(64,init_subsample=(1, 1),dilation = 2)(res_blk_6)
		res_blk_8 = residual_units(64,init_subsample=(1, 1),dilation = 2)(res_blk_7)
		
		# Density map estimation
		#dm_upsample_1 = tf.image.resize_images(res_blk_8,[150,150])
		dm_upsample_1 = Lambda(resize_like,arguments={'H':150,'W':150},name = 'Lambda_1')(res_blk_8)
		dm_1_conv_1_1 = Conv2D(64,(1,1),padding = 'valid', kernel_initializer = 'he_normal')(res_blk_4)
		dm_add_1 = add([dm_upsample_1,dm_1_conv_1_1])
		dm_res_blk_1 = residual_units(64,init_subsample=(1, 1),dilation = 1)(dm_add_1)
		#dm_upsample_2 = tf.image.resize_images(dm_res_blk_1,[300,300])
		dm_upsample_2 = Lambda(resize_like,arguments={'H':300,'W':300}, name = 'Lambda_2')(dm_res_blk_1)
		dm_2_conv_1_1 = Conv2D(64,(1,1),padding = 'valid', kernel_initializer = 'he_normal')(conv_1)
		dm_add_2 = add([dm_upsample_2,dm_2_conv_1_1])
		dm_out = Conv2D(1,(1,1),kernel_initializer = 'he_normal')(dm_add_2)
		
		model = Model(inputs = [ip],outputs = [dm_out])
		return model		
		
		
		
		
		
		
		
		
		

