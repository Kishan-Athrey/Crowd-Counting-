from keras.models import Model
from keras.layers import Input, concatenate, ZeroPadding2D, add, SeparableConv2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import keras.backend as K
import tensorflow as tf

from custom_layers import Scale

def DenseNet(nb_dense_block=3, growth_rate=20, nb_filter=32, reduction=0.0, dropout_rate=0.0, weight_decay=0, classes=1000, weights_path=None):
    '''Instantiate the DenseNet 121 architecture,
        # Arguments
            nb_dense_block: number of dense blocks to add to end
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters
            reduction: reduction factor of transition blocks.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            classes: optional number of classes to classify images
            weights_path: path to pre-trained weights
        # Returns
            A Keras model instance.
    '''
    with tf.device('/gpu:0'):
	    eps = 1.1e-5

	    # compute compression factor
	    compression = 1.0 - reduction

	    # Handle Dimension Ordering for different backends
	    global concat_axis
	    if K.image_dim_ordering() == 'tf':
	      concat_axis = 3
	      img_input = Input(shape=(512 ,512, 3), name='data')
	    else:
	      concat_axis = 1
	      img_input = Input(shape=(3, 224, 224), name='data')

	    # From architecture for ImageNet (Table 1 in the paper)
	    nb_filter = 32 ##make this 16
	    nb_layers = [3,6,12,8]#[6,12,24,16] # For DenseNet-121

	    # Initial convolution
	    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
	    x = Conv2D(nb_filter, (7, 7) ,name='conv1',  kernel_initializer='he_normal', padding = 'valid')(x)  ### Put strides of 2*2 here and padding = valid
	    x = BatchNormalization(axis=concat_axis, name='conv1_bn')(x)
	    x = Scale(axis=concat_axis, name='conv1_scale')(x)
	    x = Activation('relu', name='relu1')(x)
	    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
	    x = MaxPooling2D((3,3), strides=(4, 4), name='pool1')(x) ### Put strides of 2*2 and pooling  3*3
	    #x = SeparableConv2D(nb_filter,(4, 4), strides = (4,4),name='pool1',kernel_initializer='he_normal',activation = 'relu')(x)
		
	    # Add dense blocks
	    for block_idx in range(nb_dense_block - 1):
		stage = block_idx+2
		x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

		# Add transition_block
		x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
		nb_filter = int(nb_filter * compression)

	    final_stage = stage + 1
	    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

	    #x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
	    x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
	    
	    '''x1 = Conv2D(32,(3,3),dilation_rate = 6, padding = 'same', kernel_initializer = 'he_normal', name = 'dil_conv1', activation = 'relu',data_format='channels_last')(x)
	    x2 = Conv2D(32,(3,3),dilation_rate = 12, padding = 'same', kernel_initializer = 'he_normal', name = 'dil_conv2', activation = 'relu',data_format='channels_last')(x)
	    x3 = Conv2D(32,(3,3),dilation_rate = 18, padding = 'same', kernel_initializer = 'he_normal', name = 'dil_conv3', activation = 'relu',data_format='channels_last')(x)
	    x4 = Conv2D(32,(3,3),dilation_rate = 24, padding = 'same', kernel_initializer = 'he_normal', name = 'dil_conv4', activation = 'relu',data_format='channels_last')(x)
	    
	    sum_b = add([x1,x2,x3,x4],name = 'sum_b')'''
	    
	    #x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)
	    #x = Conv2D(128,(1,1),padding = 'valid',kernel_initializer='he_normal')(x)
	    x = Conv2D(1,(1,1),padding = 'valid',kernel_initializer='he_normal')(x)

	    model = Model(img_input, x, name='densenet')
            '''xp = model.get_layer(name = 'conv5_blk_scale').output
 	    x1 = Conv2D(16,(3,3),dilation_rate = 6, padding = 'same', kernel_initializer = 'he_normal', name = 'dil_conv1', activation = 'relu',data_format='channels_last')(xp)
	    x2 = Conv2D(16,(3,3),dilation_rate = 12, padding = 'same', kernel_initializer = 'he_normal', name = 'dil_conv2', activation = 'relu',data_format='channels_last')(xp)
	    x3 = Conv2D(16,(3,3),dilation_rate = 18, padding = 'same', kernel_initializer = 'he_normal', name = 'dil_conv3', activation = 'relu',data_format='channels_last')(xp)
            x4 = Conv2D(16,(3,3),dilation_rate = 24, padding = 'same', kernel_initializer = 'he_normal', name = 'dil_conv4', activation = 'relu',data_format='channels_last')(xp)    
	    sum_b = add([x1,x2,x3,x4],name = 'sum_b')
	    #sum_b_activation = Activation('relu', name='relu_sum')(sum_b)
	    out = Conv2D(1,(1,1),padding = 'valid',kernel_initializer='he_normal',activation = 'relu')(sum_b)
	    model_2 = Model(inputs = model.input, outputs = out) '''      
	    if weights_path is not None:
	      model.load_weights(weights_path)

	    return model


def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=0):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor 
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4  
    #x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
    #x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Conv2D(nb_filter, (1, 1), name=conv_name_base+'_x1',kernel_initializer='he_normal',activation = 'relu')(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    #x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    #x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = SeparableConv2D(inter_channel, (3, 3),name=conv_name_base+'_x2',kernel_initializer='he_normal',activation = 'relu')(x)
    #x = Conv2D(nb_filter, (1, 1), name=conv_name_base+'_x3',kernel_initializer='he_normal',activation = 'relu')(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=0):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout 
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage) 

    #x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    #x = Activation('relu', name=relu_name_base)(x)
    x = Conv2D(int(nb_filter * compression), (1, 1), name=conv_name_base,kernel_initializer='he_normal',activation = 'relu')(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    #x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=0, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = concatenate([concat_feat, x], axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter
    

#model = DenseNet(nb_dense_block=4, growth_rate=16, nb_filter=32, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, classes=1000, weights_path=None)
#model.summary()

