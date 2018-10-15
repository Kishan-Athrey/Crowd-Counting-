from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, ZeroPadding2D,Flatten, Input, merge, Merge, Dense, Lambda
from keras.layers.advanced_activations import PReLU, ELU
from models import ResNet
from models import Dilated_ResNet_Model
from models import SegNet
from models import DenseNet
from models import Densception, Mobile_densenet

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

def save_model(model, json_path, weight_path):
    json_string = model.to_json()
    open(json_path,'w').write(json_string)
    dict = {}
    i = 0
    for layer in model.layers:
        weights = layer.get_weights()
        my_list = np.zeros(len(weights), dtype=np.object)
        my_list[:] = weights
        dict[str(i)] = my_list
        i += 1
    savemat(weight_path, dict)

def load_CC_ResNet(input_shape=(512,512,3)):
	return DenseNet.DenseNet(nb_dense_block=4, growth_rate=16, nb_filter=32, reduction=0.5, dropout_rate=0.0, weight_decay=0, classes=1000, weights_path=None) 



