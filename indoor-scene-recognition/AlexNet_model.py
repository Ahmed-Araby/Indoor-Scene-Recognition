"""
my deep learning model in tensorflow (tf learn)
"""
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


# initializer
from tflearn.initializations import xavier

def Alex_net():
    conv_input = input_data(shape=[None , 227 , 227 , 3] , name='input')

    # conv layer1
    conv1 = conv_2d(conv_input, 96, filter_size=11, strides=4, activation='relu', bias=True, weights_init=xavier(), name='conv1') # padding !?
    lrn1 = local_response_normalization(conv1, name='lrn1')  # ***************************
    max_pool1 = max_pool_2d(lrn1, kernel_size=3, strides=2, name='max_pool1')  # padding !?

    # conv layer 2
    conv2 = conv_2d(max_pool1, 256, filter_size=5, activation='relu', bias=True, weights_init=xavier(), name='conv2')  # ***
    lrn2 = local_response_normalization(conv2, name='lrn2')
    max_pool2 = max_pool_2d(lrn2, kernel_size=3, strides=2, name='max_pool2')

    # conv layer 3

    conv3 = conv_2d(max_pool2, 384, filter_size=3, activation='relu', bias='True', weights_init=xavier(), name='conv3')  # *** stride !?

    # conv layer 4
    conv4 = conv_2d(conv3, 384, filter_size=3, activation='relu', bias=True, weights_init=xavier(), name='conv4')

    # conv layer 5
    conv5 = conv_2d(conv4, 256, filter_size=3, activation='relu', bias=True, weights_init=xavier(), name='conv5')

    # dense layer
    # fc1
    fc1  = fully_connected(conv5, 4096, activation='relu', bias=True, weights_init=xavier(), name='fc1')  # built in flatten
    fcd1 = dropout(fc1, 0.5, name='fcd1')  # how does it handle testing

    # fc2
    fc2 = fully_connected(fcd1, 4096, activation='relu', bias=True, weights_init=xavier(), name='fc2')
    fcd2 = dropout(fc2, 0.5, name='fcd2')

    # fc3 (output layer)
    fc3 = fully_connected(fcd2, 10, activation='softmax', bias=True, weights_init=xavier(), name='output')

    estimator = regression(fc3, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='estimator')

    model = tflearn.DNN(estimator, tensorboard_dir='alex_net_graph', tensorboard_verbose=3) # verbous !?

    return model

model = Alex_net()
print(model)
