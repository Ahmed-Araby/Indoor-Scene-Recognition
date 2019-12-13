"""
    apply some preprocessing
    some get applied during new data set generation
    and others will be applied in memory during training
"""


import pickle
import numpy as np
import cv2
from utils import *
"""
def resize_list(X , Y , inputH , inputW):

   # - resize a list of images
   # :param X:  list of original images
   # :param Y : list of the labels of the X images
   # :return: return 4d and 2d numpy arrays (input , labels) with input resized

    resized_X = np.zeros(shape=[len(X) , inputH , inputW , 3] , dtype=np.uint8)
    resized_Y = np.zeros(shape=[len(X) , 1])

    for i in range(0 , len(X) , 1):
        resized_X[i] = resize_image(X[i] , inputH , inputW)
        resized_Y[i] = Y[i]

    return resized_X , resized_Y

def subtract_mean_img(X , mean):
    X[: , : , 0] -= int(mean[0])
    X[: , : , 1] -= int(mean[1])
    X[: , : , 2] -= int(mean[2])
    return X

def subtract_mean_list(X , mean):
    for i in range(0 , X.shape[0] , 1):
        X[i] = subtract_mean_img(X[i] , mean)
    return X
"""
def global_mean_per_channel(X):
    # if the data set is very large we may do this for each patch and take
    # the mean across batches or using window mean

    # BGR
    mean= np.mean(X , axis = (0 , 1 , 2))   # it divides by (number of examples * rows * cols )
    print("mean of the whole data set is per channel " , mean)
    assert (len(mean) == 3)
    save_into_dics(mean  , "DS_local_mean")
    return mean

def normalize_img_0_1(X):
    shape = X.shape
    if len(shape) == 3:
        X = X[: , : , :] / 255
    else:
        X = X[: , :] / 255
    return X

def normalize_min_max(img):
    # nearly all imgs are have min = 0 and max = 255
    # so min max norm will make no big diff

    min =  np.min(img)
    max = np.max(img)
    img = (img - min) / (max-min)  # [0 , 1]  # preserving our histogram distribution
    return img



def resize_image(img , inputH = 256 , inputW = 256):
    H , W , _ = img.shape

    # resize the shorter side
    if W >= H:
        inputW = W
    if H >= W :
        inputH = H

    resized_img = cv2.resize(img, dsize=(inputW, inputH))

    # get the start/ end points of the center 256*256
    H , W  , _ = resized_img.shape

    midH = int(H/2)
    midW = int(W/2)

    startH = midH - 128
    endH = midH + 128   # excluded

    startW  = midW - 128
    endW = midW + 128

    # crop center 256 * 256
    croped_img = resized_img[startH:endH , startW:endW , :]

    H , W , C = croped_img.shape
    assert (H==256 and W == 256 and C== 3)
    return croped_img


def preprocessing(img):
    # prepare anther data set to be saved
    # resize  256*256
    img = resize_image(img)
    return img

def dataset_preprocessing(X , mode=1):
    # this will happen during training (in memory ) as saving will reverse the norm ,
    # centering could cuz in -ve val which will be saved in wrong way !?
    # it can accept the whole data set , or batch

    # BGR
    # centering then normalization [0 , 1]
    if mode==1:
        print("centering then normalization ")
        #  centering
        mean = global_mean_per_channel(X)
        mean = np.array(mean).reshape(1 , 1 , 1  , 3)  # to do broad casting
        X = X - mean # broad casing
        # normalization
        for i in range(0 , X.shape[0], 1):
            X[i] = normalize_min_max(X[i])


    #[ TO DO ]
    # normalization then centering
    else :
        pass

    return X

def disp(img):
    cv2.imshow("1" , img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return




