import pickle
import numpy as np
import cv2
from Load_Data import load_data

def resize_image(X , inputH , inputW):
    resized_X = cv2.resize(X , dsize=(inputH , inputW))
    return resized_X

def resize_list(X , Y , inputH , inputW):
    """
    - resize a list of images
    :param X:  list of original images
    :param Y : list of the labels of the X images
    :return: return 4d and 2d numpy arrays (input , labels) with input resized
    """
    resized_X = np.zeros(shape=[len(X) , inputH , inputW , 3] , dtype=np.uint8)
    resized_Y = np.zeros(shape=[len(X) , 1])

    for i in range(0 , len(X) , 1):
        resized_X[i] = resize_image(X[i] , inputH , inputW)
        resized_Y[i] = Y[i]

    return resized_X , resized_Y

def PCA():
    pass

def preprocessing(X , Y ,  inputH , inputW):
    # resizeing
    resized_X = resize_list(X , Y , inputH , inputW)

    # PCA on RGB values for illumination invariance
    # *********************************
    return resized_X


def save_the_data(X , Y):
    # save the resized data
    pickle_out_X = open("resized_images", 'wb')
    pickle.dump(X, pickle_out_X)
    pickle_out_X.close()

    pickle_out_Y = open("resized_images_labels", 'wb')
    pickle.dump(Y, pickle_out_Y)
    pickle_out_Y.close()
    return

path = "/home/ahmedaraby/PycharmProjects/indoor-scene-recognition/train"

X , Y = load_data(path)
print(len(X))
print("start")
resized_X  , resized_Y= preprocessing(X , Y , 256 , 256)
save_the_data(resized_X , resized_Y)
print("pre processing the data is done")


