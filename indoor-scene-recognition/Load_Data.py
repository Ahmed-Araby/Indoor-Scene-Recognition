"""
    - load imgs from csv file
    - put them into numpy array
    - cast
    - apply remaining preprocessing
    - shuffle the numpy arrays that have the imgs

    # final output
    and deliver it to me to be passed into the tflearn model
"""


import pandas as pd
import numpy as np
from Preprocessing import dataset_preprocessing
import cv2
from utils import shuffle

H = 227
W = 227
C = 3

def load_data(path):

    # load from csv file
    Data = pd.read_csv(path)
    # split
    img_paths = Data["image_path"]
    X = []
    Y = Data["labelid"]

    # get images
    for path in img_paths:
        img =  cv2.imread(path , -1)  # BGR
        X.append(img)

    # convert into numpy arrays
    X = np.array(X).reshape(-1, H, W, C)
    Y = np.array(Y).reshape(-1, 1)  # 0 - based labels (scaller)

    # cast the data set to keep good precision
    X = X.astype(np.float64)  # 64 give accurate prevision for mean calc and subtraction

    # apply preprocessing
    X = dataset_preprocessing(X, 1)

    # shuffle
    X , Y = shuffle(X , Y)

    # demo
    print(Y[:100])
    print("\n\n")
    print(Y[500:700])

    return X , Y


X , Y = load_data('train.csv')

print(X)
print("\n\n")
print(Y)