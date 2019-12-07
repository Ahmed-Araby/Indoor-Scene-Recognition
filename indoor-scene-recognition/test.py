import pickle
import cv2
from os import listdir
import pandas as pd




pickle_in = open("resized_images" , 'rb')
pickle_in2 = open("resized_images_labels"  , 'rb')
X = pickle.load(pickle_in)
Y = pickle.load(pickle_in2)
print(X.shape)
print(Y)
for i in range(X.shape[0]):
    cv2.imshow("img" , X[i])
    cv2.waitKey(0)


