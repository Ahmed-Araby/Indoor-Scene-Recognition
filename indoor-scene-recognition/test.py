import cv2
import numpy as np
from Preprocessing import *
def disp(img):
    cv2.imshow("1" , img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

X = []
img = cv2.imread("100_1414.jpg" , -1)


X.append(img)
X.append(img)
X.append(img)



H , W , C = img.shape
X = np.array(X).reshape(-1 , H , W , C)

"""
H , W , C = img.shape
X = np.array(X).reshape(-1 , H , W , C)
# first method
a = np.mean(X[: , : , : ,  0])
b = np.mean(X[: , : , : , 1])
c = np.mean(X[: , : , : , 2])

print(a , b , c)
# second method
me = X.mean(axis=(0 , 1 , 2))
print(me)


# third method

a = np.sum(X[: , : , :  , 2]) / (H*W*X.shape[0])
print(a)
"""

"""
# this proof that my broad casting work well
mean = global_mean_per_channel(X)
Y = dataset_preprocessing(X)
disp(img , Y[0])
 
"""