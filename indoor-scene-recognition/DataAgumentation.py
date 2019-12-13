"""
apply some agumentation techniques
"""
import numpy as np
import cv2
from Preprocessing import preprocessing

"""
def random_shift(X , inputH , inputW):
    limh = X.shape[0] - inputH
    limw = X.shape[1] - inputW

    starth = np.squeeze(np.random.randint(0 , limh+1 , 1))
    startw = np.squeeze(np.random.randint(0 , limw+1 , 1))

    endh = starth + inputH
    endw = startw + inputW

    assert (endh <= X.shape[0] and endw <= X.shape[1])

    patch = X[starth:endh , startw:endw , :]

    assert(patch.shape[0]==inputH and patch.shape[1]==inputW)

    return patch
"""

def horizontal_reflication_img(img):
    H, W, C = img.shape
    assert (H == 227 and W == 227 and C == 3)
    hflip = cv2.flip(img , 1)
    return hflip

def extract_random_portion(img):

    H , W , C = img.shape
    assert (H==256 and W==256 and C==3)

    startH = np.random.randint(low=0 , high=H - 227)
    startW = np.random.randint(low=0 , high=W - 227)
    endH = startH + 227
    endW = startW + 227

    assert (endH <= H and endW <=W)
    ex_img = img[startH:endH , startW:endW , :]
    H , W , C = ex_img.shape
    assert (H == 227 and W == 227 and C == 3)
    return ex_img

def Agumentation(img):

    # get random 227 * 227 portion of the image
    # ********************* how many should I extract !? *************************
    ex_img = extract_random_portion(img)
    Hfliped_img = horizontal_reflication_img(ex_img)
    # string is to make the image unique form the original
    img_l = [ ( ex_img  , "227"), ( Hfliped_img , "flip")]
    return img_l


def disp(img):
    cv2.imshow("1" , img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return