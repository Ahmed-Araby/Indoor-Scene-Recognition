"""
it will generate new data set for me
from old data set after applying some preprocessing and data agumentation techniques
"""

from os import listdir
from Preprocessing import preprocessing
from DataAgumentation import Agumentation
import os
import numpy as np
import cv2

#  ********************************************  opencv load images in BGR ****************************************************************
# currently I load it and save in the same format  ->>> SO i GUESS THERE IS NO NEED FOR CHANGE
cnt1 = 0
cnt2 = 0
def get_path_list(path):
    path_list = listdir(path)
    return path_list

def check_image(img):
    # image that we will ignore is
    # too small or  gray scale , images with alpha channel
    global cnt1
    if len(img.shape) != 3 or img.shape[0] < 256 or img.shape[1] < 256 or img.shape[2] != 3:
        cnt1+=1
        return 0
    return 1

def generate_data(path):

    corrupted = ['indooPool_Inside_gif.jpg']
    # file with 10 folders
    file_list =  get_path_list(path)

    for i in range(0 , len(file_list) , 1):  # i is a label
        # specific file of the 10
        class_name = file_list[i]
        class_path = os.path.join(path , class_name)

        # images paths in 1 folder of the 10 folders
        class_path_list = get_path_list(class_path)

        # iterate throw the images of 1 class
        for j in range(0 , len(class_path_list) , 1):
            img_name = class_path_list[j]
            final_path = os.path.join(class_path ,img_name)

            # corrupted images
            if class_path_list[j] in corrupted:
                continue
            img = cv2.imread(final_path, -1)

            # bad image
            if check_image(img) == 0:
                continue

            # preprocess the img , then agument the img , then save it in the new file
            pimg = preprocessing(img)
            img_l = Agumentation(pimg)
            label = i
            # save the image
            save_new_imgs(img_l , img_name , class_name)
    return

def save_new_imgs(img_l , img_name , class_name):
    save_PATH = "train2"
    for img , name in img_l:
        global cnt2
        cnt2 +=1
        final_name = name + img_name  # this order to preserve the extension
        class_path  = os.path.join(save_PATH , class_name)
        final_path = os.path.join(class_path , final_name)
        print("new final path " , final_path)
        cv2.imwrite(final_path , img)

    return




"""
print("start")
generate_data("train")
print("end")
print("we lost " , cnt1)
print("we have " , cnt2)
"""