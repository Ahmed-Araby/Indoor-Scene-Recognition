from os import listdir
import os
import numpy as np
import cv2
# opencv load images in BGR
cnt1 = 0
cnt2 = 0
def get_path_list(path):
    path_list = listdir(path)
    return path_list

def load_data(path):

    corrupted = ['indooPool_Inside_gif.jpg']
    X = []
    Y = []
    global cnt1 , cnt2
    # file with 10 folders
    file_list =  get_path_list(path)

    for i in range(0 , len(file_list) , 1):
        # specific file of the 10
        class_path = os.path.join(path , file_list[i])

        # images paths in 1 folder of the 10 folders
        class_path_list = get_path_list(class_path)

        # iterate throw the images of 1 class
        for j in range(0 , len(class_path_list) , 1):
            final_path = os.path.join(class_path , class_path_list[j])

            # corrupted images
            if class_path_list[j] in corrupted:
                continue
            img = cv2.imread(final_path, -1)
            # image that we will ignore is
            # too small or  gray scale , images with alpha channel
            if len(img.shape) != 3 or img.shape[0] < 256 or img.shape[1] < 256 or img.shape[2] != 3:
                continue

            X.append(img)
            # i is out label
            Y.append(i)

    return X , Y



def get_batch(image_list , path ,  index , batch_size):
    images =[]
    start = index*batch_size
    end = start+batch_size
    end = min(end , len(image_list))
    for i in range(start , end , 1):
        image_path = path + image_list[i]
        tmp_img = cv2.imread(image_path , -1)
        images.append(tmp_img)
    return images


def shuffle(images):
    np.random.seed(1)  #  ***********
    index = np.random.permutation(len(images))
    images = images[index]
    return images

