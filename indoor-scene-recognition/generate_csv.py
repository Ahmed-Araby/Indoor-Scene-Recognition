"""
it will generate CSV file for me from a file with data
"""
from os import listdir
from generate_Data import get_path_list
import pandas as pd
import cv2

# image.shape (height ,  width , channels )

SLASH = '/'
corrupted = ['indooPool_Inside_gif.jpg']
label_dict = dict()

def get_path_list(path):
    path_list = listdir(path)
    return path_list

def generate_csv_train_file(path):
    ID =[]
    image_path =[]
    labelid = []
    labelstr = []
    index = 0

    # file with 10 folders
    file_list =  get_path_list(path)

    for i in range(0 , len(file_list) , 1):
        # specific file of the 10
        class_path = path + SLASH + file_list[i]

        # images paths in 1 folder of the 10 folders
        class_path_list = get_path_list(class_path)

        # label for bunch of images
        label_dict[i] = file_list[i]

        # iterate throw the images of 1 class
        for j in range(0 , len(class_path_list) , 1):
            final_path = class_path + SLASH + class_path_list[j]

            # corrupted images
            if class_path_list[j] in corrupted:
                continue

            img = cv2.imread(final_path , -1)

            print("final_path : "  , final_path)
            # store id , final path to load the image , it's label
            ID.append(index)
            index+=1
            image_path.append(final_path)
            labelid.append(i)
            labelstr.append(label_dict[i])

    save_to_csv(ID , image_path , labelid , labelstr)
    return

def save_to_csv(ID , image_path , labelid , labelstr):
    # we have build a pandas data frame
    tmp_dict = {"id":ID , "image_path":image_path , "labelid":labelid , "labelstr":labelstr}
    Data = pd.DataFrame(tmp_dict)
    Data.to_csv("train.csv")
    return Data

"""
print("start")
generate_csv_train_file("/home/ahmedaraby/PycharmProjects/indoor-scene-recognition/train2")
print("end")
"""