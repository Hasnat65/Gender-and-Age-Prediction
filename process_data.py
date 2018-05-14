import fnmatch
import os
import cv2     
import scipy.io as sio 
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten  
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D, MaxPooling2D 
import keras


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

''' get the age and gender data from the .mat file'''
def get_data(mat_path):
    mat_data = sio.loadmat(mat_path)
    listdata = mat_data['wiki'].tolist()[0][0]
   
    age_list = []
    gender_list = []
    img_paths = []
    pic_name_list = listdata[2].tolist()[0]
    img_path_list = [a.tolist()[0] for a in pic_name_list]
    for i in range (len(pic_name_list)):
        gender = listdata[3][0][i]
        pic_path = pic_name_list[i][0]
        age = int(listdata[1][0][i]) - int(pic_path.split('_')[1].split('-')[0])
        age_list.append(age)
        gender_list.append(gender)

    gender_list_re=[]
    age_list_re = []
    img_path_list_re = []
    for i in range(len(gender_list)):
        if gender_list[i] == 1.0 or gender_list[i] == 0.0:
            gender_list_re.append(gender_list[i])
            age_list_re.append(age_list[i])
            img_path_list_re.append(img_path_list[i])
   
    

    return gender_list_re, age_list_re, img_path_list_re


#process images and delete low quality training data.
def process_image_data(img_path_list,gender_list,age_list):
    image_data_list = []
    gender_label_list = []
    image_name_list = []
    age_label_list = []

    image_list = []
    final_gender_list = []
    final_age_list = []
    final_imaga_data = []
    counter = 0
    black = np.array([0,0,0])
    white = np.array([255,255,255])
    print(len(age_list))
    counter = 0
    for i in range (len(age_list)):
        if i%2500 ==0:
            print(str(i/len(age_list)*100) +"% of the images are processed")
        img = cv2.imread(os.path.join("wiki_crop",img_path_list[i])) 
        if img is None:
            continue
        else:
            if np.equal(img[0][0],white).all() or np.equal(img[0][0],black).all() or age_list[i]>99 or age_list[i] < 0:
                continue
            else:
                img_resized = cv2.resize(img, (150,150))
                gender_label_list.append(gender_list[i])  
                age_label_list.append(age_list[i])
                imgname = "picture/"+str(counter)+".jpg"
                image_name_list.append(imgname)
                cv2.imwrite(imgname, img_resized);
                counter+=1
    text_file = open("picture/gender_data", "w")
    gender_list_string = ','.join(str(e) for e in gender_label_list)
    text_file.write(gender_list_string)
    text_file.close()

    text_file = open("picture/age_data", "w")
    age_list_string = ','.join(str(e) for e in age_label_list)
    text_file.write(age_list_string)
    text_file.close()
    print("picture / gender_data / age_data")





gender_list, age_list,img_path_list = get_data('wiki_crop/wiki.mat')
process_image_data(img_path_list,gender_list,age_list)
