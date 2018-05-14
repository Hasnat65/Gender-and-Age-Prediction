# Date: 2018-05-03
# Author: Yuehan Wang
import fnmatch
import os
import cv2     
import scipy.io as sio 
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten  
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D, MaxPooling2D 
from keras.utils import np_utils
import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras.layers import LeakyReLU


if __name__ == "__main__":
################################### main ###########################

    #initialize parameters
    DAATA_DIR = ""

    modelFile = "age-network.h6"

    model = load_model(os.path.join(DAATA_DIR, modelFile))
    print("model loaded")

    #momentum = 0.9,nesterov = True
     
    sgd = SGD(lr = 0.1,momentum = 0.9,nesterov = True)
    model.compile(loss='categorical_crossentropy',  
              optimizer=sgd,  
              metrics=['accuracy'])



    file = open('test_picture/age_data','r')
    filecontent = file.read()
    age_list = filecontent.split(",")

    image_list = []
    for i in range (len(age_list)):
        imgname = "test_picture/"+str(i)+".jpg"
        img = cv2.imread(imgname)
        img = cv2.resize(img, (64,64))
        image_list.append(img)
    age_list_0_to_99 = []
    selected_image_list = []

    for i in range (len(age_list)):
        age = int(age_list[i])
        if age  >=0 and age <100:
            age_list_0_to_99.append(age)
            selected_image_list.append(image_list[i])
    age_list_cato =np_utils.to_categorical(age_list_0_to_99,100)

    testX = selected_image_list
    testY = age_list_cato
    testX = np.array(testX)
    testY = np.array(testY)
    



    score = model.evaluate(testX, testY, verbose=0)  
    print('age prediction model \n data size: ',len(testY),'\nTest accuracy:', score[1]) 

                
