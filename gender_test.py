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
import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras.layers import LeakyReLU
from keras.utils import np_utils


if __name__ == "__main__":
################################### main ###########################

    #initialize parameters
    DAATA_DIR = ""
    modelFile = "gender-network.h6"

    model = load_model(os.path.join(DAATA_DIR, modelFile))
    print("model loaded")

    #momentum = 0.9,nesterov = True  
    sgd = SGD(lr = 0.1,momentum = 0.9,nesterov = True)
    model.compile(loss='categorical_crossentropy',  
              optimizer=sgd,  
              metrics=['accuracy'])

    file = open('test_picture/gender_data','r')
    filecontent = file.read()
    gender_list = filecontent.split(",")

    image_list = []
    for i in range (len(gender_list)):
        imgname = "test_picture/"+str(i)+".jpg"
        img = cv2.imread(imgname)

        image_list.append(img)
    gender_list_int= []
    for i in gender_list:
        gender_list_int.append(int(float(i)))
    gender_list_cato =np_utils.to_categorical(gender_list_int,2)

    
    length = len(gender_list)
    testX = image_list
    testY = gender_list_cato
    testX = np.array(testX)
    testY = np.array(testY)

    score = model.evaluate(testX, testY, verbose=0)  
    print('gender prediction model \nData size: ',len(testY),'\nTest accuracy:', score[1]) 


                
