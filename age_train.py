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
    batch_size = 64
    epochs = 1000
    pool_size = (2,2)

    modelFile = "age-network.h6"

    if os.path.exists(modelFile):
        model = load_model(os.path.join(DAATA_DIR, modelFile))
        print("model loaded")
    else: 
        print("build the model")
        
        model = Sequential()
        model.add(Conv2D(128, (5, 5), strides = 3, padding='same', activation="linear", input_shape=(64,64,3)))
        model.add(LeakyReLU(alpha=.01))
        model.add(Conv2D(256, (5,5),  activation="linear"))
        model.add(LeakyReLU(alpha=.01))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (5, 5), padding='same', activation="linear", ))
        model.add(LeakyReLU(alpha=.01))
        model.add(Conv2D(256, (5,5),  activation="linear"))
        model.add(LeakyReLU(alpha=.01))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(100, activation='softmax'))
     
       

    #momentum = 0.9,nesterov = True
     
    sgd = SGD(lr = 0.1,momentum = 0.9,nesterov = True)
    model.compile(loss='categorical_crossentropy',  
              optimizer=sgd,  
              metrics=['accuracy'])



    file = open('picture/age_data','r')
    filecontent = file.read()
    age_list = filecontent.split(",")

    image_list = []
    for i in range (len(age_list)):
        imgname = "picture/"+str(i)+".jpg"
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
        
    length = len(age_list_cato)
    trainX = selected_image_list[0:length*10//10]
    trainY = age_list_cato[0:length*10//10]
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    validX = selected_image_list[length*8//10:length*9//10]
    validY = age_list_cato[length*8//10:length*9//10]
    validX = np.array(validX)
    validY = np.array(validY)
    testX = selected_image_list[length*9//10:-1]
    testY = age_list_cato[length*9//10:-1]
    testX = np.array(testX)
    testY = np.array(testY)
    

    history = model.fit(trainX, trainY, batch_size=batch_size, epochs=5, verbose=1, 
                    validation_data=(testX, testY))


    model.save(os.path.join(DAATA_DIR, modelFile), overwrite = True)



    # Accuracy Curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['acc'],'r',linewidth=3.0)
    plt.plot(history.history['val_acc'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Age Accuracy Curves',fontsize=16)

    plt.show()


    score = model.evaluate(testX, testY, verbose=0)  
    print('Test score:', score[0])  
    print('Test accuracy:', score[1]) 

                
